from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TMP_DIR = str(REPO_ROOT / "tmp")#r"E:\codex_code\lab_sim\tmp"
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["TMP"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
tempfile.tempdir = TMP_DIR
from dataset_record.config import LEROBOT_FEATURES, LEROBOT_SRC, STATE_NAMES, RecordConfig
from env import LabSimMujocoEnv, StateSample, relative_rotation_euler, viewer_is_running

if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from teleop.automated_teleop import AutomatedTeleop, PickPlaceCommand, parse_task
from teleop.keyboard_teleop import KeyboardTeleop


DEFAULT_TASK_DESCRIPTIONS_PATH = (
    REPO_ROOT / "dataset_record" / "info" / "task1" / "task_descriptions.json"
)
ACTION_MODES = ("theta", "abs")


@dataclass(frozen=True)
class TaskDescription:
    description_id: str
    english: str
    commands: tuple[PickPlaceCommand, ...]


@dataclass
class RecordedEpisode:
    samples: list[StateSample]
    task: str
    description_id: str | None = None
    env_info: dict[str, object] | None = None


class RecordingHotkeys:
    VK_RETURN = 0x0D
    VK_ESCAPE = 0x1B
    VK_R = 0x52

    def __init__(self) -> None:
        self._previous = {"finish": False, "stop": False, "discard": False}
        self._user32 = ctypes.windll.user32 if sys.platform == "win32" else None

    def poll(self) -> dict[str, bool]:
        current = {
            "finish": self._pressed(self.VK_RETURN),
            "stop": self._pressed(self.VK_ESCAPE),
            "discard": self._pressed(self.VK_R),
        }
        rising_edges = {
            key: pressed and not self._previous[key] for key, pressed in current.items()
        }
        self._previous = current
        return rising_edges

    def _pressed(self, vk_code: int) -> bool:
        if self._user32 is None:
            return False
        return bool(self._user32.GetAsyncKeyState(vk_code) & 0x8000)


def parse_args() -> argparse.Namespace:
    cfg = RecordConfig()
    parser = argparse.ArgumentParser(
        description="Record keyboard or automated MuJoCo episodes as a LeRobot dataset."
    )
    parser.add_argument("--root", type=Path, default=cfg.dataset_root)
    parser.add_argument(
        "--teleop",
        choices=("keyboard", "autoteleop"),
        default="keyboard",
        help="Control source used to collect episodes.",
    )
    parser.add_argument(
        "--task",
        default=cfg.task,
        help="Language instruction stored for every keyboard episode.",
    )
    parser.add_argument(
        "--task-descriptions",
        type=Path,
        default=DEFAULT_TASK_DESCRIPTIONS_PATH,
        help="JSON descriptions used by autoteleop.",
    )
    parser.add_argument("--num-episodes", type=int, default=cfg.num_episodes)
    parser.add_argument("--episode-time-s", type=float, default=cfg.episode_time_s)
    parser.add_argument("--fps", type=int, default=cfg.fps)
    parser.add_argument(
        "--spatial-episode-rd",
        action="store_true",
        help="Enable per-episode external-camera pose and table-height RD.",
    )
    parser.add_argument(
        "--spatial-frame-rd",
        action="store_true",
        help="Enable per-frame external-camera pose RD.",
    )
    parser.add_argument(
        "--appearance-rd",
        action="store_true",
        help=(
            "Enable per-frame table-color, lighting, and floor-material RD."
        ),
    )
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default="theta",
        help="Action label representation: theta is delta pose, abs is next absolute pose.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        RecordConfig(),
        dataset_root=args.root,
        task=args.task,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s,
        fps=args.fps,
        spatial_episode_rd_enabled=args.spatial_episode_rd,
        spatial_frame_rd_enabled=args.spatial_frame_rd,
        appearance_rd_enabled=args.appearance_rd,
    )

    raw_episodes = collect_episodes(
        cfg,
        teleop=args.teleop,
        task_descriptions_path=args.task_descriptions,
        headless=args.headless,
    )
    saved = write_lerobot_dataset(
        raw_episodes,
        cfg,
        overwrite=args.overwrite,
        action_mode=args.action_mode,
    )
    print(f"Saved {saved} episode(s) to {cfg.dataset_root}")


def collect_episodes(
    cfg: RecordConfig,
    *,
    teleop: str = "keyboard",
    task_descriptions_path: Path = DEFAULT_TASK_DESCRIPTIONS_PATH,
    headless: bool = False,
) -> Iterable[RecordedEpisode]:
    if teleop == "keyboard":
        return collect_keyboard_episodes(cfg, headless=headless)
    if teleop == "autoteleop":
        return collect_automated_episodes(
            cfg,
            task_descriptions_path=task_descriptions_path,
            headless=headless,
        )
    raise ValueError(f"unsupported teleop mode: {teleop}")


def collect_keyboard_episodes(
    cfg: RecordConfig,
    *,
    headless: bool = False,
) -> list[RecordedEpisode]:
    sim_env = LabSimMujocoEnv(cfg)
    teleop = KeyboardTeleop(hz=cfg.fps, gripper_closed=sim_env.initial_gripper_closed)
    hotkeys = RecordingHotkeys()

    raw_episodes: list[RecordedEpisode] = []
    sim_env.reset()

    print("Controls: arrows/PageUp/PageDown move XYZ, backslash toggles gripper.")
    print("Recording: Enter starts episode, Enter again ends episode, R discards current episode, Esc stops recording.")

    try:
        with sim_env.viewer_context(headless=headless) as viewer:
            episode_idx = 0
            while episode_idx < cfg.num_episodes:
                sim_env.reset()
                teleop.reset()
                env_info = sim_env.capture_env_info()

                should_start, stop_requested = _wait_for_episode_start(cfg, hotkeys, viewer, episode_idx)
                if stop_requested or not should_start or not viewer_is_running(viewer):
                    break

                samples, stop_requested, discard_requested = _collect_one_episode(
                    cfg,
                    sim_env,
                    teleop,
                    hotkeys,
                    viewer,
                    episode_idx,
                )

                if discard_requested:
                    print(f"Discarded episode {episode_idx}.")
                    if stop_requested:
                        break
                    continue

                if len(samples) >= 2:
                    raw_episodes.append(
                        RecordedEpisode(
                            samples=samples,
                            task=cfg.task,
                            env_info=env_info,
                        )
                    )
                    print(f"Captured episode {len(raw_episodes) - 1}: {len(samples)} raw frames.")
                else:
                    print(f"Skipped episode {episode_idx}: not enough frames.")

                if stop_requested or not viewer_is_running(viewer):
                    break
                episode_idx += 1
    finally:
        teleop.stop()
        sim_env.close()

    return raw_episodes


def load_task_descriptions(path: Path) -> list[TaskDescription]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("descriptions")
    if not isinstance(items, list) or not items:
        raise ValueError(f"description file has no non-empty 'descriptions' list: {path}")

    descriptions = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"description {index} must be an object")

        description_id = item.get("id")
        english = item.get("english")
        command_items = item.get("commands")
        if not isinstance(description_id, str) or not description_id.strip():
            raise ValueError(f"description {index} has no valid id")
        if not isinstance(english, str) or not english.strip():
            raise ValueError(f"description {description_id} has no English instruction")
        if not isinstance(command_items, list) or not command_items:
            raise ValueError(f"description {description_id} has no commands")

        try:
            expected_commands = tuple(
                PickPlaceCommand(**command) for command in command_items
            )
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"description {description_id} contains an invalid command"
            ) from exc

        parsed_commands = tuple(parse_task(english))
        if parsed_commands != expected_commands:
            raise ValueError(
                f"English description does not match commands for {description_id}: "
                f"parsed={parsed_commands}, expected={expected_commands}"
            )
        descriptions.append(
            TaskDescription(
                description_id=description_id,
                english=english,
                commands=parsed_commands,
            )
        )

    return descriptions


def collect_automated_episodes(
    cfg: RecordConfig,
    *,
    task_descriptions_path: Path,
    headless: bool = False,
) -> Iterator[RecordedEpisode]:
    descriptions = load_task_descriptions(task_descriptions_path)
    if cfg.num_episodes > len(descriptions):
        raise ValueError(
            f"requested {cfg.num_episodes} episodes, but {task_descriptions_path} "
            f"contains only {len(descriptions)} descriptions"
        )

    selected_descriptions = descriptions[: cfg.num_episodes]
    sim_env = LabSimMujocoEnv(cfg)
    automated_teleop = AutomatedTeleop(sim_env)

    try:
        with sim_env.viewer_context(headless=headless) as viewer:
            for index, description in enumerate(selected_descriptions, start=1):
                if not viewer_is_running(viewer):
                    break

                print(
                    f"\n[{index}/{len(selected_descriptions)}] "
                    f"{description.description_id}: {description.english}",
                    flush=True,
                )
                samples: list[StateSample] = []
                env_info: dict[str, object] | None = None

                def capture_step() -> None:
                    samples.append(sim_env.capture_state())

                try:
                    sim_env.reset()
                    env_info = sim_env.capture_env_info()
                    print(
                        f"Automatically started recording "
                        f"{description.description_id}."
                    )
                    samples.append(sim_env.capture_state())
                    automated_teleop.run(
                        list(description.commands),
                        viewer=viewer,
                        step_callback=capture_step,
                    )
                except Exception as exc:
                    if not viewer_is_running(viewer):
                        print("Viewer closed; stopping automated recording.")
                        break
                    print(
                        f"Skipped {description.description_id} after automated "
                        f"recording error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    samples.clear()
                    continue

                print(
                    f"Automatically finished recording {description.description_id}."
                )

                episode = RecordedEpisode(
                    samples=samples,
                    task=description.english,
                    description_id=description.description_id,
                    env_info=env_info,
                )
                print(
                    f"Captured {description.description_id}: {len(samples)} raw frames."
                )
                yield episode
                del episode, samples
    finally:
        sim_env.close()


def _wait_for_episode_start(
    cfg: RecordConfig,
    hotkeys: RecordingHotkeys,
    viewer,
    episode_idx: int,
) -> tuple[bool, bool]:
    print(f"Ready for episode {episode_idx}. Press Enter to start recording, Esc to stop.")
    control_dt = 1.0 / float(cfg.fps)
    next_tick = time.perf_counter()

    while viewer_is_running(viewer):
        events = hotkeys.poll()
        if events["stop"]:
            return False, True
        if events["finish"]:
            print(f"Started episode {episode_idx}. Press Enter again to finish this episode.")
            return True, False

        if viewer is not None:
            viewer.sync()

        next_tick += control_dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    return False, False


def _collect_one_episode(
    cfg: RecordConfig,
    sim_env: LabSimMujocoEnv,
    teleop: KeyboardTeleop,
    hotkeys: RecordingHotkeys,
    viewer,
    episode_idx: int,
) -> tuple[list[StateSample], bool, bool]:
    control_dt = 1.0 / float(cfg.fps)
    next_tick = time.perf_counter()
    episode_start = next_tick
    samples = [sim_env.capture_state()]

    print(f"Recording episode {episode_idx} for up to {cfg.episode_time_s:.1f}s...")
    stop_requested = False
    discard_requested = False

    while viewer_is_running(viewer):
        events = hotkeys.poll()
        if events["stop"]:
            stop_requested = True
            break
        if events["finish"]:
            break
        if events["discard"]:
            discard_requested = True
            break
        if time.perf_counter() - episode_start >= cfg.episode_time_s:
            break

        step_xyz = teleop.update()
        gripper_closed = teleop.get_gripper_closed()
        sim_env.apply_teleop_command(step_xyz, gripper_closed=gripper_closed)
        sim_env.step_for_duration(control_dt)

        samples.append(sim_env.capture_state())

        if viewer is not None:
            viewer.sync()

        next_tick += control_dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    return samples, stop_requested, discard_requested


def write_lerobot_dataset(
    raw_episodes: Iterable[RecordedEpisode],
    cfg: RecordConfig,
    *,
    overwrite: bool = False,
    action_mode: str = "theta",
) -> int:
    root = cfg.dataset_root
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{root} already exists. Pass --overwrite to replace it, or choose --root."
            )
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        root=root,
        robot_type=cfg.robot_type,
        features=lerobot_features_for_action_mode(action_mode),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    saved = 0
    manifest_entries = []
    try:
        for episode_idx, raw_episode in enumerate(raw_episodes):
            processed = process_episode(
                raw_episode.samples,
                cfg,
                action_mode=action_mode,
            )
            if not processed:
                print(f"Skipped episode {episode_idx}: no useful motion or gripper change.")
                del processed, raw_episode
                continue

            for state, action in processed:
                frame = {
                    "observation.state": state.state,
                    "action": action,
                    "task": raw_episode.task,
                }
                frame.update(state.images)
                dataset.add_frame(frame)
            dataset.save_episode()
            frame_domain_randomization = compress_frame_domain_randomization(
                processed
            )
            manifest_entries.append(
                {
                    "episode_index": saved,
                    "description_id": raw_episode.description_id,
                    "task": raw_episode.task,
                    "env_info": raw_episode.env_info,
                    "frame_domain_randomization": frame_domain_randomization,
                    "action_mode": action_mode,
                }
            )
            saved += 1
            print(f"Saved episode {saved - 1}: {len(processed)} frames.")
            del processed, raw_episode
    finally:
        try:
            close_episodes = getattr(raw_episodes, "close", None)
            if close_episodes is not None:
                close_episodes()
        finally:
            dataset.finalize()

    manifest_path = root / "meta" / "recording_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(entry, ensure_ascii=False) + "\n"
            for entry in manifest_entries
        ),
        encoding="utf-8",
    )

    return saved


def compress_frame_domain_randomization(
    processed: list[tuple[StateSample, np.ndarray]],
) -> dict[str, list[dict[str, object]]]:
    component_events: dict[str, list[dict[str, object]]] = {}
    for component in (
        "camera",
        "table_color",
        "lighting",
        "floor_material",
    ):
        events = []
        previous_state: object = object()
        for frame_index, (sample, _) in enumerate(processed):
            component_state = sample.domain_randomization.get(component)
            if component_state is None or component_state == previous_state:
                continue
            event = {"frame_index": frame_index}
            event.update(deepcopy(component_state))
            events.append(event)
            previous_state = component_state
        if events:
            component_events[component] = events
    return component_events


def process_episode(
    samples: list[StateSample],
    cfg: RecordConfig,
    *,
    action_mode: str = "theta",
) -> list[tuple[StateSample, np.ndarray]]:
    kept = trim_idle_edges(samples, cfg)
    if len(kept) < 2:
        return []

    processed: list[tuple[StateSample, np.ndarray]] = []
    for idx, sample in enumerate(kept):
        if idx == len(kept) - 1:
            action = _last_frame_action(sample, action_mode, cfg)
        else:
            action = _state_action(sample, kept[idx + 1], action_mode, cfg)
        processed.append((sample, action))
    return processed


def lerobot_features_for_action_mode(action_mode: str) -> dict:
    features = deepcopy(LEROBOT_FEATURES)
    if action_mode == "abs":
        features["action"]["names"] = STATE_NAMES
    return features


def trim_idle_edges(samples: list[StateSample], cfg: RecordConfig) -> list[StateSample]:
    if len(samples) < 2:
        return []

    first_useful_transition = None
    for idx in range(len(samples) - 1):
        if _is_useful_transition(samples[idx], samples[idx + 1], cfg):
            first_useful_transition = idx
            break

    if first_useful_transition is None:
        return []

    last_useful_transition = first_useful_transition
    for idx in range(len(samples) - 2, -1, -1):
        if _is_useful_transition(samples[idx], samples[idx + 1], cfg):
            last_useful_transition = idx
            break

    first_useful_frame = first_useful_transition + 1
    last_useful_frame = last_useful_transition + 1
    start = max(0, first_useful_frame - cfg.trim_context_frames)
    end = min(len(samples), last_useful_frame + cfg.trim_context_frames + 1)
    return samples[start:end]


def _is_useful_transition(
    current: StateSample,
    next_sample: StateSample,
    cfg: RecordConfig,
) -> bool:
    position_delta = float(np.linalg.norm(next_sample.state[:3] - current.state[:3]))
    rotation_delta = float(
        np.linalg.norm(relative_rotation_euler(current.rotation_matrix, next_sample.rotation_matrix))
    )
    gripper_changed = bool(next_sample.state[6] != current.state[6])
    return (
        position_delta > cfg.idle_translation_eps_m
        or rotation_delta > cfg.idle_rotation_eps_rad
        or gripper_changed
    )


def _state_delta_action(
    current: StateSample,
    next_sample: StateSample,
    cfg: RecordConfig,
) -> np.ndarray:
    action = np.zeros((7,), dtype=np.float32)
    action[:3] = next_sample.state[:3] - current.state[:3]
    action[3:6] = relative_rotation_euler(
        current.rotation_matrix,
        next_sample.rotation_matrix,
    ).astype(np.float32)
    action[6] = 1.0 if next_sample.state[6] > cfg.gripper_closed_eps else 0.0
    return action


def _state_abs_action(next_sample: StateSample, cfg: RecordConfig) -> np.ndarray:
    action = next_sample.state.copy().astype(np.float32, copy=False)
    action[6] = 1.0 if next_sample.state[6] > cfg.gripper_closed_eps else 0.0
    return action


def _state_action(
    current: StateSample,
    next_sample: StateSample,
    action_mode: str,
    cfg: RecordConfig,
) -> np.ndarray:
    if action_mode == "theta":
        return _state_delta_action(current, next_sample, cfg)
    if action_mode == "abs":
        return _state_abs_action(next_sample, cfg)
    raise ValueError(f"unsupported action mode: {action_mode!r}")


def _last_frame_action(
    sample: StateSample,
    action_mode: str,
    cfg: RecordConfig,
) -> np.ndarray:
    if action_mode == "theta":
        return np.zeros((7,), dtype=np.float32)
    if action_mode == "abs":
        return _state_abs_action(sample, cfg)
    raise ValueError(f"unsupported action mode: {action_mode!r}")


if __name__ == "__main__":
    main()
