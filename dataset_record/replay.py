from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pyarrow.parquet as pq
from mink import SE3


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_record.config import RecordConfig
from env import LabSimMujocoEnv, viewer_is_running


def parse_args() -> argparse.Namespace:
    cfg = RecordConfig()
    parser = argparse.ArgumentParser(description="Replay one recorded episode.")
    parser.add_argument("--root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--episode", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_episode(root=args.root, episode_index=args.episode)


def replay_episode(*, root: Path, episode_index: int) -> None:
    root = root.resolve()
    info = load_info(root)
    episode = load_episode_metadata(root, episode_index)
    manifest = load_manifest_entry(root, episode_index)
    actions = load_actions(root, info, episode, episode_index)
    action_mode = manifest.get("action_mode", "theta")
    frame_rd_events = index_frame_domain_randomization_events(
        manifest.get("frame_domain_randomization", {})
    )

    cfg = replace(RecordConfig(), dataset_root=root, fps=int(info["fps"]))
    sim_env = LabSimMujocoEnv(cfg)
    try:
        sim_env.reset()
        apply_env_info(sim_env, manifest["env_info"])

        with sim_env.viewer_context() as viewer:
            sim_env.hold(0.8, viewer=viewer)
            sim_env.solver.configuration.update(sim_env.data.qpos.copy())
            sim_env.solver.reset_target_to_current()
            next_tick = time.perf_counter()

            current_frame_rd: dict[str, object] = {}
            for frame_index, action in enumerate(actions):
                if not viewer_is_running(viewer):
                    break
                for component, state in frame_rd_events.get(
                    frame_index, []
                ):
                    current_frame_rd[component] = state
                if current_frame_rd:
                    sim_env.apply_frame_domain_randomization_state(
                        current_frame_rd
                    )
                execute_action(sim_env, action, action_mode)
                viewer.sync()
                next_tick += sim_env.control_dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        sim_env.close()


def execute_action(
    sim_env: LabSimMujocoEnv,
    action: list[float],
    action_mode: str,
) -> None:
    if action_mode == "theta":
        execute_theta_action(sim_env, action)
        return
    if action_mode == "abs":
        execute_abs_action(sim_env, action)
        return
    raise ValueError(f"unsupported action mode: {action_mode!r}")


def execute_theta_action(sim_env: LabSimMujocoEnv, action: list[float]) -> None:
    delta_xyz = np.asarray(action[:3], dtype=np.float64)
    gripper_closed = bool(float(action[6]) >= 0.5)

    sim_env.gripper_closed = gripper_closed
    sim_env.solver.configuration.update(sim_env.data.qpos.copy())
    sim_env.solver.step(delta_xyz, scale=1.0)
    sim_env.sync_ctrl_from_qpos(sim_env.solver.qpos(), sim_env.arm_actuator_ids)
    sim_env.data.ctrl[sim_env.gripper_actuator_id] = (
        sim_env.cfg.gripper_closed_ctrl
        if gripper_closed
        else sim_env.cfg.gripper_open_ctrl
    )
    sim_env.step_for_duration(sim_env.control_dt)


def execute_abs_action(sim_env: LabSimMujocoEnv, action: list[float]) -> None:
    target_xyz = np.asarray(action[:3], dtype=np.float64)
    gripper_closed = bool(float(action[6]) >= 0.5)

    sim_env.gripper_closed = gripper_closed
    sim_env.solver.configuration.update(sim_env.data.qpos.copy())
    current_target = sim_env.solver.get_current_pose()
    sim_env.solver.set_target(
        SE3.from_translation(target_xyz) @ SE3.from_rotation(
            current_target.rotation()
        )
    )
    sim_env.solver.solve_step()
    sim_env.sync_ctrl_from_qpos(sim_env.solver.qpos(), sim_env.arm_actuator_ids)
    sim_env.data.ctrl[sim_env.gripper_actuator_id] = (
        sim_env.cfg.gripper_closed_ctrl
        if gripper_closed
        else sim_env.cfg.gripper_open_ctrl
    )
    sim_env.step_for_duration(sim_env.control_dt)


def apply_env_info(sim_env: LabSimMujocoEnv, env_info: dict[str, Any]) -> None:
    domain_randomization = env_info.get("domain_randomization")
    if isinstance(domain_randomization, dict):
        sim_env.apply_episode_domain_randomization_info(domain_randomization)

    for pose in env_info["objects"].values():
        sim_env.set_freejoint_pose(
            pose["freejoint"],
            position=np.asarray(pose["position"], dtype=np.float64),
            quaternion=np.asarray(pose["quaternion"], dtype=np.float64),
        )
    sim_env.data.qvel[:] = 0.0
    mujoco.mj_forward(sim_env.model, sim_env.data)
    sim_env.sync_solver_to_data()


def index_frame_domain_randomization_events(
    components: object,
) -> dict[int, list[tuple[str, dict[str, object]]]]:
    indexed: dict[int, list[tuple[str, dict[str, object]]]] = {}
    if not isinstance(components, dict):
        return indexed

    for component, events in components.items():
        if not isinstance(component, str) or not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict) or "frame_index" not in event:
                continue
            frame_index = int(event["frame_index"])
            state = {
                key: value
                for key, value in event.items()
                if key != "frame_index"
            }
            indexed.setdefault(frame_index, []).append((component, state))
    return indexed


def load_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def load_episode_metadata(root: Path, episode_index: int) -> dict[str, Any]:
    columns = ["episode_index", "data/chunk_index", "data/file_index"]
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            path,
            columns=columns,
            filters=[("episode_index", "=", episode_index)],
        )
        if table.num_rows:
            return table.to_pylist()[0]
    raise IndexError(f"episode {episode_index} not found")


def load_manifest_entry(root: Path, episode_index: int) -> dict[str, Any]:
    with (root / "meta" / "recording_manifest.jsonl").open(
        "r",
        encoding="utf-8",
    ) as manifest_file:
        for line in manifest_file:
            entry = json.loads(line)
            if int(entry["episode_index"]) == episode_index:
                return entry
    raise IndexError(f"episode {episode_index} not found in manifest")


def load_actions(
    root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    episode_index: int,
) -> list[list[float]]:
    data_path = root / info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    table = pq.read_table(
        data_path,
        columns=["frame_index", "action"],
        filters=[("episode_index", "=", episode_index)],
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    return [row["action"] for row in rows]


if __name__ == "__main__":
    main()
