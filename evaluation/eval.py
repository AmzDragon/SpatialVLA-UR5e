"""Evaluate final positions of two-stage UR5e tabletop rearrangement rollouts.

The independent suite uses A->B/C->D tasks from the existing recording task
file.  The chained suite generates A->B/C->A tasks.  After the rollout ends,
each subtask is judged only by the planar distance between its source center
site and requested MuJoCo relation site.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import permutations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_record.config import RecordConfig
from env import LabSimMujocoEnv, viewer_is_running
from evaluation.config import EvaluationConfig
from teleop.automated_teleop import MOVABLE_OBJECTS, PickPlaceCommand


ALL_OBJECTS = (
    "red_cube",
    "yellow_cylinder",
    "cyan_cuboid",
    "white_square_sheet",
    "black_rectangular_sheet",
)
SHEET_OBJECTS = frozenset({"white_square_sheet", "black_rectangular_sheet"})
POSITIONS = ("up", "down", "left", "right", "center")
NON_CENTER_POSITIONS = ("up", "down", "left", "right")
MOVABLE_OBJECT_NAMES = tuple(sorted(MOVABLE_OBJECTS))

DISPLAY_NAMES = {
    "red_cube": "red cube",
    "yellow_cylinder": "yellow cylinder",
    "cyan_cuboid": "cyan cuboid",
    "white_square_sheet": "white square paper",
    "black_rectangular_sheet": "black rectangular paper",
}
RELATION_PHRASES = {
    "up": "above",
    "down": "below",
    "left": "to the left of",
    "right": "to the right of",
    "center": "at the center of",
}


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    suite: str
    prompt: str
    commands: tuple[PickPlaceCommand, PickPlaceCommand]


@dataclass
class EpisodeResult:
    task_id: str
    suite: str
    prompt: str
    commands: list[dict[str, str]]
    subtasks: list[dict[str, Any]]
    subtask_1_success: bool
    subtask_2_success: bool
    double_stage_success: bool
    final_relation_errors_m: list[float]
    chunks: int
    steps: int
    termination: str
    initial_env_info: dict[str, object]
    final_env_info: dict[str, object]
    error: str | None = None


def relation_error(
    env: LabSimMujocoEnv,
    command: PickPlaceCommand,
) -> float:
    """Return final XY distance from source center to the requested relation site."""
    source_position = env.get_site_position(command.source_site)
    destination_position = env.get_site_position(command.destination_site)
    return float(np.linalg.norm(source_position[:2] - destination_position[:2]))


def load_independent_tasks(path: Path) -> list[EvaluationTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("descriptions")
    if not isinstance(items, list):
        raise ValueError(f"task file has no 'descriptions' list: {path}")

    tasks = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"description {index} must be an object")
        command_items = item.get("commands")
        if not isinstance(command_items, list) or len(command_items) != 2:
            continue
        try:
            commands = tuple(PickPlaceCommand(**command) for command in command_items)
        except TypeError as exc:
            raise ValueError(f"invalid command in description {index}") from exc
        for command in commands:
            _validate_command(command, description_index=index)

        first_objects = {commands[0].source_object, commands[0].target_object}
        second_objects = {commands[1].source_object, commands[1].target_object}
        if not first_objects.isdisjoint(second_objects):
            continue

        task_id = item.get("id", f"independent_{index + 1:03d}")
        prompt = item.get("english")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ValueError(f"description {index} has an invalid id or English prompt")
        tasks.append(
            EvaluationTask(
                task_id=task_id,
                suite="independent",
                prompt=prompt,
                commands=(commands[0], commands[1]),
            )
        )

    if not tasks:
        raise ValueError(f"no independent two-stage tasks found in {path}")
    return tasks


def _validate_command(
    command: PickPlaceCommand,
    *,
    description_index: int,
) -> None:
    if command.source_object not in MOVABLE_OBJECTS:
        raise ValueError(
            f"description {description_index} has a non-graspable source: "
            f"{command.source_object}"
        )
    if command.target_object not in ALL_OBJECTS:
        raise ValueError(
            f"description {description_index} has an unknown target: "
            f"{command.target_object}"
        )
    if command.source_object == command.target_object:
        raise ValueError(
            f"description {description_index} uses the same source and target"
        )
    if command.target_position not in POSITIONS:
        raise ValueError(
            f"description {description_index} has an unknown relation: "
            f"{command.target_position}"
        )
    if (
        command.target_position == "center"
        and command.target_object not in SHEET_OBJECTS
    ):
        raise ValueError(
            f"description {description_index} centers an object on a non-sheet target"
        )


def generate_chained_tasks(seed: int) -> list[EvaluationTask]:
    """Generate A->B then C->A tasks, where A, B and C are distinct."""
    scenarios: list[tuple[PickPlaceCommand, PickPlaceCommand]] = []
    for source_a, source_c in permutations(MOVABLE_OBJECT_NAMES, 2):
        target_candidates = (
            object_name
            for object_name in ALL_OBJECTS
            if object_name not in {source_a, source_c}
        )
        for target_b in target_candidates:
            first_positions = (
                POSITIONS if target_b in SHEET_OBJECTS else NON_CENTER_POSITIONS
            )
            for first_position, second_position in product(
                first_positions,
                NON_CENTER_POSITIONS,
            ):
                scenarios.append(
                    (
                        PickPlaceCommand(source_a, target_b, first_position),
                        PickPlaceCommand(source_c, source_a, second_position),
                    )
                )

    random.Random(seed).shuffle(scenarios)
    return [
        EvaluationTask(
            task_id=f"chained_{index + 1:03d}",
            suite="chained",
            prompt=_render_prompt(commands),
            commands=commands,
        )
        for index, commands in enumerate(scenarios)
    ]


def _render_prompt(
    commands: tuple[PickPlaceCommand, PickPlaceCommand],
) -> str:
    clauses = []
    for command in commands:
        clauses.append(
            f"place the {DISPLAY_NAMES[command.source_object]} "
            f"{RELATION_PHRASES[command.target_position]} "
            f"the {DISPLAY_NAMES[command.target_object]}"
        )
    return f"{clauses[0].capitalize()}, then {clauses[1]}."


def select_tasks(
    suites: Sequence[str],
    episodes_per_suite: int,
    cfg: EvaluationConfig,
) -> list[EvaluationTask]:
    if episodes_per_suite <= 0:
        raise ValueError("episodes_per_suite must be positive")

    selected = []
    if "independent" in suites:
        independent = load_independent_tasks(cfg.independent_tasks_path)
        if episodes_per_suite > len(independent):
            raise ValueError(
                f"requested {episodes_per_suite} independent episodes, but only "
                f"{len(independent)} are available"
            )
        selected.extend(independent[:episodes_per_suite])
    if "chained" in suites:
        chained = generate_chained_tasks(cfg.chained_task_seed)
        if episodes_per_suite > len(chained):
            raise ValueError(
                f"requested {episodes_per_suite} chained episodes, but only "
                f"{len(chained)} are available"
            )
        selected.extend(chained[:episodes_per_suite])
    return selected


def validate_config(cfg: EvaluationConfig) -> None:
    positive_values = {
        "episodes_per_suite": cfg.episodes_per_suite,
        "max_chunks": cfg.max_chunks,
        "execution_horizon": cfg.execution_horizon,
        "relation_tolerance_m": cfg.relation_tolerance_m,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"evaluation config values must be positive: {invalid}")


def execute_action_steps(
    sim_env: LabSimMujocoEnv,
    action_chunk: np.ndarray,
    *,
    execution_horizon: int,
    gripper_threshold: float,
    viewer: Any,
    real_time: bool,
) -> int:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
        raise ValueError(
            f"expected action chunk with shape (T, >=7), got {action_chunk.shape}"
        )
    if not np.all(np.isfinite(action_chunk)):
        raise ValueError("policy action chunk contains NaN or infinity")

    steps_to_execute = min(execution_horizon, action_chunk.shape[0])
    sim_env.solver.configuration.update(sim_env.data.qpos.copy())
    sim_env.solver.reset_target_to_current()
    next_tick = time.perf_counter()
    executed_steps = 0

    for action in action_chunk[:steps_to_execute]:
        if not viewer_is_running(viewer):
            break

        gripper_closed = bool(float(action[6]) >= gripper_threshold)
        sim_env.gripper_closed = gripper_closed
        sim_env.solver.configuration.update(sim_env.data.qpos.copy())
        sim_env.solver.step(np.asarray(action[:3], dtype=np.float64), scale=1.0)
        sim_env.sync_ctrl_from_qpos(
            sim_env.solver.qpos(),
            sim_env.arm_actuator_ids,
        )
        sim_env.data.ctrl[sim_env.gripper_actuator_id] = (
            sim_env.cfg.gripper_closed_ctrl
            if gripper_closed
            else sim_env.cfg.gripper_open_ctrl
        )
        sim_env.step_for_duration(sim_env.control_dt)

        if viewer is not None:
            viewer.sync()
        if real_time:
            next_tick += sim_env.control_dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))

        executed_steps += 1

    return executed_steps


def evaluate_episode(
    client: Any,
    sim_env: LabSimMujocoEnv,
    task: EvaluationTask,
    cfg: EvaluationConfig,
    *,
    viewer: Any,
) -> EpisodeResult:
    client.reset()
    sim_env.reset()
    initial_env_info = sim_env.capture_env_info()
    chunks = 0
    steps = 0
    termination = "horizon_reached"
    error: str | None = None

    try:
        for _ in range(cfg.max_chunks):
            if not viewer_is_running(viewer):
                termination = "viewer_closed"
                break
            action_chunk = client.infer_action_chunk_from_env(
                sim_env,
                prompt=task.prompt,
            )
            chunks += 1
            steps += execute_action_steps(
                sim_env,
                action_chunk,
                execution_horizon=cfg.execution_horizon,
                gripper_threshold=cfg.gripper_threshold,
                viewer=viewer,
                real_time=cfg.real_time,
            )
    except Exception as exc:
        termination = "error"
        error = f"{type(exc).__name__}: {exc}"

    final_errors = [relation_error(sim_env, command) for command in task.commands]
    subtask_successes = [
        error_m <= cfg.relation_tolerance_m for error_m in final_errors
    ]
    subtasks = [
        {
            "command": asdict(command),
            "final_relation_error_m": final_errors[index],
            "success": subtask_successes[index],
        }
        for index, command in enumerate(task.commands)
    ]
    return EpisodeResult(
        task_id=task.task_id,
        suite=task.suite,
        prompt=task.prompt,
        commands=[asdict(command) for command in task.commands],
        subtasks=subtasks,
        subtask_1_success=subtask_successes[0],
        subtask_2_success=subtask_successes[1],
        double_stage_success=all(subtask_successes),
        final_relation_errors_m=final_errors,
        chunks=chunks,
        steps=steps,
        termination=termination,
        initial_env_info=initial_env_info,
        final_env_info=sim_env.capture_env_info(),
        error=error,
    )


def summarize(results: Iterable[EpisodeResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.suite, []).append(result)

    summaries = {}
    for suite, suite_results in grouped.items():
        episode_count = len(suite_results)
        summaries[suite] = {
            "episodes": episode_count,
            "completed_without_error": sum(result.error is None for result in suite_results),
            "subtask_1_success_rate": _mean_bool(
                result.subtask_1_success for result in suite_results
            ),
            "subtask_2_success_rate": _mean_bool(
                result.subtask_2_success for result in suite_results
            ),
            "double_stage_success_rate": _mean_bool(
                result.double_stage_success for result in suite_results
            ),
        }
        if suite == "chained":
            summaries[suite]["chained_task_success_rate"] = summaries[suite][
                "double_stage_success_rate"
            ]
    return summaries


def _mean_bool(values: Iterable[bool]) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)


def write_report(
    path: Path,
    cfg: EvaluationConfig,
    results: Sequence[EpisodeResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _jsonable_config(cfg),
        "summary": summarize(results),
        "episodes": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable_config(cfg: EvaluationConfig) -> dict[str, Any]:
    result = asdict(cfg)
    for key, value in tuple(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\nEvaluation summary")
    for suite, metrics in summary.items():
        print(f"[{suite}] episodes={metrics['episodes']}")
        for key in (
            "subtask_1_success_rate",
            "subtask_2_success_rate",
            "double_stage_success_rate",
        ):
            print(f"  {key}: {_format_rate(metrics[key])}")


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def parse_args() -> argparse.Namespace:
    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate independent A->B/C->D and chained A->B/C->A policy tasks."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("all", "independent", "chained"),
        default="all",
    )
    parser.add_argument(
        "--episodes-per-suite",
        type=int,
        default=defaults.episodes_per_suite,
    )
    parser.add_argument(
        "--independent-tasks",
        type=Path,
        default=defaults.independent_tasks_path,
    )
    parser.add_argument("--host", default=defaults.policy_host)
    parser.add_argument("--port", type=int, default=defaults.policy_port)
    parser.add_argument("--max-chunks", type=int, default=defaults.max_chunks)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=defaults.execution_horizon,
    )
    parser.add_argument("--output", type=Path, default=defaults.output_path)
    parser.add_argument("--seed", type=int, default=defaults.reset_random_seed)
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show the MuJoCo viewer instead of evaluating headlessly.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep between control steps to match the configured FPS.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print selected task prompts without connecting to the policy server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = EvaluationConfig()
    suites = defaults.suites if args.suite == "all" else (args.suite,)
    cfg = replace(
        defaults,
        policy_host=args.host,
        policy_port=args.port,
        suites=suites,
        episodes_per_suite=args.episodes_per_suite,
        independent_tasks_path=args.independent_tasks,
        output_path=args.output,
        max_chunks=args.max_chunks,
        execution_horizon=args.execution_horizon,
        headless=not args.show_viewer,
        real_time=args.real_time,
        reset_random_seed=args.seed,
    )
    validate_config(cfg)
    tasks = select_tasks(cfg.suites, cfg.episodes_per_suite, cfg)

    if args.list_tasks:
        for task in tasks:
            print(f"{task.task_id} [{task.suite}] {task.prompt}")
        return

    # Imported only for an actual rollout so task generation and final-relation
    # tests do not require the lightweight openpi client package globally.
    from inference.client import RemoteUR5EInferenceClient

    record_cfg = replace(
        RecordConfig(),
        reset_random_seed=cfg.reset_random_seed,
    )
    client = RemoteUR5EInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        image_size=record_cfg.image_size,
    )
    sim_env = LabSimMujocoEnv(record_cfg)
    results: list[EpisodeResult] = []
    try:
        with sim_env.viewer_context(headless=cfg.headless) as viewer:
            for index, task in enumerate(tasks, start=1):
                print(f"[{index}/{len(tasks)}] {task.task_id}: {task.prompt}")
                result = evaluate_episode(client, sim_env, task, cfg, viewer=viewer)
                results.append(result)
                write_report(cfg.output_path, cfg, results)
                print(
                    f"  stage1={result.subtask_1_success} "
                    f"stage2={result.subtask_2_success} "
                    f"double={result.double_stage_success} "
                    f"termination={result.termination}"
                )
                if not viewer_is_running(viewer):
                    break
    finally:
        sim_env.close()

    summary = summarize(results)
    print_summary(summary)
    print(f"Report: {cfg.output_path}")


if __name__ == "__main__":
    main()
