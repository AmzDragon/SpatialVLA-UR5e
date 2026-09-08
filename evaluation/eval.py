"""Run the fixed 60+60 independent and chained evaluation suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_record.config import RecordConfig
from env import LabSimMujocoEnv, viewer_is_running
from evaluation.cases import load_or_create_cases
from evaluation.config import EvaluationConfig
from evaluation.scene import EvaluationScene


def relation_error(env: LabSimMujocoEnv, command: dict[str, str]) -> float:
    source = env.get_site_position(f"{command['source_object']}_center_site")
    destination = env.get_site_position(
        f"{command['target_object']}_{command['target_position']}_site"
    )
    return float(np.linalg.norm(source[:2] - destination[:2]))


def evaluate_case(
    client: Any,
    env: LabSimMujocoEnv,
    scene: EvaluationScene,
    case: dict[str, Any],
    cfg: EvaluationConfig,
    *,
    viewer: Any,
    rtc: bool,
    rtc_warmup: bool,
) -> dict[str, Any]:
    scene.reset(case)
    initial_env_info = env.capture_env_info()
    chunks = 0
    steps = 0
    termination = "horizon_reached"
    error = None

    try:
        if rtc:
            chunks, steps, termination = _rollout_rtc(
                client,
                env,
                case["prompt"],
                cfg,
                viewer=viewer,
                warmup=rtc_warmup,
            )
        else:
            chunks, steps, termination = _rollout_standard(
                client,
                env,
                case["prompt"],
                cfg,
                viewer=viewer,
            )
    except Exception as exc:
        termination = "error"
        error = f"{type(exc).__name__}: {exc}"

    final_errors = [relation_error(env, command) for command in case["commands"]]
    successes = [value <= cfg.relation_tolerance_m for value in final_errors]
    subtasks = [
        {
            "command": command,
            "final_relation_error_m": final_errors[index],
            "success": successes[index],
        }
        for index, command in enumerate(case["commands"])
    ]
    return {
        "case_id": case["id"],
        "suite": case["suite"],
        "group": case["group"],
        "distribution": case["distribution"],
        "comparison_pair": case.get("comparison_pair"),
        "base_case_id": case.get("base_case_id"),
        "paired_case_id": case.get("paired_case_id"),
        "source_task_id": case.get("source_task_id"),
        "prompt": case["prompt"],
        "commands": case["commands"],
        "scene": case["scene"],
        "subtasks": subtasks,
        "subtask_1_success": successes[0],
        "subtask_2_success": successes[1],
        "double_stage_success": all(successes),
        "final_relation_errors_m": final_errors,
        "chunks": chunks,
        "steps": steps,
        "termination": termination,
        "initial_env_info": initial_env_info,
        "final_env_info": env.capture_env_info(),
        "error": error,
    }


def _rollout_standard(
    client: Any,
    env: LabSimMujocoEnv,
    prompt: str,
    cfg: EvaluationConfig,
    *,
    viewer: Any,
) -> tuple[int, int, str]:
    from inference.client import execute_action_steps

    client.reset()
    chunks = 0
    steps = 0
    for _ in range(cfg.max_chunks):
        if not viewer_is_running(viewer):
            return chunks, steps, "viewer_closed"
        actions = client.infer_action_chunk_from_env(env, prompt=prompt)
        chunks += 1
        steps += execute_action_steps(
            env,
            actions,
            execution_horizon=cfg.execution_horizon,
            gripper_threshold=cfg.gripper_threshold,
            viewer=viewer,
            real_time=cfg.real_time,
        )
    return chunks, steps, "horizon_reached"


def _rollout_rtc(
    client: Any,
    env: LabSimMujocoEnv,
    prompt: str,
    cfg: EvaluationConfig,
    *,
    viewer: Any,
    warmup: bool,
) -> tuple[int, int, str]:
    from inference.client import capture_ur5e_observation
    from inference.rtc import RTCActionQueue, execute_sim_action

    client.remote.reset()
    observation = capture_ur5e_observation(env, prompt=prompt)
    if warmup:
        initial_actions = client.warmup(
            observation,
            num_inferences=cfg.rtc_warmup_inferences,
        )
    else:
        initial_actions = client.seed(observation)

    queue = RTCActionQueue(action_horizon=client.action_horizon)
    queue.replace(initial_actions)
    env.sync_solver_to_data()

    completed_chunks = 0
    next_tick = time.perf_counter()
    while viewer_is_running(viewer) and completed_chunks < cfg.max_chunks:
        if client.poll(queue) is not None:
            completed_chunks += 1

        if completed_chunks < cfg.max_chunks and client.should_request(queue):
            observation = capture_ur5e_observation(env, prompt=prompt)
            client.request(observation, queue)

        action = queue.pop()
        if action is None:
            env.step_for_duration(env.control_dt)
        else:
            execute_sim_action(
                env,
                action,
                gripper_threshold=cfg.gripper_threshold,
            )

        if viewer is not None:
            viewer.sync()
        next_tick += env.control_dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    termination = (
        "horizon_reached"
        if completed_chunks >= cfg.max_chunks
        else "viewer_closed"
    )
    return completed_chunks, queue.total_control_steps, termination


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    suites = sorted({row["suite"] for row in rows})
    return {
        "overall": _metrics(rows),
        "by_suite": {
            suite: _metrics([row for row in rows if row["suite"] == suite])
            for suite in suites
        },
        "by_group": {
            suite: {
                group: _metrics(
                    [
                        row
                        for row in rows
                        if row["suite"] == suite and row["group"] == group
                    ]
                )
                for group in sorted(
                    {
                        row["group"]
                        for row in rows
                        if row["suite"] == suite
                    }
                )
            }
            for suite in suites
        },
        "by_distribution": {
            suite: {
                group: {
                    distribution: _metrics(
                        [
                            row
                            for row in rows
                            if row["suite"] == suite
                            and row["group"] == group
                            and row["distribution"] == distribution
                        ]
                    )
                    for distribution in sorted(
                        {
                            row["distribution"]
                            for row in rows
                            if row["suite"] == suite and row["group"] == group
                        }
                    )
                }
                for group in sorted(
                    {row["group"] for row in rows if row["suite"] == suite}
                )
            }
            for suite in suites
        },
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "episodes": count,
        "completed_without_error": sum(row["error"] is None for row in rows),
        "subtask_1_success_rate": _rate(
            row["subtask_1_success"] for row in rows
        ),
        "subtask_2_success_rate": _rate(
            row["subtask_2_success"] for row in rows
        ),
        "double_stage_success_rate": _rate(
            row["double_stage_success"] for row in rows
        ),
    }


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_report(
    cfg: EvaluationConfig,
    results: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    config = asdict(cfg)
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    payload = {
        "mode": mode,
        "cases_path": str(cfg.cases_path),
        "config": config,
        "summary": summarize(results),
        "episodes": results,
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def print_summary(summary: dict[str, Any]) -> None:
    print("\nEvaluation summary")
    for suite, metrics in summary["by_suite"].items():
        print(
            f"[{suite}] n={metrics['episodes']} "
            f"stage1={metrics['subtask_1_success_rate']:.1%} "
            f"stage2={metrics['subtask_2_success_rate']:.1%} "
            f"double={metrics['double_stage_success_rate']:.1%}"
        )
        for group, group_metrics in summary["by_group"][suite].items():
            print(
                f"  {group}: n={group_metrics['episodes']}, "
                f"double={group_metrics['double_stage_success_rate']:.1%}"
            )
            distributions = summary["by_distribution"][suite][group]
            if len(distributions) > 1:
                for distribution, distribution_metrics in distributions.items():
                    print(
                        f"    {distribution}: "
                        f"n={distribution_metrics['episodes']}, "
                        f"double="
                        f"{distribution_metrics['double_stage_success_rate']:.1%}"
                    )


def parse_args() -> argparse.Namespace:
    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate fixed 60+60 A->B/C->D and A->B/C->A suites."
    )
    parser.add_argument(
        "--suite",
        choices=("all", "independent", "chained"),
        default="all",
    )
    parser.add_argument("--rtc", action="store_true", help="Use real-time chunking.")
    parser.add_argument("--host", default=defaults.policy_host)
    parser.add_argument("--port", type=int, default=defaults.policy_port)
    parser.add_argument("--source-tasks", type=Path, default=defaults.source_tasks_path)
    parser.add_argument("--cases", type=Path, default=defaults.cases_path)
    parser.add_argument("--output", type=Path, default=defaults.output_path)
    parser.add_argument("--seed", type=int, default=defaults.task_seed)
    parser.add_argument("--max-chunks", type=int, default=defaults.max_chunks)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=defaults.execution_horizon,
    )
    parser.add_argument(
        "--tolerance-cm",
        type=float,
        default=100.0 * defaults.relation_tolerance_m,
    )
    parser.add_argument(
        "--rtc-warmup-inferences",
        type=int,
        default=defaults.rtc_warmup_inferences,
    )
    parser.add_argument(
        "--regenerate-cases",
        action="store_true",
        help="Overwrite the persisted cases JSON using the configured seed.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Create/load and print cases without connecting to the policy server.",
    )
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run standard chunking at simulation FPS; RTC is always real-time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = EvaluationConfig()
    cfg = replace(
        defaults,
        policy_host=args.host,
        policy_port=args.port,
        source_tasks_path=args.source_tasks,
        cases_path=args.cases,
        output_path=args.output,
        task_seed=args.seed,
        max_chunks=args.max_chunks,
        execution_horizon=args.execution_horizon,
        relation_tolerance_m=args.tolerance_cm / 100.0,
        rtc_warmup_inferences=args.rtc_warmup_inferences,
        headless=not args.show_viewer,
        real_time=args.real_time,
    )
    record_cfg = replace(
        RecordConfig(),
        reset_random_seed=cfg.task_seed,
        spatial_episode_rd_enabled=False,
        spatial_frame_rd_enabled=False,
        appearance_rd_enabled=False,
    )

    env = LabSimMujocoEnv(record_cfg)
    try:
        payload = load_or_create_cases(
            cfg,
            record_cfg,
            env.sample_reset_object_poses,
            regenerate=args.regenerate_cases,
        )
        print(f"Evaluation cases ready: {cfg.cases_path}")
        cases = [
            case
            for case in payload["cases"]
            if args.suite == "all" or case["suite"] == args.suite
        ]

        if args.list_cases:
            for case in cases:
                changes = ", ".join(case["scene"]) or "nominal"
                print(
                    f"{case['id']} [{case['group']}; "
                    f"{case['distribution']}; {changes}] "
                    f"{case['prompt']}"
                )
            return

        from inference.client import RemoteUR5EInferenceClient

        if args.rtc:
            from inference.rtc import RTCRemoteClient

            remote = RemoteUR5EInferenceClient(
                host=cfg.policy_host,
                port=cfg.policy_port,
                image_size=record_cfg.image_size,
                timeout_s=cfg.rtc_network_timeout_s,
            )
            client = RTCRemoteClient(
                remote,
                fps=record_cfg.fps,
                queue_threshold=cfg.rtc_queue_threshold,
            )
            mode = "rtc"
        else:
            client = RemoteUR5EInferenceClient(
                host=cfg.policy_host,
                port=cfg.policy_port,
                image_size=record_cfg.image_size,
                timeout_s=cfg.network_timeout_s,
            )
            mode = "standard"

        results: list[dict[str, Any]] = []
        rtc_warmed = False
        scene = EvaluationScene(env)
        try:
            with env.viewer_context(headless=cfg.headless) as viewer:
                for index, case in enumerate(cases, start=1):
                    print(f"[{index}/{len(cases)}] {case['id']}: {case['prompt']}")
                    result = evaluate_case(
                        client,
                        env,
                        scene,
                        case,
                        cfg,
                        viewer=viewer,
                        rtc=args.rtc,
                        rtc_warmup=not rtc_warmed,
                    )
                    if args.rtc:
                        rtc_warmed = True
                    results.append(result)
                    write_report(cfg, results, mode=mode)
                    print(
                        f"  stage1={result['subtask_1_success']} "
                        f"stage2={result['subtask_2_success']} "
                        f"double={result['double_stage_success']} "
                        f"termination={result['termination']}"
                    )
                    if result["error"]:
                        print(f"  error={result['error']}")
                    if not viewer_is_running(viewer) or (
                        args.rtc and result["error"]
                    ):
                        break
        finally:
            client.close()

        summary = summarize(results)
        print_summary(summary)
        print(f"Report: {cfg.output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
