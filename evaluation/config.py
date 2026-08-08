from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_record.config import REPO_ROOT


@dataclass(frozen=True)
class EvaluationConfig:
    """Default policy-rollout and final-position evaluation parameters."""

    policy_host: str = "10.21.22.46"
    policy_port: int = 8088
    suites: tuple[str, ...] = ("independent", "chained")
    episodes_per_suite: int = 20
    independent_tasks_path: Path = (
        REPO_ROOT / "dataset_record" / "info" / "task1" / "task_descriptions.json"
    )
    output_path: Path = REPO_ROOT / "evaluation" / "results" / "latest.json"

    # 72 chunks * 25 steps / 30 FPS = the 60 s episode horizon already used
    # by dataset_record.config.RecordConfig.
    max_chunks: int = 72
    execution_horizon: int = 25
    gripper_threshold: float = 0.5
    headless: bool = True
    real_time: bool = False

    # All five spatial relations use the same 5 cm final XY site tolerance.
    relation_tolerance_m: float = 0.050

    reset_random_seed: int = 20260806
    chained_task_seed: int = 1024
