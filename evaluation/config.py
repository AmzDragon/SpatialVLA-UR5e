from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_record.config import REPO_ROOT


@dataclass(frozen=True)
class CameraVariant:
    name: str
    translation_xyz_m: tuple[float, float, float]
    rotation_rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class TableColorVariant:
    name: str
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class LightingVariant:
    name: str
    intensity_scale: float
    rgb_scale: tuple[float, float, float]


# Episode camera training support is +/-5 cm translation per axis. Every OOD
# variant keeps rotation in range and moves one axis strictly into (5, 6) cm.
CAMERA_IN_DISTRIBUTION_VARIANTS = (
    CameraVariant("camera_id_x_positive", (0.04, 0.00, 0.00), (0.0, 2.0, -4.0)),
    CameraVariant("camera_id_x_negative", (-0.04, 0.01, 0.00), (3.0, -3.0, 4.0)),
    CameraVariant("camera_id_y_positive", (0.00, 0.04, 0.01), (-4.0, 3.0, 0.0)),
    CameraVariant("camera_id_y_negative", (0.01, -0.04, 0.00), (4.0, -2.0, 3.0)),
    CameraVariant("camera_id_pitch_positive", (0.00, -0.01, 0.04), (0.0, 4.0, 0.0)),
)

CAMERA_OOD_VARIANTS = (
    CameraVariant("camera_ood_x_positive", (0.052, 0.00, 0.00), (0.0, 2.0, -4.0)),
    CameraVariant("camera_ood_x_negative", (-0.054, 0.01, 0.00), (3.0, -3.0, 4.0)),
    CameraVariant("camera_ood_y_positive", (0.00, 0.056, 0.01), (-4.0, 3.0, 0.0)),
    CameraVariant("camera_ood_y_negative", (0.01, -0.058, 0.00), (4.0, -2.0, 3.0)),
    CameraVariant("camera_ood_z_positive", (0.00, -0.01, 0.055), (0.0, 4.0, 0.0)),
)


# ID colors refer to RecordConfig.table_color_palette. OOD colors stay close to
# their partners but are not members of that discrete training palette.
TABLE_IN_DISTRIBUTION_INDICES = (1, 3, 4, 7, 9)
TABLE_OOD_VARIANTS = (
    TableColorVariant("table_ood_purple", (0.36, 0.18, 0.52, 1.0)),
    TableColorVariant("table_ood_rose", (0.57, 0.26, 0.49, 1.0)),
    TableColorVariant("table_ood_green", (0.08, 0.43, 0.14, 1.0)),
    TableColorVariant("table_ood_brown", (0.43, 0.25, 0.18, 1.0)),
    TableColorVariant("table_ood_ochre", (0.53, 0.40, 0.14, 1.0)),
)


# Training support is intensity [0.60, 1.40] and RGB scale [0.92, 1.08].
LIGHTING_IN_DISTRIBUTION_VARIANTS = (
    LightingVariant("lighting_id_dim", 0.65, (1.00, 1.00, 1.00)),
    LightingVariant("lighting_id_bright", 1.35, (1.00, 1.00, 1.00)),
    LightingVariant("lighting_id_warm", 1.00, (1.06, 1.00, 0.94)),
    LightingVariant("lighting_id_cool", 1.00, (0.94, 1.00, 1.06)),
    LightingVariant("lighting_id_green", 1.20, (1.02, 1.06, 0.98)),
)

LIGHTING_OOD_VARIANTS = (
    LightingVariant("lighting_ood_dim", 0.52, (1.00, 1.00, 1.00)),
    LightingVariant("lighting_ood_bright", 1.48, (1.00, 1.00, 1.00)),
    LightingVariant("lighting_ood_warm", 1.00, (1.12, 1.00, 0.88)),
    LightingVariant("lighting_ood_cool", 1.00, (0.88, 1.00, 1.12)),
    LightingVariant("lighting_ood_green", 1.20, (1.02, 1.12, 0.98)),
)

COLOR_RGBA = {
    "red": (0.95, 0.22, 0.04, 1.0),
    "yellow": (0.95, 0.68, 0.02, 1.0),
    "cyan": (0.00, 0.42, 0.55, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    "black": (0.03, 0.03, 0.03, 1.0),
}

# Object order: cube, cylinder, cuboid, square paper, rectangular paper.
# Every OOD assignment is a derangement: colors and shapes were each seen, but
# none of these color-shape pairings was seen during training.
NOVEL_COLOR_ASSIGNMENTS = (
    ("yellow", "cyan", "white", "black", "red"),
    ("cyan", "white", "black", "red", "yellow"),
    ("white", "black", "red", "yellow", "cyan"),
    ("black", "red", "yellow", "cyan", "white"),
    ("yellow", "red", "black", "cyan", "white"),
)


@dataclass(frozen=True)
class EvaluationConfig:
    policy_host: str = "10.21.22.46"
    policy_port: int = 8088
    source_tasks_path: Path = (
        REPO_ROOT / "dataset_record" / "info" / "task1" / "task_descriptions.json"
    )
    cases_path: Path = REPO_ROOT / "evaluation" / "eval_cases.json"
    output_path: Path = REPO_ROOT / "evaluation" / "results" / "latest.json"
    analysis_output_dir: Path = (
        REPO_ROOT / "evaluation" / "results" / "analysis"
    )
    analysis_tolerances_m: tuple[float, float] = (0.03, 0.05)

    task_seed: int = 20260806
    language_pool_size: int = 50
    language_cases_per_split: int = 10
    factor_cases_per_distribution: int = 5
    camera_in_distribution_variants: tuple[CameraVariant, ...] = (
        CAMERA_IN_DISTRIBUTION_VARIANTS
    )
    camera_ood_variants: tuple[CameraVariant, ...] = CAMERA_OOD_VARIANTS
    table_in_distribution_indices: tuple[int, ...] = TABLE_IN_DISTRIBUTION_INDICES
    table_ood_variants: tuple[TableColorVariant, ...] = TABLE_OOD_VARIANTS
    lighting_in_distribution_variants: tuple[LightingVariant, ...] = (
        LIGHTING_IN_DISTRIBUTION_VARIANTS
    )
    lighting_ood_variants: tuple[LightingVariant, ...] = LIGHTING_OOD_VARIANTS
    novel_color_assignments: tuple[tuple[str, ...], ...] = NOVEL_COLOR_ASSIGNMENTS

    max_chunks: int = 72
    execution_horizon: int = 25
    gripper_threshold: float = 0.5
    relation_tolerance_m: float = 0.03
    headless: bool = True
    real_time: bool = False

    network_timeout_s: float = 2.0
    rtc_network_timeout_s: float = 120.0
    rtc_warmup_inferences: int = 10
    rtc_queue_threshold: int = 25
