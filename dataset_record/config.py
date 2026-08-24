from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "thirdparty" / "lerobot" / "src"


@dataclass(frozen=True)
class ResetObjectConfig:
    name: str
    z: float
    half_size_xy: tuple[float, float]
    rotation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def clearance_radius(self) -> float:
        half_x, half_y = self.half_size_xy
        return float(np.hypot(half_x, half_y))

STATE_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper_closed",
)
ACTION_NAMES = (
    "theta_x",
    "theta_y",
    "theta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper_closed",
)
EXTERIOR_IMAGE_KEY = "observation.images.exterior_image_1_left"
WRIST_IMAGE_KEY = "observation.images.wrist_image_left"
IMAGE_FEATURE_KEYS = (EXTERIOR_IMAGE_KEY, WRIST_IMAGE_KEY)

LEROBOT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": STATE_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ACTION_NAMES,
    },
    EXTERIOR_IMAGE_KEY: {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    WRIST_IMAGE_KEY: {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
}


@dataclass(frozen=True)
class RecordConfig:
    repo_id: str = ""
    robot_type: str = "ur5e_mujoco"
    task: str = "Pick and place the yellow cylinder on the left side of the black rectangular paper. Then move the cyan cuboid under the red cube."
    dataset_root: Path = REPO_ROOT / "dataset_record" / "data" / "task1" / "bucket1_zero_completed1"

    xml_path: Path = REPO_ROOT / "description" / "desktop_scene.xml"
    frame_name: str = "attachment_site"
    frame_type: str = "site"
    camera_names: tuple[str, str] = ("right_side_camera", "wrist_camera")
    image_feature_keys: tuple[str, str] = IMAGE_FEATURE_KEYS
    camera_render_width: int = 640
    camera_render_height: int = 480
    image_size: int = 224

    spatial_episode_rd_enabled: bool = False
    spatial_frame_rd_enabled: bool = False
    appearance_rd_enabled: bool = False
    rd_random_seed: int | None = None

    external_camera_name: str = "right_side_camera"
    episode_camera_translation_limit_m: float = 0.05
    episode_camera_rotation_limit_deg: float = 5.0
    frame_camera_translation_radius_m: float = 0.01
    frame_camera_rotation_limit_deg: float = 1.0
    frame_camera_change_interval_frames: tuple[int, int] = (25, 50)

    table_geom_names: tuple[str, ...] = (
        "table_top",
        "table_surface",
        "front_left_leg",
        "front_right_leg",
        "rear_left_leg",
        "rear_right_leg",
    )
    table_color_change_interval_frames: tuple[int, int] = (25, 50)
    table_color_palette: tuple[tuple[float, float, float, float], ...] = (
        (0.34, 0.36, 0.40, 1.0),
        (0.32, 0.22, 0.48, 1.0),
        (0.46, 0.28, 0.56, 1.0),
        (0.52, 0.30, 0.45, 1.0),
        (0.12, 0.38, 0.18, 1.0),
        (0.28, 0.40, 0.16, 1.0),
        (0.20, 0.14, 0.42, 1.0),
        (0.38, 0.29, 0.22, 1.0),
        (0.25, 0.44, 0.30, 1.0),
        (0.48, 0.36, 0.18, 1.0),
    )
    lighting_change_interval_frames: tuple[int, int] = (25, 50)
    lighting_intensity_scale_range: tuple[float, float] = (0.60, 1.40)
    lighting_rgb_jitter_limit: float = 0.08
    floor_geom_name: str = "floor"
    floor_material_names: tuple[str, ...] = (
        "floor_wood",
        "floor_concrete",
        "floor_tiles",
    )
    floor_material_change_interval_frames: tuple[int, int] = (25, 50)

    fps: int = 30
    num_episodes: int = 10
    episode_time_s: float = 60.0

    reset_region_size_xy: tuple[float, float] = (0.85, 0.65)
    reset_min_object_gap_m: float = 0.10
    reset_sampling_max_attempts: int = 10_000
    reset_random_seed: int | None = None
    reset_objects: tuple[ResetObjectConfig, ...] = (
        ResetObjectConfig("cyan_cuboid", 0.70, (0.03, 0.015)),
        ResetObjectConfig("yellow_cylinder", 0.70, (0.015, 0.015)),
        ResetObjectConfig("red_cube", 0.70, (0.015, 0.015)),
        ResetObjectConfig("white_square_sheet", 0.70, (0.05, 0.05)),
        ResetObjectConfig("black_rectangular_sheet", 0.70, (0.04, 0.025)),
    )

    end_effector_speed_m_s: float = 0.1
    gripper_open_ctrl: float = 0.0
    gripper_closed_ctrl: float = 0.8
    initial_ctrl: tuple[float, ...] = (0.0, -1.76, 2.0, 4.46, -1.57, 0.0, 0.0)

    robot_joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    gripper_actuator_name: str = "robotiq_85_left_knuckle_joint"

    position_cost: float = 1.0
    orientation_cost: float = 1.0
    ik_gain: float = 0.2
    ik_solver: str = "daqp"
    ik_damping: float = 1e-4

    idle_translation_eps_m: float = 1e-4
    idle_rotation_eps_rad: float = 1e-4
    gripper_closed_eps: float = 1e-2
    trim_context_frames: int = 1

    def initial_ctrl_array(self) -> np.ndarray:
        return np.asarray(self.initial_ctrl, dtype=np.float64)

    @property
    def translation_step_m(self) -> float:
        return self.end_effector_speed_m_s / float(self.fps)

@dataclass(frozen=True)
class AutomatedTeleopConfig:
    hover_height_m: float = 0.10
    place_clearance_m: float = 0.015
    move_speed_m_s: float = 0.1
    position_tolerance_m: float = 0.008
    max_motion_time_s: float = 20.0
    gripper_settle_time_s: float = 0.8
