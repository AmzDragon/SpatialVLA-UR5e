from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from dataset_record.config import RecordConfig
from env.domain_randomization import DomainRandomizer
from teleop.mink_ik_solver import MinkIKSolver


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    crop_top = (height - crop_size) // 2
    crop_left = (width - crop_size) // 2
    return image[
        crop_top : crop_top + crop_size,
        crop_left : crop_left + crop_size,
    ]


@dataclass
class StateSample:
    state: np.ndarray
    rotation_matrix: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    domain_randomization: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = self.state.astype(np.float32, copy=False)
        self.rotation_matrix = self.rotation_matrix.astype(np.float64, copy=True)


class LabSimMujocoEnv:
    """MuJoCo environment wrapper used by dataset recording and inference."""

    def __init__(self, cfg: RecordConfig) -> None:
        self.cfg = cfg
        self.solver = MinkIKSolver.from_xml_path(
            cfg.xml_path,
            frame_name=cfg.frame_name,
            frame_type=cfg.frame_type,
            fps=cfg.fps,
            position_cost=cfg.position_cost,
            orientation_cost=cfg.orientation_cost,
            gain=cfg.ik_gain,
            solver=cfg.ik_solver,
            solve_ik_kwargs={"damping": cfg.ik_damping},
        )
        self.model = self.solver.configuration.model
        self.data = mujoco.MjData(self.model)
        self.arm_actuator_ids = [
            self.actuator_id(name) for name in cfg.robot_joint_names
        ]
        self.gripper_actuator_id = self.actuator_id(cfg.gripper_actuator_name)
        self.initial_ctrl = cfg.initial_ctrl_array()
        self.gripper_closed = self.initial_gripper_closed
        self.rng = np.random.default_rng(cfg.reset_random_seed)
        self.domain_randomizer = DomainRandomizer(self.model, cfg)
        self.renderers = {
            camera_name: mujoco.Renderer(
                self.model,
                height=cfg.camera_render_height,
                width=cfg.camera_render_width,
            )
            for camera_name in cfg.camera_names
        }

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.cfg.fps)

    @property
    def initial_gripper_closed(self) -> bool:
        return bool(self.initial_ctrl[-1] > self.cfg.gripper_closed_eps)

    def reset(self) -> None:
        self.domain_randomizer.reset_episode()
        mujoco.mj_resetData(self.model, self.data)
        self.initialize_from_ctrl(self.initial_ctrl)
        self.gripper_closed = self.initial_gripper_closed
        object_poses = self.sample_reset_object_poses()
        for object_name, pose in object_poses:
            self.set_object_pose(object_name, pose)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.sync_solver_to_data()

    def sample_reset_object_poses(
        self,
    ) -> tuple[tuple[str, tuple[float, float, float, float, float, float]], ...]:
        region_x, region_y = self.cfg.reset_region_size_xy
        if region_x <= 0.0 or region_y <= 0.0:
            raise ValueError("reset_region_size_xy values must be positive")
        if self.cfg.reset_min_object_gap_m < 0.0:
            raise ValueError("reset_min_object_gap_m must be non-negative")
        if self.cfg.reset_sampling_max_attempts <= 0:
            raise ValueError("reset_sampling_max_attempts must be positive")

        for spec in self.cfg.reset_objects:
            half_x, half_y = spec.half_size_xy
            if 2.0 * half_x > region_x or 2.0 * half_y > region_y:
                raise ValueError(
                    f"object {spec.name} does not fit inside reset region "
                    f"{self.cfg.reset_region_size_xy}"
                )

        for _ in range(self.cfg.reset_sampling_max_attempts):
            placements = []
            valid_layout = True
            for spec in self.cfg.reset_objects:
                half_x, half_y = spec.half_size_xy
                x = float(self.rng.uniform(-region_x / 2.0 + half_x, region_x / 2.0 - half_x))
                y = float(self.rng.uniform(-region_y / 2.0 + half_y, region_y / 2.0 - half_y))

                for placed_spec, placed_x, placed_y in placements:
                    center_distance = float(np.hypot(x - placed_x, y - placed_y))
                    required_distance = (
                        spec.clearance_radius
                        + placed_spec.clearance_radius
                        + self.cfg.reset_min_object_gap_m
                    )
                    if center_distance < required_distance:
                        valid_layout = False
                        break

                if not valid_layout:
                    break
                placements.append((spec, x, y))

            if valid_layout:
                return tuple(
                    (
                        spec.name,
                        (
                            x,
                            y,
                            spec.z,
                            *spec.rotation_rpy,
                        ),
                    )
                    for spec, x, y in placements
                )

        raise RuntimeError(
            "failed to sample object positions with the configured workspace "
            "and minimum gap"
        )

    def set_object_pose(
        self,
        object_name: str,
        pose: tuple[float, float, float, float, float, float],
    ) -> None:
        position = np.asarray(pose[:3], dtype=np.float64)
        quaternion = euler_xyz_to_quaternion(
            roll=pose[3],
            pitch=pose[4],
            yaw=pose[5],
        )
        self.set_freejoint_pose(
            f"{object_name}_freejoint",
            position=position,
            quaternion=quaternion,
        )

    def set_freejoint_pose(
        self,
        joint_name: str,
        *,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise KeyError(f"joint not found in model: {joint_name}")
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_addr : qpos_addr + 3] = position
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quaternion

    def capture_env_info(self) -> dict[str, object]:
        return {
            "pose_format": "freejoint_position_xyz_quaternion_wxyz",
            "objects": {
                spec.name: self.capture_object_pose(spec.name)
                for spec in self.cfg.reset_objects
            },
            "domain_randomization": (
                self.domain_randomizer.capture_episode_info()
            ),
        }

    def apply_episode_domain_randomization_info(
        self,
        info: dict[str, object],
    ) -> None:
        self.domain_randomizer.apply_episode_info(info)

    def apply_frame_domain_randomization_state(
        self,
        state: dict[str, object],
    ) -> None:
        self.domain_randomizer.apply_frame_state(state)

    def capture_object_pose(self, object_name: str) -> dict[str, object]:
        joint_name = f"{object_name}_freejoint"
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise KeyError(f"joint not found in model: {joint_name}")

        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        position = self.data.qpos[qpos_addr : qpos_addr + 3].copy()
        quaternion = self.data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()
        return {
            "freejoint": joint_name,
            "position": position.astype(float).tolist(),
            "quaternion": quaternion.astype(float).tolist(),
        }

    def sync_solver_to_data(self) -> None:
        self.solver.configuration.update(self.data.qpos.copy())
        self.solver.reset_target_to_current()

    def apply_teleop_command(
        self,
        step_xyz: list[float] | tuple[float, float, float] | np.ndarray,
        *,
        gripper_closed: bool,
    ) -> None:
        self.gripper_closed = bool(gripper_closed)
        self.solver.configuration.update(self.data.qpos.copy())
        self.solver.step(step_xyz, scale=self.cfg.translation_step_m)
        self.sync_ctrl_from_qpos(self.solver.qpos(), self.arm_actuator_ids)
        self.data.ctrl[self.gripper_actuator_id] = (
            self.cfg.gripper_closed_ctrl if gripper_closed else self.cfg.gripper_open_ctrl
        )

    def get_site_position(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise KeyError(f"site not found in model: {site_name}")
        return self.data.site_xpos[site_id].copy()

    def move_pinch_to(
        self,
        target_position: np.ndarray,
        *,
        max_speed_m_s: float,
        position_tolerance_m: float,
        max_motion_time_s: float,
        viewer=None,
        step_callback: Callable[[], None] | None = None, # 让自动遥操作每完成一个仿真控制周期，就通知采集程序记录一帧数据
    ) -> None:
        target_position = np.asarray(target_position, dtype=np.float64)
        if max_speed_m_s <= 0.0:
            raise ValueError("max_speed_m_s must be positive")

        self.solver.configuration.update(self.data.qpos.copy())
        commanded_position = self.get_site_position("pinch")
        self.solver.reset_target_to_current()
        
        max_steps = int(max_motion_time_s / self.control_dt) # 意思是最多执行n个控制周期
        max_distance_per_step = max_speed_m_s * self.control_dt # 每个控制周期内允许的最大移动距离
        next_tick = time.perf_counter()
        for _ in range(max_steps):
            if not viewer_is_running(viewer):
                raise RuntimeError("MuJoCo viewer was closed")

            remaining = target_position - commanded_position # 当前目标位置与命令位置之间的差距
            remaining_distance = np.linalg.norm(remaining) 
            if remaining_distance > 0.0:
                target_step = remaining * min(
                    1.0,
                    max_distance_per_step / remaining_distance,
                ) # 计算当前控制周期内的目标位置增量，确保不会超过最大速度限制
                self.solver.translate_target(target_step)
                commanded_position += target_step

            self.solver.configuration.update(self.data.qpos.copy())
            self.solver.solve_step()
            self.sync_ctrl_from_qpos(self.solver.qpos(), self.arm_actuator_ids)
            self.data.ctrl[self.gripper_actuator_id] = (
                self.cfg.gripper_closed_ctrl
                if self.gripper_closed
                else self.cfg.gripper_open_ctrl
            )
            self.step_for_duration(self.control_dt)
            if step_callback is not None:
                step_callback()

            if viewer is not None:
                viewer.sync()

            error = np.linalg.norm(
                self.get_site_position("pinch") - target_position
            )
            if error <= position_tolerance_m:
                return

            next_tick += self.control_dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))

        raise RuntimeError(
            f"failed to reach target {target_position.tolist()} within "
            f"{max_motion_time_s:.1f}s"
        )

    def set_gripper(
        self,
        closed: bool,
        *,
        settle_time_s: float = 0.0,
        viewer=None,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        self.gripper_closed = bool(closed)
        self.data.ctrl[self.gripper_actuator_id] = (
            self.cfg.gripper_closed_ctrl
            if self.gripper_closed
            else self.cfg.gripper_open_ctrl
        )
        if settle_time_s > 0.0:
            self.hold(
                settle_time_s,
                viewer=viewer,
                step_callback=step_callback,
            )

    def hold(
        self,
        duration_s: float,
        *,
        viewer=None,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        steps = max(1, round(duration_s / self.control_dt))
        next_tick = time.perf_counter()
        for _ in range(steps):
            if not viewer_is_running(viewer):
                raise RuntimeError("MuJoCo viewer was closed")
            self.step_for_duration(self.control_dt)
            if step_callback is not None:
                step_callback()
            if viewer is not None:
                viewer.sync()
            next_tick += self.control_dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def step_for_duration(self, duration_s: float) -> None:
        sim_time_target = self.data.time + duration_s
        while self.data.time < sim_time_target:
            mujoco.mj_step(self.model, self.data)

    def capture_state(self, *, gripper_closed: bool | None = None) -> StateSample:
        self.domain_randomizer.prepare_frame()
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.cfg.frame_name,
        )
        if site_id < 0:
            raise KeyError(f"site not found in model: {self.cfg.frame_name}")

        position = self.data.site_xpos[site_id].copy()
        rotation_matrix = self.data.site_xmat[site_id].reshape(3, 3).copy()
        euler = matrix_to_euler_xyz(rotation_matrix)
        if gripper_closed is None:
            gripper_closed = self.gripper_closed
        gripper = np.asarray([1.0 if gripper_closed else 0.0], dtype=np.float32)
        state = np.concatenate(
            [position.astype(np.float32), euler.astype(np.float32), gripper]
        )
        return StateSample(
            state=state,
            rotation_matrix=rotation_matrix,
            images=self.render_images(),
            domain_randomization=(
                self.domain_randomizer.capture_frame_state()
            ),
        )

    def render_images(self) -> dict[str, np.ndarray]:
        images = {}
        for feature_key, camera_name in zip(
            self.cfg.image_feature_keys,
            self.cfg.camera_names,
            strict=True,
        ):
            renderer = self.renderers[camera_name]
            renderer.update_scene(self.data, camera=camera_name)
            rendered = renderer.render().astype(np.uint8, copy=False)
            cropped = _center_crop_square(rendered)
            images[feature_key] = cv2.resize(
                cropped,
                (self.cfg.image_size, self.cfg.image_size),
                interpolation=cv2.INTER_AREA,
            )
        return images

    def viewer_context(self, *, headless: bool = False):
        if headless:
            return contextlib.nullcontext(None)
        return mujoco.viewer.launch_passive(self.model, self.data)

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()

    def initialize_from_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl
        for actuator_id, value in enumerate(ctrl):
            self.data.qpos[self.actuator_qpos_addr(actuator_id)] = value
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def sync_ctrl_from_qpos(self, qpos: np.ndarray, actuator_ids: list[int]) -> None:
        for actuator_id in actuator_ids:
            qpos_addr = self.actuator_qpos_addr(actuator_id)
            value = float(qpos[qpos_addr])
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                value = float(np.clip(value, low, high))
            self.data.ctrl[actuator_id] = value

    def actuator_qpos_addr(self, actuator_id: int) -> int:
        joint_id = int(self.model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            raise ValueError(f"actuator does not target a joint: {actuator_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def actuator_id(self, actuator_name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            raise KeyError(f"actuator not found in model: {actuator_name}")
        return int(actuator_id)


def viewer_is_running(viewer: Any) -> bool:
    return True if viewer is None else bool(viewer.is_running())


def relative_rotation_euler(
    current_rotation: np.ndarray,
    next_rotation: np.ndarray,
) -> np.ndarray:
    delta_rotation = next_rotation @ current_rotation.T
    return wrap_to_pi(matrix_to_euler_xyz(delta_rotation))


def euler_xyz_to_quaternion(
    *,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
    r = rotation_matrix
    sy = math.sqrt(float(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = 0.0

    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi
