from __future__ import annotations

import math

import mujoco
import numpy as np

from dataset_record.config import RecordConfig


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, cfg: RecordConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._validate_config()

        self.external_camera_id = self._named_id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            cfg.external_camera_name,
        )
        self.table_geom_ids = np.asarray(
            [
                self._named_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in cfg.table_geom_names
            ],
            dtype=np.int32,
        )
        self.floor_geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            cfg.floor_geom_name,
        )
        self.floor_material_ids = np.asarray(
            [
                self._named_id(mujoco.mjtObj.mjOBJ_MATERIAL, name)
                for name in cfg.floor_material_names
            ],
            dtype=np.int32,
        )
        self._floor_material_id_by_name = dict(
            zip(cfg.floor_material_names, self.floor_material_ids, strict=True)
        )

        self._nominal_camera_position = model.cam_pos[
            self.external_camera_id
        ].copy()
        self._nominal_camera_quaternion = model.cam_quat[
            self.external_camera_id
        ].copy()
        self._nominal_table_rgba = model.geom_rgba[self.table_geom_ids].copy()
        self._nominal_floor_material_id = int(
            model.geom_matid[self.floor_geom_id]
        )
        self._nominal_light_ambient = model.light_ambient.copy()
        self._nominal_light_diffuse = model.light_diffuse.copy()
        self._nominal_light_specular = model.light_specular.copy()
        self._nominal_headlight_ambient = model.vis.headlight.ambient.copy()
        self._nominal_headlight_diffuse = model.vis.headlight.diffuse.copy()
        self._nominal_headlight_specular = model.vis.headlight.specular.copy()

        seed_sequence = np.random.SeedSequence(cfg.rd_random_seed)
        child_seeds = seed_sequence.spawn(5)
        self._episode_camera_rng = np.random.default_rng(child_seeds[0])
        self._frame_camera_rng = np.random.default_rng(child_seeds[1])
        self._table_color_rng = np.random.default_rng(child_seeds[2])
        self._lighting_rng = np.random.default_rng(child_seeds[3])
        self._floor_material_rng = np.random.default_rng(child_seeds[4])

        self._episode_camera_translation = np.zeros(3, dtype=np.float64)
        self._episode_camera_rotation_rpy = np.zeros(3, dtype=np.float64)
        self._episode_camera_position = self._nominal_camera_position.copy()
        self._episode_camera_quaternion = self._nominal_camera_quaternion.copy()
        self._frame_camera_translation = np.zeros(3, dtype=np.float64)
        self._frame_camera_rotation_rpy = np.zeros(3, dtype=np.float64)
        self._current_table_color_index: int | None = None
        self._current_floor_material_index: int | None = None
        self._lighting_intensity_scale = 1.0
        self._lighting_rgb_scale = np.ones(3, dtype=np.float64)
        self._frame_index = 0
        self._next_camera_change_frame: int | None = None
        self._next_table_color_change_frame: int | None = None
        self._next_lighting_change_frame: int | None = None
        self._next_floor_material_change_frame: int | None = None

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.cfg.spatial_episode_rd_enabled
            or self.cfg.spatial_frame_rd_enabled
            or self.cfg.appearance_rd_enabled
        )

    def reset_episode(self) -> None:
        self._restore_nominal_model_values()
        self._frame_index = 0

        self._episode_camera_translation = np.zeros(3, dtype=np.float64)
        self._episode_camera_rotation_rpy = np.zeros(3, dtype=np.float64)
        if self.cfg.spatial_episode_rd_enabled:
            limit = self.cfg.episode_camera_translation_limit_m
            self._episode_camera_translation = self._episode_camera_rng.uniform(
                -limit,
                limit,
                size=3,
            )
            self._episode_camera_rotation_rpy = self._sample_rotation_rpy(
                self._episode_camera_rng,
                self.cfg.episode_camera_rotation_limit_deg,
            )

        episode_rotation = _euler_xyz_to_quaternion(
            self._episode_camera_rotation_rpy
        )
        self._episode_camera_position = (
            self._nominal_camera_position + self._episode_camera_translation
        )
        self._episode_camera_quaternion = _quaternion_multiply(
            self._nominal_camera_quaternion,
            episode_rotation,
        )

        self._frame_camera_translation = np.zeros(3, dtype=np.float64)
        self._frame_camera_rotation_rpy = np.zeros(3, dtype=np.float64)
        self._next_camera_change_frame = None
        if self.cfg.spatial_frame_rd_enabled:
            self._sample_and_apply_frame_camera()
            self._next_camera_change_frame = self._sample_interval(
                self._frame_camera_rng,
                self.cfg.frame_camera_change_interval_frames,
            )
        else:
            self._apply_camera_pose(
                self._episode_camera_position,
                self._episode_camera_quaternion,
            )

        self._current_table_color_index = None
        self._current_floor_material_index = None
        self._next_table_color_change_frame = None
        self._next_lighting_change_frame = None
        self._next_floor_material_change_frame = None
        if self.cfg.appearance_rd_enabled:
            self._sample_and_apply_table_color()
            self._sample_and_apply_lighting()
            self._sample_and_apply_floor_material()
            self._next_table_color_change_frame = self._sample_interval(
                self._table_color_rng,
                self.cfg.table_color_change_interval_frames,
            )
            self._next_lighting_change_frame = self._sample_interval(
                self._lighting_rng,
                self.cfg.lighting_change_interval_frames,
            )
            self._next_floor_material_change_frame = self._sample_interval(
                self._floor_material_rng,
                self.cfg.floor_material_change_interval_frames,
            )

    def prepare_frame(self) -> None:
        if (
            self.cfg.spatial_frame_rd_enabled
            and self._frame_index == self._next_camera_change_frame
        ):
            self._sample_and_apply_frame_camera()
            self._next_camera_change_frame += self._sample_interval(
                self._frame_camera_rng,
                self.cfg.frame_camera_change_interval_frames,
            )

        if (
            self.cfg.appearance_rd_enabled
            and self._frame_index == self._next_table_color_change_frame
        ):
            self._sample_and_apply_table_color()
            self._next_table_color_change_frame += self._sample_interval(
                self._table_color_rng,
                self.cfg.table_color_change_interval_frames,
            )

        if (
            self.cfg.appearance_rd_enabled
            and self._frame_index == self._next_lighting_change_frame
        ):
            self._sample_and_apply_lighting()
            self._next_lighting_change_frame += self._sample_interval(
                self._lighting_rng,
                self.cfg.lighting_change_interval_frames,
            )

        if (
            self.cfg.appearance_rd_enabled
            and self._frame_index == self._next_floor_material_change_frame
        ):
            self._sample_and_apply_floor_material()
            self._next_floor_material_change_frame += self._sample_interval(
                self._floor_material_rng,
                self.cfg.floor_material_change_interval_frames,
            )

        self._frame_index += 1

    def capture_episode_info(self) -> dict[str, object]:
        return {
            "enabled": {
                "spatial_episode": self.cfg.spatial_episode_rd_enabled,
                "spatial_frame": self.cfg.spatial_frame_rd_enabled,
                "appearance": self.cfg.appearance_rd_enabled,
            },
            "random_seed": self.cfg.rd_random_seed,
            "episode_camera": {
                "translation_xyz_m": self._episode_camera_translation.tolist(),
                "rotation_rpy_deg": np.rad2deg(
                    self._episode_camera_rotation_rpy
                ).tolist(),
                "position": self._episode_camera_position.tolist(),
                "quaternion_wxyz": self._episode_camera_quaternion.tolist(),
            },
            "parameters": {
                "episode_camera_translation_limit_m": (
                    self.cfg.episode_camera_translation_limit_m
                ),
                "episode_camera_rotation_limit_deg": (
                    self.cfg.episode_camera_rotation_limit_deg
                ),
                "frame_camera_translation_radius_m": (
                    self.cfg.frame_camera_translation_radius_m
                ),
                "frame_camera_rotation_limit_deg": (
                    self.cfg.frame_camera_rotation_limit_deg
                ),
                "frame_camera_change_interval_frames": list(
                    self.cfg.frame_camera_change_interval_frames
                ),
                "table_color_change_interval_frames": list(
                    self.cfg.table_color_change_interval_frames
                ),
                "lighting_change_interval_frames": list(
                    self.cfg.lighting_change_interval_frames
                ),
                "floor_material_change_interval_frames": list(
                    self.cfg.floor_material_change_interval_frames
                ),
                "floor_material_names": list(self.cfg.floor_material_names),
            },
        }

    def capture_frame_state(self) -> dict[str, object]:
        state: dict[str, object] = {}
        if self.cfg.spatial_frame_rd_enabled:
            state["camera"] = {
                "translation_xyz_m": self._frame_camera_translation.tolist(),
                "rotation_rpy_deg": np.rad2deg(
                    self._frame_camera_rotation_rpy
                ).tolist(),
                "position": self.model.cam_pos[
                    self.external_camera_id
                ].astype(float).tolist(),
                "quaternion_wxyz": self.model.cam_quat[
                    self.external_camera_id
                ].astype(float).tolist(),
            }

        if self.cfg.appearance_rd_enabled:
            state["table_color"] = {
                "palette_index": self._current_table_color_index,
                "rgba": self.model.geom_rgba[
                    self.table_geom_ids[0]
                ].astype(float).tolist(),
            }
            state["lighting"] = {
                "intensity_scale": self._lighting_intensity_scale,
                "rgb_scale": self._lighting_rgb_scale.tolist(),
                "light_ambient": self.model.light_ambient.astype(float).tolist(),
                "light_diffuse": self.model.light_diffuse.astype(float).tolist(),
                "light_specular": self.model.light_specular.astype(float).tolist(),
                "headlight_ambient": (
                    self.model.vis.headlight.ambient.astype(float).tolist()
                ),
                "headlight_diffuse": (
                    self.model.vis.headlight.diffuse.astype(float).tolist()
                ),
                "headlight_specular": (
                    self.model.vis.headlight.specular.astype(float).tolist()
                ),
            }
            floor_index = self._current_floor_material_index
            state["floor_material"] = {
                "palette_index": floor_index,
                "material_name": self.cfg.floor_material_names[floor_index],
            }
        return state

    def apply_episode_info(self, info: dict[str, object]) -> None:
        self._restore_nominal_model_values()
        episode_camera = info.get("episode_camera")
        if isinstance(episode_camera, dict):
            self._apply_camera_pose(
                np.asarray(episode_camera["position"], dtype=np.float64),
                np.asarray(
                    episode_camera["quaternion_wxyz"],
                    dtype=np.float64,
                ),
            )

    def apply_frame_state(self, state: dict[str, object]) -> None:
        camera = state.get("camera")
        if isinstance(camera, dict):
            self._apply_camera_pose(
                np.asarray(camera["position"], dtype=np.float64),
                np.asarray(camera["quaternion_wxyz"], dtype=np.float64),
            )

        table_color = state.get("table_color")
        if isinstance(table_color, dict):
            self.model.geom_rgba[self.table_geom_ids] = np.asarray(
                table_color["rgba"],
                dtype=np.float64,
            )

        lighting = state.get("lighting")
        if isinstance(lighting, dict):
            self.model.light_ambient[:] = np.asarray(
                lighting["light_ambient"], dtype=np.float64
            )
            self.model.light_diffuse[:] = np.asarray(
                lighting["light_diffuse"], dtype=np.float64
            )
            self.model.light_specular[:] = np.asarray(
                lighting["light_specular"], dtype=np.float64
            )
            self.model.vis.headlight.ambient[:] = np.asarray(
                lighting["headlight_ambient"], dtype=np.float64
            )
            self.model.vis.headlight.diffuse[:] = np.asarray(
                lighting["headlight_diffuse"], dtype=np.float64
            )
            self.model.vis.headlight.specular[:] = np.asarray(
                lighting["headlight_specular"], dtype=np.float64
            )

        floor_material = state.get("floor_material")
        if isinstance(floor_material, dict):
            material_name = str(floor_material["material_name"])
            self.model.geom_matid[self.floor_geom_id] = (
                self._floor_material_id_by_name[material_name]
            )

    def _sample_and_apply_frame_camera(self) -> None:
        self._frame_camera_translation = self._sample_uniform_ball(
            self._frame_camera_rng,
            self.cfg.frame_camera_translation_radius_m,
        )
        self._frame_camera_rotation_rpy = self._sample_rotation_rpy(
            self._frame_camera_rng,
            self.cfg.frame_camera_rotation_limit_deg,
        )
        frame_rotation = _euler_xyz_to_quaternion(
            self._frame_camera_rotation_rpy
        )
        self._apply_camera_pose(
            self._episode_camera_position + self._frame_camera_translation,
            _quaternion_multiply(
                self._episode_camera_quaternion,
                frame_rotation,
            ),
        )

    def _sample_and_apply_table_color(self) -> None:
        selected = self._sample_different_index(
            self._table_color_rng,
            len(self.cfg.table_color_palette),
            self._current_table_color_index,
        )
        self._current_table_color_index = selected
        self.model.geom_rgba[self.table_geom_ids] = np.asarray(
            self.cfg.table_color_palette[selected],
            dtype=np.float64,
        )

    def _sample_and_apply_lighting(self) -> None:
        low, high = self.cfg.lighting_intensity_scale_range
        self._lighting_intensity_scale = float(
            self._lighting_rng.uniform(low, high)
        )
        jitter = self.cfg.lighting_rgb_jitter_limit
        self._lighting_rgb_scale = self._lighting_rng.uniform(
            1.0 - jitter,
            1.0 + jitter,
            size=3,
        )
        scale = self._lighting_intensity_scale * self._lighting_rgb_scale

        self.model.light_ambient[:] = np.clip(
            self._nominal_light_ambient * scale,
            0.0,
            1.0,
        )
        self.model.light_diffuse[:] = np.clip(
            self._nominal_light_diffuse * scale,
            0.0,
            1.0,
        )
        self.model.light_specular[:] = np.clip(
            self._nominal_light_specular * scale,
            0.0,
            1.0,
        )
        self.model.vis.headlight.ambient[:] = np.clip(
            self._nominal_headlight_ambient * scale,
            0.0,
            1.0,
        )
        self.model.vis.headlight.diffuse[:] = np.clip(
            self._nominal_headlight_diffuse * scale,
            0.0,
            1.0,
        )
        self.model.vis.headlight.specular[:] = np.clip(
            self._nominal_headlight_specular * scale,
            0.0,
            1.0,
        )

    def _sample_and_apply_floor_material(self) -> None:
        selected = self._sample_different_index(
            self._floor_material_rng,
            len(self.cfg.floor_material_names),
            self._current_floor_material_index,
        )
        self._current_floor_material_index = selected
        self.model.geom_matid[self.floor_geom_id] = self.floor_material_ids[
            selected
        ]

    def _apply_camera_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        self.model.cam_pos[self.external_camera_id] = position
        self.model.cam_quat[self.external_camera_id] = quaternion

    def _restore_nominal_model_values(self) -> None:
        self.model.cam_pos[
            self.external_camera_id
        ] = self._nominal_camera_position
        self.model.cam_quat[
            self.external_camera_id
        ] = self._nominal_camera_quaternion
        self.model.geom_rgba[self.table_geom_ids] = self._nominal_table_rgba
        self.model.geom_matid[
            self.floor_geom_id
        ] = self._nominal_floor_material_id
        self.model.light_ambient[:] = self._nominal_light_ambient
        self.model.light_diffuse[:] = self._nominal_light_diffuse
        self.model.light_specular[:] = self._nominal_light_specular
        self.model.vis.headlight.ambient[:] = self._nominal_headlight_ambient
        self.model.vis.headlight.diffuse[:] = self._nominal_headlight_diffuse
        self.model.vis.headlight.specular[:] = self._nominal_headlight_specular

    @staticmethod
    def _sample_uniform_ball(
        rng: np.random.Generator,
        radius: float,
    ) -> np.ndarray:
        if radius == 0.0:
            return np.zeros(3, dtype=np.float64)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        distance = radius * float(rng.random()) ** (1.0 / 3.0)
        return direction * distance

    @staticmethod
    def _sample_rotation_rpy(
        rng: np.random.Generator,
        limit_deg: float,
    ) -> np.ndarray:
        limit_rad = math.radians(limit_deg)
        return rng.uniform(-limit_rad, limit_rad, size=3)

    @staticmethod
    def _sample_interval(
        rng: np.random.Generator,
        interval: tuple[int, int],
    ) -> int:
        low, high = interval
        return int(rng.integers(low, high + 1))

    @staticmethod
    def _sample_different_index(
        rng: np.random.Generator,
        count: int,
        current: int | None,
    ) -> int:
        candidates = [index for index in range(count) if index != current]
        return int(rng.choice(candidates))

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return int(object_id)

    def _validate_config(self) -> None:
        nonnegative_values = {
            "episode_camera_translation_limit_m": (
                self.cfg.episode_camera_translation_limit_m
            ),
            "episode_camera_rotation_limit_deg": (
                self.cfg.episode_camera_rotation_limit_deg
            ),
            "frame_camera_translation_radius_m": (
                self.cfg.frame_camera_translation_radius_m
            ),
            "frame_camera_rotation_limit_deg": (
                self.cfg.frame_camera_rotation_limit_deg
            ),
            "lighting_rgb_jitter_limit": self.cfg.lighting_rgb_jitter_limit,
        }
        for name, value in nonnegative_values.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        for name, interval in (
            (
                "frame_camera_change_interval_frames",
                self.cfg.frame_camera_change_interval_frames,
            ),
            (
                "table_color_change_interval_frames",
                self.cfg.table_color_change_interval_frames,
            ),
            (
                "lighting_change_interval_frames",
                self.cfg.lighting_change_interval_frames,
            ),
            (
                "floor_material_change_interval_frames",
                self.cfg.floor_material_change_interval_frames,
            ),
        ):
            low, high = interval
            if low <= 0 or high < low:
                raise ValueError(f"invalid {name}: {interval}")

        if len(self.cfg.table_color_palette) != 10:
            raise ValueError("table_color_palette must contain exactly 10 colors")
        table_palette = np.asarray(
            self.cfg.table_color_palette,
            dtype=np.float64,
        )
        if table_palette.shape != (10, 4) or np.any(
            (table_palette < 0.0) | (table_palette > 1.0)
        ):
            raise ValueError("table_color_palette must contain valid RGBA colors")

        if len(self.cfg.floor_material_names) != 3:
            raise ValueError("floor_material_names must contain exactly 3 names")
        if len(set(self.cfg.floor_material_names)) != 3:
            raise ValueError("floor_material_names must be unique")

        low, high = self.cfg.lighting_intensity_scale_range
        if low <= 0.0 or high < low:
            raise ValueError(
                "lighting_intensity_scale_range must be positive and ordered"
            )


def _euler_xyz_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
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


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)
