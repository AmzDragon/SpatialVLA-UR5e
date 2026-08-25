from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import websockets.sync.client


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = (
    REPO_ROOT / "thirdparty" / "openpi" / "packages" / "openpi-client" / "src"
)

for import_path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from dataset_record.config import EXTERIOR_IMAGE_KEY, WRIST_IMAGE_KEY, RecordConfig
from env import LabSimMujocoEnv, StateSample, viewer_is_running
from openpi_client import image_tools, msgpack_numpy, websocket_client_policy


STATE_KEY = "observation.state"
PROMPT_KEY = "prompt"
ACTION_KEY = "actions"
ACTION_DIM = 7
EXPECTED_ACTION_HORIZON = 50
DEFAULT_EXECUTION_HORIZON = 25
DEFAULT_NUM_CHUNKS = 1000

DEFAULT_HOST = "10.21.22.46"
DEFAULT_PORT = 8088
NETWORK_TIMEOUT_S = 2.0


class _TimeoutWebsocketClientPolicy(websocket_client_policy.WebsocketClientPolicy):
    """OpenPI websocket policy with bounded connect and inference waits."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        timeout_s: float = NETWORK_TIMEOUT_S,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout_s = float(timeout_s)
        super().__init__(host=host, port=port, api_key=api_key)

    def _wait_for_server(self):
        headers = (
            {"Authorization": f"Api-Key {self._api_key}"}
            if self._api_key
            else None
        )
        connection = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=self._timeout_s,
            close_timeout=self._timeout_s,
        )
        try:
            metadata = msgpack_numpy.unpackb(
                connection.recv(timeout=self._timeout_s)
            )
        except BaseException:
            connection.close()
            raise
        return connection, metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv(timeout=self._timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


def build_ur5e_observation(
    state_sample: StateSample,
    *,
    prompt: str,
    image_size: int = 224,
) -> dict[str, Any]:
    """Build the observation dict expected by openpi.policies.ur5e_policy.UR5Inputs."""
    missing_images = [
        key
        for key in (EXTERIOR_IMAGE_KEY, WRIST_IMAGE_KEY)
        if key not in state_sample.images
    ]
    if missing_images:
        raise KeyError(f"state sample is missing image key(s): {missing_images}")

    return {
        STATE_KEY: _as_state(state_sample.state),
        EXTERIOR_IMAGE_KEY: _as_image(
            state_sample.images[EXTERIOR_IMAGE_KEY],
            image_size=image_size,
        ),
        WRIST_IMAGE_KEY: _as_image(
            state_sample.images[WRIST_IMAGE_KEY],
            image_size=image_size,
        ),
        PROMPT_KEY: prompt,
    }


def capture_ur5e_observation(
    sim_env: LabSimMujocoEnv,
    *,
    prompt: str,
    gripper_closed: bool | None = None,
    image_size: int = 224,
) -> dict[str, Any]:
    """Capture state/images from LabSimMujocoEnv and format them for the server."""
    if gripper_closed is None:
        gripper_closed = _infer_gripper_closed(sim_env)
    state_sample = sim_env.capture_state(gripper_closed=gripper_closed)
    return build_ur5e_observation(
        state_sample,
        prompt=prompt,
        image_size=image_size,
    )


class RemoteUR5EInferenceClient:
    """Small wrapper around openpi-client for the lab_sim UR5E observation format."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int | None = DEFAULT_PORT,
        api_key: str | None = None,
        image_size: int = 224,
        timeout_s: float = NETWORK_TIMEOUT_S,
    ) -> None:
        self.image_size = image_size
        self._policy = _TimeoutWebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def infer_action_chunk_from_env(
        self,
        sim_env: LabSimMujocoEnv,
        *,
        prompt: str,
        gripper_closed: bool | None = None,
    ) -> np.ndarray:
        observation = capture_ur5e_observation(
            sim_env,
            prompt=prompt,
            gripper_closed=gripper_closed,
            image_size=self.image_size,
        )
        return self.infer_action_chunk(observation)

    def infer_action_chunk_from_sample(
        self,
        state_sample: StateSample,
        *,
        prompt: str,
    ) -> np.ndarray:
        observation = build_ur5e_observation(
            state_sample,
            prompt=prompt,
            image_size=self.image_size,
        )
        return self.infer_action_chunk(observation)

    def infer_action_chunk_from_collector(
        self,
        collector: Any,
        *,
        prompt: str,
        anchor_timestamp: float | None = None,
    ) -> np.ndarray:
        """Capture one causally aligned real-world sample and run pi0.5."""
        state_sample = collector.get_observation(anchor_timestamp)
        return self.infer_action_chunk_from_sample(state_sample, prompt=prompt)

    def infer_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        result = self._policy.infer(observation)
        if ACTION_KEY not in result:
            raise KeyError(
                f"policy response did not contain {ACTION_KEY!r}; got {list(result)}"
            )

        action_chunk = np.asarray(result[ACTION_KEY], dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != ACTION_DIM:
            raise ValueError(
                f"expected {ACTION_KEY!r} shape (T, {ACTION_DIM}), "
                f"got {action_chunk.shape}"
            )
        if action_chunk.shape[0] == 0:
            raise ValueError("policy returned an empty action chunk")
        return action_chunk

    def rollout(
        self,
        sim_env: LabSimMujocoEnv,
        *,
        prompt: str,
        num_chunks: int,
        execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
        gripper_threshold: float = 0.5,
        viewer=None,
        real_time: bool = True,
    ) -> int:
        """Repeatedly query the server and execute the first actions of each chunk."""
        executed_steps = 0
        for chunk_idx in range(num_chunks):
            if not viewer_is_running(viewer):
                break

            action_chunk = self.infer_action_chunk_from_env(
                sim_env,
                prompt=prompt,
            )
            steps = execute_action_steps(
                sim_env,
                action_chunk,
                execution_horizon=execution_horizon,
                gripper_threshold=gripper_threshold,
                viewer=viewer,
                real_time=real_time,
            )
            executed_steps += steps
            print(
                f"chunk {chunk_idx}: received {tuple(action_chunk.shape)}, "
                f"executed {steps} step(s)"
            )
        return executed_steps

    def reset(self) -> None:
        self._policy.reset()

    @property
    def server_metadata(self) -> dict[str, Any]:
        return self._policy.get_server_metadata()

    def close(self) -> None:
        websocket = getattr(self._policy, "_ws", None)
        if websocket is not None:
            websocket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def execute_action_steps(
    sim_env: LabSimMujocoEnv,
    action_chunk: np.ndarray,
    *,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    gripper_threshold: float = 0.5,
    viewer=None,
    real_time: bool = True,
) -> int:
    """Execute the first action steps as XYZ deltas plus gripper state.

    The recorded action format is [delta_xyz, delta_rpy, gripper_closed]. During
    inference rollout we intentionally ignore delta_rpy and keep the current
    end-effector orientation target.
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    steps_to_execute = min(execution_horizon, action_chunk.shape[0])
    sim_env.solver.configuration.update(sim_env.data.qpos.copy())
    sim_env.solver.reset_target_to_current()

    next_tick = time.perf_counter()
    executed_steps = 0
    for action in action_chunk[:steps_to_execute]:
        if not viewer_is_running(viewer):
            return executed_steps

        delta_xyz = np.asarray(action[:3], dtype=np.float64)
        gripper_closed = bool(float(action[6]) >= gripper_threshold)

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

        if viewer is not None:
            viewer.sync()
        if real_time:
            next_tick += sim_env.control_dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))
        executed_steps += 1

    return executed_steps


def _as_state(state: Any) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (ACTION_DIM,):
        raise ValueError(f"expected {STATE_KEY} shape {(ACTION_DIM,)}, got {state_array.shape}")
    return np.ascontiguousarray(state_array)


def _as_image(image: Any, *, image_size: int) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(f"expected HWC image with 3 dims, got {image_array.shape}")

    if image_array.shape[0] == 3 and image_array.shape[-1] != 3:
        image_array = np.moveaxis(image_array, 0, -1)
    if image_array.shape[-1] != 3:
        raise ValueError(f"expected RGB image with 3 channels, got {image_array.shape}")

    image_array = image_tools.convert_to_uint8(image_array)
    height, width = image_array.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    image_array = image_array[top : top + crop_size, left : left + crop_size]
    image_array = image_tools.resize_with_pad(image_array, image_size, image_size)
    return np.ascontiguousarray(image_array)


def _infer_gripper_closed(sim_env: LabSimMujocoEnv) -> bool:
    return sim_env.gripper_closed


def parse_args() -> argparse.Namespace:
    cfg = RecordConfig()
    parser = argparse.ArgumentParser(
        description="Query a remote openpi UR5E policy server from LabSimMujocoEnv."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default="Pick and place the yellow cylinder on the left side of the black rectangular paper. Transfer the red cube over the yellow cylinder.")
    parser.add_argument("--fps", type=int, default=cfg.fps)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Number of remote inference chunks to request and execute.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=DEFAULT_EXECUTION_HORIZON,
        help="Number of leading actions to execute from each returned chunk.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Treat action[6] >= this value as gripper closed.",
    )
    parser.add_argument(
        "--no-real-time",
        action="store_true",
        help="Execute simulation steps without sleeping to match cfg.fps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        RecordConfig(),
        task=args.prompt,
        fps=args.fps,
        image_size=args.image_size,
    )

    client = RemoteUR5EInferenceClient(
        host=args.host,
        port=args.port,
        image_size=args.image_size,
    )
    sim_env = LabSimMujocoEnv(cfg)
    try:
        sim_env.reset()
        with sim_env.viewer_context() as viewer:
            executed_steps = client.rollout(
                sim_env,
                prompt=args.prompt,
                num_chunks=args.num_chunks,
                execution_horizon=args.execution_horizon,
                gripper_threshold=args.gripper_threshold,
                viewer=viewer,
                real_time=not args.no_real_time,
            )
            print(f"executed_steps={executed_steps}")
    finally:
        sim_env.close()


if __name__ == "__main__":
    main()
