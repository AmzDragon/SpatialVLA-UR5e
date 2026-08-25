"""Real-Time Chunking client and MuJoCo rollout for the UR5e pi0.5 policy."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import math
from pathlib import Path
import sys
import threading
import time
from dataclasses import replace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from dataset_record.config import RecordConfig
from env import LabSimMujocoEnv, viewer_is_running
from inference.client import (
    ACTION_DIM,
    DEFAULT_HOST,
    DEFAULT_NUM_CHUNKS,
    DEFAULT_PORT,
    RemoteUR5EInferenceClient,
    capture_ur5e_observation,
)


RTC_REQUEST_KEY = "rtc"
RTC_PREV_ACTION_CHUNK_KEY = "prev_action_chunk"
RTC_INFERENCE_DELAY_KEY = "inference_delay"
RTC_EXECUTION_HORIZON_KEY = "execution_horizon"
RTC_HAS_PREFIX_KEY = "has_prefix"

ACTION_HORIZON = 50
QUEUE_THRESHOLD = 25
EXECUTION_HORIZON_BASE = 15
WARMUP_INFERENCES = 10
DELAY_WINDOW_SIZE = 20
DELAY_PERCENTILE = 95.0
DELAY_MARGIN_STEPS = 1
RTC_NETWORK_TIMEOUT_S = 120.0


class RTCDelayTracker:
    """Track recent inference delays and predict P95 plus one control step."""

    def __init__(
        self,
        *,
        fps: float,
        window_size: int = DELAY_WINDOW_SIZE,
        margin_steps: int = DELAY_MARGIN_STEPS,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if margin_steps < 0:
            raise ValueError("margin_steps must be non-negative")
        self.fps = float(fps)
        self.margin_steps = int(margin_steps)
        self._delay_steps: deque[int] = deque(maxlen=window_size)

    def add_latency(self, latency_s: float) -> int:
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError(f"invalid latency: {latency_s}")
        delay_steps = int(math.ceil(latency_s * self.fps))
        self.add_steps(delay_steps)
        return delay_steps

    def add_steps(self, delay_steps: int) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self._delay_steps.append(int(delay_steps))

    @property
    def samples(self) -> tuple[int, ...]:
        return tuple(self._delay_steps)

    def predicted_steps(self) -> int:
        if not self._delay_steps:
            return self.margin_steps
        ordered = sorted(self._delay_steps)
        # This is np.percentile(..., method="higher") without depending on a
        # particular NumPy percentile API version.
        percentile_index = math.ceil(DELAY_PERCENTILE / 100.0 * (len(ordered) - 1))
        return ordered[percentile_index] + self.margin_steps


def compute_execution_horizon(
    inference_delay_steps: int,
    *,
    horizon_base: int = EXECUTION_HORIZON_BASE,
) -> int:
    """Apply the requested RTC rule H = 15 - d_pred."""
    if inference_delay_steps < 0:
        raise ValueError("inference_delay_steps must be non-negative")
    if horizon_base <= 0:
        raise ValueError("horizon_base must be positive")
    return max(0, horizon_base - inference_delay_steps)


def pad_action_chunk(actions: np.ndarray, *, horizon: int = ACTION_HORIZON) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected actions with shape (T, {ACTION_DIM}), got {actions.shape}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    padded = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    steps = min(horizon, len(actions))
    padded[:steps] = actions[:steps]
    return padded


def exp_prefix_weights(start: int, end: int, total: int) -> np.ndarray:
    """NumPy equivalent of the Kinetix EXP prefix schedule for diagnostics."""
    if total <= 0:
        raise ValueError("total must be positive")
    start = min(max(int(start), 0), total)
    end = min(max(int(end), 0), total)
    start = min(start, end)
    indices = np.arange(total, dtype=np.float32)
    denominator = end - start + 1
    weights = np.clip((start - 1 - indices) / denominator + 1, 0, 1)
    weights = weights * np.expm1(weights) / np.expm1(np.float32(1.0))
    return np.where(indices >= end, 0.0, weights).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class RTCAlignmentReport:
    predicted_delay: int
    actual_delay: int
    execution_horizon: int
    strict_steps: int
    transition_steps: int
    strict_mae: float
    strict_max_abs: float
    transition_weighted_mae: float
    handoff_mae: float

    def format(self) -> str:
        return (
            f"RTC alignment: d_pred={self.predicted_delay}, d_actual={self.actual_delay}, "
            f"H={self.execution_horizon}, strict={self.strict_steps}, transition={self.transition_steps}, "
            f"strict_mae={self.strict_mae:.6f}, strict_max={self.strict_max_abs:.6f}, "
            f"transition_weighted_mae={self.transition_weighted_mae:.6f}, "
            f"handoff_mae={self.handoff_mae:.6f}"
        )


def check_chunk_alignment(
    previous_chunk: np.ndarray,
    new_chunk: np.ndarray,
    *,
    predicted_delay: int,
    actual_delay: int,
    execution_horizon: int,
) -> RTCAlignmentReport:
    """Validate alignment and report errors between overlapping action chunks."""
    previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
    new_chunk = np.asarray(new_chunk, dtype=np.float32)
    expected_shape = (ACTION_HORIZON, ACTION_DIM)
    if previous_chunk.shape != expected_shape or new_chunk.shape != expected_shape:
        raise ValueError(
            f"alignment expects two {expected_shape} chunks, got "
            f"{previous_chunk.shape} and {new_chunk.shape}"
        )
    if not np.isfinite(previous_chunk).all() or not np.isfinite(new_chunk).all():
        raise ValueError("RTC alignment received NaN or Inf actions")

    weights = exp_prefix_weights(predicted_delay, execution_horizon, ACTION_HORIZON)
    strict_steps = min(predicted_delay, execution_horizon, ACTION_HORIZON)
    transition_start = strict_steps
    transition_end = min(execution_horizon, ACTION_HORIZON)
    error = np.abs(new_chunk - previous_chunk)

    if strict_steps:
        strict_error = error[:strict_steps]
        strict_mae = float(np.mean(strict_error))
        strict_max_abs = float(np.max(strict_error))
    else:
        strict_mae = 0.0
        strict_max_abs = 0.0

    transition_weights = weights[transition_start:transition_end]
    if transition_weights.size and float(np.sum(transition_weights)) > 0:
        transition_error = np.mean(error[transition_start:transition_end], axis=-1)
        transition_weighted_mae = float(
            np.sum(transition_error * transition_weights) / np.sum(transition_weights)
        )
    else:
        transition_weighted_mae = 0.0

    if 0 <= actual_delay < ACTION_HORIZON:
        handoff_mae = float(np.mean(error[actual_delay]))
    else:
        handoff_mae = math.inf

    return RTCAlignmentReport(
        predicted_delay=predicted_delay,
        actual_delay=actual_delay,
        execution_horizon=execution_horizon,
        strict_steps=strict_steps,
        transition_steps=max(0, transition_end - transition_start),
        strict_mae=strict_mae,
        strict_max_abs=strict_max_abs,
        transition_weighted_mae=transition_weighted_mae,
        handoff_mae=handoff_mae,
    )


class RTCActionQueue:
    """Thread-safe action queue with a monotonic control-timeline counter."""

    def __init__(self, *, action_horizon: int = ACTION_HORIZON) -> None:
        self.action_horizon = action_horizon
        self._actions = np.empty((0, ACTION_DIM), dtype=np.float32)
        self._cursor = 0
        self._total_control_steps = 0
        self._lock = threading.Lock()

    @property
    def total_control_steps(self) -> int:
        with self._lock:
            return self._total_control_steps

    def remaining(self) -> int:
        with self._lock:
            return len(self._actions) - self._cursor

    def pop(self) -> np.ndarray | None:
        """Advance one control tick and return its queued action, if available."""
        with self._lock:
            self._total_control_steps += 1
            if self._cursor >= len(self._actions):
                return None
            action = self._actions[self._cursor].copy()
            self._cursor += 1
            return action

    def replace(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"expected replacement actions shaped (T, {ACTION_DIM}), got {actions.shape}")
        with self._lock:
            self._actions = actions.copy()
            self._cursor = 0

    def padded_remaining(self) -> np.ndarray:
        with self._lock:
            remaining = self._actions[self._cursor :].copy()
        return pad_action_chunk(remaining, horizon=self.action_horizon)


@dataclasses.dataclass(frozen=True)
class _PendingInference:
    previous_chunk: np.ndarray
    predicted_delay: int
    execution_horizon: int
    control_step_at_start: int


@dataclasses.dataclass(frozen=True)
class _InferenceResult:
    actions: np.ndarray
    latency_s: float


class RTCRemoteClient:
    """Run remote policy inference in one background thread while actions execute."""

    def __init__(
        self,
        remote: RemoteUR5EInferenceClient,
        *,
        fps: float,
        action_horizon: int = ACTION_HORIZON,
        queue_threshold: int = QUEUE_THRESHOLD,
    ) -> None:
        if not 0 < queue_threshold < action_horizon:
            raise ValueError("queue_threshold must be between zero and action_horizon")
        self.remote = remote
        self.action_horizon = action_horizon
        self.queue_threshold = queue_threshold
        if remote.server_metadata.get("rtc_enabled") is not True:
            raise RuntimeError("The OpenPI server must be started with --rtc before using the RTC client")
        self.delay_tracker = RTCDelayTracker(fps=fps)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-rtc")
        self._future: Future[_InferenceResult] | None = None
        self._pending: _PendingInference | None = None

    @property
    def inference_pending(self) -> bool:
        return self._future is not None

    def warmup(self, observation: dict[str, Any], *, num_inferences: int = WARMUP_INFERENCES) -> np.ndarray:
        """Run initial stationary inferences and seed the action queue with the last chunk."""
        if num_inferences <= 0:
            raise ValueError("num_inferences must be positive")
        zero_prefix = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
        latest_actions = zero_prefix
        print("RTC compile warmup: this request is excluded from delay statistics")
        self._infer(
            observation,
            previous_chunk=zero_prefix,
            predicted_delay=0,
            execution_horizon=0,
            has_prefix=False,
        )
        for index in range(num_inferences):
            result = self._infer(
                observation,
                previous_chunk=zero_prefix,
                predicted_delay=0,
                execution_horizon=0,
                has_prefix=False,
            )
            delay_steps = self.delay_tracker.add_latency(result.latency_s)
            latest_actions = result.actions
            print(
                f"RTC warmup {index + 1}/{num_inferences}: "
                f"latency={result.latency_s:.3f}s, delay_steps={delay_steps}"
            )

        predicted = self.delay_tracker.predicted_steps()
        print(
            f"RTC warmup complete: samples={self.delay_tracker.samples}, "
            f"d_pred=P95+1={predicted}, H={compute_execution_horizon(predicted)}"
        )
        return latest_actions

    def should_request(self, queue: RTCActionQueue) -> bool:
        return not self.inference_pending and queue.remaining() <= self.queue_threshold

    def request(self, observation: dict[str, Any], queue: RTCActionQueue) -> None:
        if self.inference_pending:
            raise RuntimeError("an RTC inference request is already pending")
        predicted_delay = min(self.delay_tracker.predicted_steps(), self.action_horizon)
        execution_horizon = compute_execution_horizon(predicted_delay)
        previous_chunk = queue.padded_remaining()
        self._pending = _PendingInference(
            previous_chunk=previous_chunk,
            predicted_delay=predicted_delay,
            execution_horizon=execution_horizon,
            control_step_at_start=queue.total_control_steps,
        )
        self._future = self._executor.submit(
            self._infer,
            observation,
            previous_chunk=previous_chunk,
            predicted_delay=predicted_delay,
            execution_horizon=execution_horizon,
            has_prefix=True,
        )

    def poll(self, queue: RTCActionQueue) -> RTCAlignmentReport | None:
        if self._future is None or not self._future.done():
            return None
        if self._pending is None:
            raise RuntimeError("RTC future completed without request metadata")

        result = self._future.result()
        pending = self._pending
        self._future = None
        self._pending = None

        actual_delay = queue.total_control_steps - pending.control_step_at_start
        measured_delay = self.delay_tracker.add_latency(result.latency_s)
        if actual_delay >= len(result.actions):
            raise RuntimeError(
                f"RTC inference consumed {actual_delay} steps but returned only {len(result.actions)} actions"
            )

        report = check_chunk_alignment(
            pending.previous_chunk,
            pad_action_chunk(result.actions, horizon=self.action_horizon),
            predicted_delay=pending.predicted_delay,
            actual_delay=actual_delay,
            execution_horizon=pending.execution_horizon,
        )
        queue.replace(result.actions[actual_delay:])
        print(
            f"RTC inference complete: latency={result.latency_s:.3f}s, "
            f"measured_delay={measured_delay}, actual_delay={actual_delay}, "
            f"remaining={queue.remaining()}, next_d_pred={self.delay_tracker.predicted_steps()}"
        )
        print(report.format())
        return report

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.remote.close()

    def _infer(
        self,
        observation: dict[str, Any],
        *,
        previous_chunk: np.ndarray,
        predicted_delay: int,
        execution_horizon: int,
        has_prefix: bool,
    ) -> _InferenceResult:
        request = dict(observation)
        request[RTC_REQUEST_KEY] = {
            RTC_PREV_ACTION_CHUNK_KEY: pad_action_chunk(previous_chunk, horizon=self.action_horizon),
            RTC_INFERENCE_DELAY_KEY: int(predicted_delay),
            RTC_EXECUTION_HORIZON_KEY: int(execution_horizon),
            RTC_HAS_PREFIX_KEY: bool(has_prefix),
        }
        start = time.perf_counter()
        actions = self.remote.infer_action_chunk(request)
        latency_s = time.perf_counter() - start
        if actions.shape != (self.action_horizon, ACTION_DIM):
            raise ValueError(
                f"RTC requires action chunk shape {(self.action_horizon, ACTION_DIM)}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("RTC policy returned NaN or Inf actions")
        return _InferenceResult(actions=actions, latency_s=latency_s)


def execute_sim_action(
    sim_env: LabSimMujocoEnv,
    action: np.ndarray,
    *,
    gripper_threshold: float,
) -> None:
    """Execute one action using the same semantics as inference.client."""
    delta_xyz = np.asarray(action[:3], dtype=np.float64)
    gripper_closed = bool(float(action[6]) >= gripper_threshold)
    sim_env.gripper_closed = gripper_closed
    sim_env.solver.configuration.update(sim_env.data.qpos.copy())
    sim_env.solver.step(delta_xyz, scale=1.0)
    sim_env.sync_ctrl_from_qpos(sim_env.solver.qpos(), sim_env.arm_actuator_ids)
    sim_env.data.ctrl[sim_env.gripper_actuator_id] = (
        sim_env.cfg.gripper_closed_ctrl if gripper_closed else sim_env.cfg.gripper_open_ctrl
    )
    sim_env.step_for_duration(sim_env.control_dt)


def parse_args() -> argparse.Namespace:
    cfg = RecordConfig()
    parser = argparse.ArgumentParser(description="Run RTC-enabled openpi pi0.5 inference in MuJoCo.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default=cfg.task)
    parser.add_argument("--fps", type=int, default=cfg.fps)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--warmup-inferences", type=int, default=WARMUP_INFERENCES)
    parser.add_argument("--queue-threshold", type=int, default=QUEUE_THRESHOLD)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--network-timeout", type=float, default=RTC_NETWORK_TIMEOUT_S)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        RecordConfig(),
        task=args.prompt,
        fps=args.fps,
        image_size=args.image_size,
    )
    sim_env = LabSimMujocoEnv(cfg)
    remote = RemoteUR5EInferenceClient(
        host=args.host,
        port=args.port,
        image_size=args.image_size,
        timeout_s=args.network_timeout,
    )
    rtc_client = RTCRemoteClient(
        remote,
        fps=args.fps,
        queue_threshold=args.queue_threshold,
    )
    queue = RTCActionQueue()

    try:
        sim_env.reset()
        initial_observation = capture_ur5e_observation(sim_env, prompt=args.prompt)
        initial_actions = rtc_client.warmup(
            initial_observation,
            num_inferences=args.warmup_inferences,
        )
        queue.replace(initial_actions)
        sim_env.solver.configuration.update(sim_env.data.qpos.copy())
        sim_env.solver.reset_target_to_current()

        completed_chunks = 0
        next_tick = time.perf_counter()
        with sim_env.viewer_context() as viewer:
            while viewer_is_running(viewer) and completed_chunks < args.num_chunks:
                report = rtc_client.poll(queue)
                if report is not None:
                    completed_chunks += 1

                if completed_chunks < args.num_chunks and rtc_client.should_request(queue):
                    observation = capture_ur5e_observation(sim_env, prompt=args.prompt)
                    rtc_client.request(observation, queue)

                action = queue.pop()
                if action is not None:
                    execute_sim_action(
                        sim_env,
                        action,
                        gripper_threshold=args.gripper_threshold,
                    )
                else:
                    sim_env.step_for_duration(sim_env.control_dt)

                if viewer is not None:
                    viewer.sync()
                next_tick += sim_env.control_dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))

        print(
            f"RTC rollout complete: chunks={completed_chunks}, "
            f"control_steps={queue.total_control_steps}, delay_samples={rtc_client.delay_tracker.samples}"
        )
    finally:
        rtc_client.close()
        sim_env.close()


if __name__ == "__main__":
    main()
