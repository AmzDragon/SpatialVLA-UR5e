"""Real-world pi0.5 inference with asynchronous Real-Time Chunking."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.client import RemoteUR5EInferenceClient, build_ur5e_observation
from inference.realworld.rtde_servo_controller import RTDEServoLController
from inference.rtc import (
    QUEUE_THRESHOLD,
    RTC_NETWORK_TIMEOUT_S,
    WARMUP_INFERENCES,
    RTCActionQueue,
    RTCRemoteClient,
    compute_execution_horizon,
)


ACTION_HZ = 30.0
ACTION_PERIOD_S = 1.0 / ACTION_HZ
GRIPPER_THRESHOLD = 0.5
RTDE_CONTROL_HZ = 125.0
RTDE_COMMAND_LEAD_S = 2.0 / RTDE_CONTROL_HZ
UR_LOOKAHEAD_TIME_S = 0.1
UR_SERVO_GAIN = 300
CAMERA_WARMUP_S = 3.0
LOG_ROOT = Path(__file__).with_name("logs")


def integrate_action_target(target_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Integrate one model-frame delta action into a base-frame TCP target."""
    target_pose = np.asarray(target_pose, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if target_pose.shape != (6,):
        raise ValueError(f"target pose must have shape (6,), got {target_pose.shape}")
    if action.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {action.shape}")
    if not np.isfinite(target_pose).all() or not np.isfinite(action).all():
        raise ValueError("target pose and action must be finite")

    next_pose = target_pose.copy()
    next_pose[:3] += np.array([action[1], -action[0], action[2]])
    target_rotation = Rotation.from_rotvec(target_pose[3:])
    delta_rotation = Rotation.from_euler("xyz", action[3:6])
    next_pose[3:] = (delta_rotation * target_rotation).as_rotvec()
    return next_pose


def capture_model_observation(
    collector: Any,
    client: RemoteUR5EInferenceClient,
    *,
    prompt: str,
) -> tuple[Any, dict[str, Any]]:
    """Capture a causal real-world sample and apply the existing model coordinates."""
    base_sample = collector.get_observation(time.time())
    state = base_sample.state.copy()
    state[:3] = [-state[1], state[0] - 0.43, state[2] + 0.70]
    state[3] = -state[3]
    model_sample = replace(base_sample, state=state)
    observation = build_ur5e_observation(
        model_sample,
        prompt=prompt,
        image_size=client.image_size,
    )
    return base_sample, observation


def _create_log_dir(cameras: dict[str, str], *, prompt: str) -> Path:
    log_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True)
    (log_dir / "run.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now().isoformat(),
                "prompt": prompt,
                "cameras": cameras,
                "action_hz": ACTION_HZ,
                "queue_threshold": QUEUE_THRESHOLD,
                "warmup_inferences": WARMUP_INFERENCES,
                "rtde_control_hz": RTDE_CONTROL_HZ,
                "rtde_command_lead_s": RTDE_COMMAND_LEAD_S,
                "rtde_lookahead_time_s": UR_LOOKAHEAD_TIME_S,
                "rtde_gain": UR_SERVO_GAIN,
                "rtde_interpolation": "fixed_time_adjacent_waypoints",
                "rtde_speed_retime": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return log_dir


def _save_observation(
    log_dir: Path,
    *,
    request_index: int,
    base_sample: Any,
    observation: dict[str, Any],
    exterior_image_key: str,
    wrist_image_key: str,
) -> str:
    filename = f"request_{request_index:06d}.npz"
    np.savez(
        log_dir / filename,
        exterior_image=observation[exterior_image_key],
        wrist_image=observation[wrist_image_key],
        raw_state=base_sample.state,
        model_state=observation["observation.state"],
    )
    return filename


def _write_event(log_file: Any, event: str, **values: Any) -> None:
    record = {
        "event": event,
        "timestamp": time.time(),
        **values,
    }
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_file.flush()


def run_rtc_control_loop(
    collector: Any,
    controller: RTDEServoLController,
    rtc_client: RTCRemoteClient,
    queue: RTCActionQueue,
    *,
    prompt: str,
    log_dir: Path,
    log_file: Any,
    exterior_image_key: str,
    wrist_image_key: str,
) -> None:
    base_sample, initial_observation = capture_model_observation(
        collector,
        rtc_client.remote,
        prompt=prompt,
    )
    initial_observation_file = _save_observation(
        log_dir,
        request_index=0,
        base_sample=base_sample,
        observation=initial_observation,
        exterior_image_key=exterior_image_key,
        wrist_image_key=wrist_image_key,
    )
    initial_actions = rtc_client.warmup(
        initial_observation,
        num_inferences=WARMUP_INFERENCES,
    )
    queue.replace(initial_actions)
    np.savez(log_dir / "initial_actions.npz", actions=initial_actions)

    target_pose = controller.get_commanded_pose()
    control_start = time.monotonic()
    action_index = 0
    request_index = 1
    completed_chunks = 0
    _write_event(
        log_file,
        "control_start",
        observation_file=initial_observation_file,
        initial_target_pose=target_pose.tolist(),
        delay_samples=list(rtc_client.delay_tracker.samples),
    )

    while True:
        if not controller.is_ready:
            raise RuntimeError("RTDE servoL controller exited during RTC rollout")

        report = rtc_client.poll(queue)
        if report is not None:
            completed_chunks += 1
            _write_event(
                log_file,
                "inference_complete",
                chunk=completed_chunks,
                queue_remaining=queue.remaining(),
                next_predicted_delay=rtc_client.delay_tracker.predicted_steps(),
                alignment=dataclasses.asdict(report),
            )

        if rtc_client.should_request(queue):
            base_sample, observation = capture_model_observation(
                collector,
                rtc_client.remote,
                prompt=prompt,
            )
            observation_file = _save_observation(
                log_dir,
                request_index=request_index,
                base_sample=base_sample,
                observation=observation,
                exterior_image_key=exterior_image_key,
                wrist_image_key=wrist_image_key,
            )
            predicted_delay = min(
                rtc_client.delay_tracker.predicted_steps(),
                rtc_client.action_horizon,
            )
            rtc_client.request(observation, queue)
            _write_event(
                log_file,
                "inference_request",
                request=request_index,
                observation_file=observation_file,
                anchor_timestamp=base_sample.timestamp,
                source_ages_ms={
                    name: age * 1000.0 for name, age in base_sample.source_ages.items()
                },
                queue_remaining=queue.remaining(),
                predicted_delay=predicted_delay,
                execution_horizon=compute_execution_horizon(predicted_delay),
                control_step=queue.total_control_steps,
            )
            request_index += 1

        action = queue.pop()
        if action is None:
            raise RuntimeError("RTC action queue exhausted before inference completed")

        target_pose = integrate_action_target(target_pose, action)
        target_time = (
            control_start
            + RTDE_COMMAND_LEAD_S
            + (action_index + 1) * ACTION_PERIOD_S
        )
        if target_time <= time.monotonic():
            raise RuntimeError(
                "missed fixed RTDE waypoint deadline; refusing to retime the action"
            )

        controller.schedule_waypoint(target_pose, target_time)
        gripper_closed = bool(float(action[6]) >= GRIPPER_THRESHOLD)
        collector.set_gripper(gripper_closed)
        action_index += 1

        _write_event(
            log_file,
            "action_submitted",
            action_index=action_index,
            action=action.tolist(),
            target_pose=target_pose.tolist(),
            target_monotonic_time=target_time,
            gripper_closed=gripper_closed,
            queue_remaining=queue.remaining(),
            inference_pending=rtc_client.inference_pending,
        )

        next_tick = control_start + action_index * ACTION_PERIOD_S
        time.sleep(max(0.0, next_tick - time.monotonic()))


def main() -> None:
    from inference.realworld.shared_memory_state_collector import (
        EXTERIOR_IMAGE_KEY,
        REALWORLD_PROMPT,
        REALWORLD_ROBOT_IP,
        WRIST_IMAGE_KEY,
        SharedMemoryRealWorldStateCollector,
    )

    cameras = {
        EXTERIOR_IMAGE_KEY: "344322074827",
        WRIST_IMAGE_KEY: "135122074815",
    }
    log_dir = _create_log_dir(cameras, prompt=REALWORLD_PROMPT)
    print(f"log_dir={log_dir}")

    remote: RemoteUR5EInferenceClient | None = None
    rtc_client: RTCRemoteClient | None = None
    try:
        with (log_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1) as log_file:
            with SharedMemoryRealWorldStateCollector(cameras) as collector:
                time.sleep(CAMERA_WARMUP_S)
                with RTDEServoLController(
                    robot_ip=REALWORLD_ROBOT_IP,
                    frequency=RTDE_CONTROL_HZ,
                    lookahead_time=UR_LOOKAHEAD_TIME_S,
                    gain=UR_SERVO_GAIN,
                ) as controller:
                    remote = RemoteUR5EInferenceClient(timeout_s=RTC_NETWORK_TIMEOUT_S)
                    rtc_client = RTCRemoteClient(
                        remote,
                        fps=ACTION_HZ,
                        queue_threshold=QUEUE_THRESHOLD,
                    )
                    run_rtc_control_loop(
                        collector,
                        controller,
                        rtc_client,
                        RTCActionQueue(),
                        prompt=REALWORLD_PROMPT,
                        log_dir=log_dir,
                        log_file=log_file,
                        exterior_image_key=EXTERIOR_IMAGE_KEY,
                        wrist_image_key=WRIST_IMAGE_KEY,
                    )
    except KeyboardInterrupt:
        print("RTC real-world rollout stopped by user")
    finally:
        if rtc_client is not None:
            rtc_client.close()
        elif remote is not None:
            remote.close()


if __name__ == "__main__":
    main()
