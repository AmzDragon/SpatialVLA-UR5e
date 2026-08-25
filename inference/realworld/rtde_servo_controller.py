"""Persistent RTDE servoL controller with fixed-time pose interpolation."""

from __future__ import annotations

import enum
import multiprocessing as mp
import sys
import time
from pathlib import Path
from queue import Empty

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_POLICY_ROOT = REPO_ROOT / "thirdparty" / "diffusion_policy"
if str(DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

from diffusion_policy.common.pose_trajectory_interpolator import (  # noqa: E402
    PoseTrajectoryInterpolator,
)


class _Command(enum.IntEnum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1


class RTDEServoLController(mp.Process):
    """Interpolate fixed-time TCP waypoints in a 125 Hz servoL process."""

    def __init__(
        self,
        robot_ip: str,
        frequency: float = 125.0,
        lookahead_time: float = 0.1,
        gain: int = 300,
        launch_timeout: float = 3.0,
    ) -> None:
        super().__init__(name="rtde-servoL-controller")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.launch_timeout = launch_timeout
        self.input_queue = mp.Queue()
        self.ready_event = mp.Event()
        self.commanded_pose = mp.Array("d", 6)

    @property
    def is_ready(self) -> bool:
        return self.is_alive() and self.ready_event.is_set()

    def start(self, wait: bool = True) -> None:
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self) -> None:
        if not self.ready_event.wait(self.launch_timeout) or not self.is_alive():
            raise TimeoutError("timed out starting RTDE servoL controller")

    def stop(self, wait: bool = True) -> None:
        if self.is_alive():
            self.input_queue.put({"cmd": _Command.STOP})
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def get_commanded_pose(self) -> np.ndarray:
        with self.commanded_pose.get_lock():
            return np.array(self.commanded_pose[:], dtype=np.float64)

    def schedule_waypoint(
        self,
        pose: np.ndarray,
        target_time: float,
    ) -> None:
        """Submit one TCP target using an absolute monotonic timestamp."""
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(f"target pose must have shape (6,), got {pose.shape}")
        if not np.isfinite(pose).all() or not np.isfinite(target_time):
            raise ValueError("target pose and time must be finite")
        self.input_queue.put(
            {
                "cmd": _Command.SCHEDULE_WAYPOINT,
                "target_pose": pose,
                "target_time": float(target_time),
            }
        )

    def _set_commanded_pose(self, pose: np.ndarray) -> None:
        with self.commanded_pose.get_lock():
            self.commanded_pose[:] = pose

    def run(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        rtde_c = RTDEControlInterface(self.robot_ip)
        rtde_r = RTDEReceiveInterface(self.robot_ip)

        try:
            dt = 1.0 / self.frequency
            current_time = time.monotonic()
            current_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
            pose_interp = PoseTrajectoryInterpolator(
                times=[current_time],
                poses=[current_pose],
            )
            last_waypoint_time = current_time
            self._set_commanded_pose(current_pose)

            running = True
            iteration = 0
            while running:
                period_start = rtde_c.initPeriod()
                now = time.monotonic()

                while True:
                    try:
                        command = self.input_queue.get_nowait()
                    except Empty:
                        break

                    if command["cmd"] == _Command.STOP:
                        running = False
                        break

                    target_time = float(command["target_time"])
                    pose_interp = append_fixed_waypoint(
                        pose_interp,
                        target_pose=command["target_pose"],
                        target_time=target_time,
                        current_time=now,
                        last_waypoint_time=last_waypoint_time,
                    )
                    last_waypoint_time = target_time

                if not running:
                    break

                pose_command = pose_interp(now)
                assert rtde_c.servoL(
                    pose_command,
                    0.5,
                    0.5,
                    dt,
                    self.lookahead_time,
                    self.gain,
                )
                self._set_commanded_pose(pose_command)
                rtde_c.waitPeriod(period_start)
                if iteration == 0:
                    self.ready_event.set()
                iteration += 1
        finally:
            rtde_c.servoStop()
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()


def append_fixed_waypoint(
    interpolator: PoseTrajectoryInterpolator,
    *,
    target_pose: np.ndarray,
    target_time: float,
    current_time: float,
    last_waypoint_time: float,
) -> PoseTrajectoryInterpolator:
    """Append a waypoint without speed-based retiming or timestamp changes."""
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if target_pose.shape != (6,):
        raise ValueError(f"target pose must have shape (6,), got {target_pose.shape}")
    if not np.isfinite(target_pose).all():
        raise ValueError("target pose must be finite")
    if target_time <= current_time:
        raise ValueError(
            f"target time {target_time:.6f} is not in the future of {current_time:.6f}"
        )
    if target_time <= last_waypoint_time:
        raise ValueError(
            f"target time {target_time:.6f} does not follow the previous waypoint "
            f"at {last_waypoint_time:.6f}"
        )

    segment_start = min(max(current_time, interpolator.times[0]), last_waypoint_time)
    retained = interpolator.trim(segment_start, last_waypoint_time)
    times = np.append(retained.times, target_time)
    poses = np.concatenate([retained.poses, target_pose[None]], axis=0)
    return PoseTrajectoryInterpolator(times=times, poses=poses)
