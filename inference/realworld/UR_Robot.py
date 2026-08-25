"""Minimal UR realtime-state reader and URScript motion client.

This module intentionally contains only arm state and arm motion. RealSense and
serial-gripper handling live in ``state_collector.py`` and ``hand_2.py``.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


_REALTIME_FIELDS = (
    ("message_size", "i"),
    ("controller_time", "d"),
    ("target_joint_positions", "6d"),
    ("target_joint_velocities", "6d"),
    ("target_joint_accelerations", "6d"),
    ("target_joint_currents", "6d"),
    ("target_joint_moments", "6d"),
    ("joint_positions", "6d"),
    ("joint_velocities", "6d"),
    ("joint_currents", "6d"),
    ("joint_control_currents", "6d"),
    ("tcp_pose", "6d"),
    ("tcp_speed", "6d"),
    ("tcp_force", "6d"),
)


class UR_Robot:
    """Read UR state and send blocking URScript motion commands.

    TCP poses accepted by motion methods are
    ``[x, y, z, roll, pitch, yaw]`` in metres and radians. Raw controller TCP
    poses returned by :meth:`get_tcp_pose` use UR's rotation-vector convention.
    """

    SOCKET_TIMEOUT_S = 2.0
    MOTION_TIMEOUT_S = 30.0

    JOINT_ACCELERATION = 1.0
    JOINT_VELOCITY = 1.05
    TOOL_ACCELERATION = 0.5
    TOOL_VELOCITY = 0.2

    JOINT_TOLERANCE_RAD = 0.01
    TCP_POSITION_TOLERANCE_M = 0.002
    TCP_ORIENTATION_TOLERANCE_RAD = 0.01

    def __init__(self, tcp_host_ip: str = "192.168.1.9", tcp_port: int = 30003):
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = int(tcp_port)
        self._state_socket: socket.socket | None = None
        self._state_socket_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        """Close the persistent realtime state connection."""
        with self._state_socket_lock:
            self._close_state_socket_locked()

    def _close_state_socket_locked(self) -> None:
        state_socket, self._state_socket = self._state_socket, None
        if state_socket is not None:
            state_socket.close()

    def _connect(self) -> socket.socket:
        return socket.create_connection(
            (self.tcp_host_ip, self.tcp_port),
            timeout=self.SOCKET_TIMEOUT_S,
        )

    @staticmethod
    def _recv_exact(tcp_socket: socket.socket, size: int) -> bytes:
        chunks = []
        received = 0
        while received < size:
            chunk = tcp_socket.recv(size - received)
            if not chunk:
                raise ConnectionError("UR controller closed the socket mid-packet")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_state_packet(self, tcp_socket: socket.socket) -> bytes:
        header = self._recv_exact(tcp_socket, 4)
        packet_size = struct.unpack("!i", header)[0]
        if not 4 <= packet_size <= 1_000_000:
            raise ValueError(f"invalid UR realtime packet size: {packet_size}")
        return header + self._recv_exact(tcp_socket, packet_size - 4)

    def get_state(self) -> bytes:
        """Read one complete packet, reconnecting once after a broken stream."""
        with self._state_socket_lock:
            for attempt in range(2):
                if self._state_socket is None:
                    self._state_socket = self._connect()
                try:
                    return self._recv_state_packet(self._state_socket)
                except (ConnectionError, OSError, struct.error, ValueError):
                    self._close_state_socket_locked()
                    if attempt == 1:
                        raise
        raise AssertionError("unreachable")

    @staticmethod
    def _decode_fields(
        packet: bytes,
        requested_fields: Iterable[str],
    ) -> dict[str, np.ndarray | float | int]:
        requested = set(requested_fields)
        known = {name for name, _ in _REALTIME_FIELDS}
        unknown = requested - known
        if unknown:
            raise KeyError(f"unknown UR realtime field(s): {sorted(unknown)}")

        decoded: dict[str, np.ndarray | float | int] = {}
        offset = 0
        for name, field_format in _REALTIME_FIELDS:
            network_format = "!" + field_format
            field_size = struct.calcsize(network_format)
            if offset + field_size > len(packet):
                raise ValueError(
                    f"incomplete UR realtime packet while decoding {name!r}: "
                    f"need {offset + field_size} bytes, got {len(packet)}"
                )
            values = struct.unpack_from(network_format, packet, offset)
            offset += field_size
            if name not in requested:
                continue
            if len(values) == 1:
                decoded[name] = values[0]
            else:
                decoded[name] = np.asarray(values, dtype=np.float64)
            if decoded.keys() == requested:
                break

        missing = requested - decoded.keys()
        if missing:
            raise ValueError(f"UR realtime packet is missing field(s): {sorted(missing)}")
        return decoded

    def get_robot_state(self) -> dict[str, np.ndarray | float]:
        """Read a coherent arm state from one controller packet."""
        packet = self.get_state()
        receive_timestamp = time.time()
        state = self._decode_fields(
            packet,
            (
                "controller_time",
                "joint_positions",
                "joint_velocities",
                "tcp_pose",
                "tcp_speed",
                "tcp_force",
            ),
        )
        tcp_pose = np.asarray(state["tcp_pose"], dtype=np.float64)
        state["tcp_pose_rpy"] = np.concatenate(
            [tcp_pose[:3], Rotation.from_rotvec(tcp_pose[3:]).as_euler("xyz")]
        )
        state["receive_timestamp"] = receive_timestamp
        return state

    def get_joint_positions(self) -> np.ndarray:
        packet = self.get_state()
        return np.asarray(
            self._decode_fields(packet, ("joint_positions",))["joint_positions"],
            dtype=np.float64,
        )

    def get_joint_velocities(self) -> np.ndarray:
        packet = self.get_state()
        return np.asarray(
            self._decode_fields(packet, ("joint_velocities",))["joint_velocities"],
            dtype=np.float64,
        )

    def get_tcp_pose(self) -> np.ndarray:
        """Return ``[x, y, z, rx, ry, rz]`` with a rotation vector."""
        packet = self.get_state()
        return np.asarray(
            self._decode_fields(packet, ("tcp_pose",))["tcp_pose"],
            dtype=np.float64,
        )

    def get_tcp_pose_rpy(self) -> np.ndarray:
        """Return ``[x, y, z, roll, pitch, yaw]`` for pi0.5 state input."""
        tcp_pose = self.get_tcp_pose()
        rpy = Rotation.from_rotvec(tcp_pose[3:]).as_euler("xyz", degrees=False)
        return np.concatenate([tcp_pose[:3], rpy]).astype(np.float32)

    def get_tcp_speed(self) -> np.ndarray:
        packet = self.get_state()
        return np.asarray(
            self._decode_fields(packet, ("tcp_speed",))["tcp_speed"],
            dtype=np.float64,
        )

    def get_tcp_force(self) -> np.ndarray:
        packet = self.get_state()
        return np.asarray(
            self._decode_fields(packet, ("tcp_force",))["tcp_force"],
            dtype=np.float64,
        )

    @staticmethod
    def _vector6(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,):
            raise ValueError(f"{name} must have shape (6,), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains NaN or infinity")
        return vector

    @staticmethod
    def _motion_parameter(value: float, name: str, *, allow_zero: bool) -> float:
        value = float(value)
        minimum = 0.0 if allow_zero else np.finfo(np.float64).eps
        if not np.isfinite(value) or value < minimum:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {relation}, got {value}")
        return value

    @staticmethod
    def _format_vector(vector: np.ndarray) -> str:
        return ",".join(f"{value:.10f}" for value in vector)

    @staticmethod
    def _pose_rpy_to_ur_pose(pose_rpy: np.ndarray) -> np.ndarray:
        rotvec = Rotation.from_euler("xyz", pose_rpy[3:], degrees=False).as_rotvec()
        return np.concatenate([pose_rpy[:3], rotvec])

    def _send_script(self, script: str) -> None:
        if not script.endswith("\n"):
            script += "\n"
        with self._connect() as tcp_socket:
            tcp_socket.sendall(script.encode("ascii"))

    def _wait_for_joints(self, target: np.ndarray) -> None:
        deadline = time.monotonic() + self.MOTION_TIMEOUT_S
        while True:
            actual = self.get_joint_positions()
            error = float(np.max(np.abs(actual - target)))
            if error <= self.JOINT_TOLERANCE_RAD:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"joint motion timed out: max_error={error:.6f}rad, "
                    f"target={target}, actual={actual}"
                )

    def _wait_for_tcp_pose(self, target_pose_rpy: np.ndarray) -> None:
        target_rotation = Rotation.from_euler("xyz", target_pose_rpy[3:])
        deadline = time.monotonic() + self.MOTION_TIMEOUT_S
        while True:
            actual = self.get_tcp_pose()
            position_error = float(np.max(np.abs(actual[:3] - target_pose_rpy[:3])))
            actual_rotation = Rotation.from_rotvec(actual[3:])
            orientation_error = float((actual_rotation.inv() * target_rotation).magnitude())
            if (
                position_error <= self.TCP_POSITION_TOLERANCE_M
                and orientation_error <= self.TCP_ORIENTATION_TOLERANCE_RAD
            ):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "TCP motion timed out: "
                    f"position_error={position_error:.6f}m, "
                    f"orientation_error={orientation_error:.6f}rad, "
                    f"target_rpy={target_pose_rpy}, actual_rotvec={actual}"
                )

    def move_j(self, joint_configuration, k_acc=1, k_vel=1, t=0, r=0) -> None:
        """Move to six joint angles and block until the target is reached."""
        joints = self._vector6(joint_configuration, "joint_configuration")
        acceleration = self.JOINT_ACCELERATION * self._motion_parameter(
            k_acc, "k_acc", allow_zero=False
        )
        velocity = self.JOINT_VELOCITY * self._motion_parameter(
            k_vel, "k_vel", allow_zero=False
        )
        duration = self._motion_parameter(t, "t", allow_zero=True)
        blend_radius = self._motion_parameter(r, "r", allow_zero=True)
        self._send_script(
            f"movej([{self._format_vector(joints)}],a={acceleration:.10f},"
            f"v={velocity:.10f},t={duration:.10f},r={blend_radius:.10f})"
        )
        self._wait_for_joints(joints)

    def move_j_p(self, tool_configuration, k_acc=0.5, k_vel=0.5, t=0, r=0) -> None:
        """Joint-space move to an XYZ+RPY TCP pose using inverse kinematics."""
        pose_rpy = self._vector6(tool_configuration, "tool_configuration")
        ur_pose = self._pose_rpy_to_ur_pose(pose_rpy)
        acceleration = self.JOINT_ACCELERATION * self._motion_parameter(
            k_acc, "k_acc", allow_zero=False
        )
        velocity = self.JOINT_VELOCITY * self._motion_parameter(
            k_vel, "k_vel", allow_zero=False
        )
        duration = self._motion_parameter(t, "t", allow_zero=True)
        blend_radius = self._motion_parameter(r, "r", allow_zero=True)
        self._send_script(
            f"movej(get_inverse_kin(p[{self._format_vector(ur_pose)}]),"
            f"a={acceleration:.10f},v={velocity:.10f},"
            f"t={duration:.10f},r={blend_radius:.10f})"
        )
        self._wait_for_tcp_pose(pose_rpy)

    def move_l(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0) -> None:
        """Linear move to an XYZ+RPY TCP pose and block until completion."""
        pose_rpy = self._vector6(tool_configuration, "tool_configuration")
        ur_pose = self._pose_rpy_to_ur_pose(pose_rpy)
        acceleration = self.TOOL_ACCELERATION * self._motion_parameter(
            k_acc, "k_acc", allow_zero=False
        )
        velocity = self.TOOL_VELOCITY * self._motion_parameter(
            k_vel, "k_vel", allow_zero=False
        )
        duration = self._motion_parameter(t, "t", allow_zero=True)
        blend_radius = self._motion_parameter(r, "r", allow_zero=True)
        self._send_script(
            f"movel(p[{self._format_vector(ur_pose)}],a={acceleration:.10f},"
            f"v={velocity:.10f},t={duration:.10f},r={blend_radius:.10f})"
        )
        self._wait_for_tcp_pose(pose_rpy)

    def move_c(
        self,
        pose_via,
        tool_configuration,
        k_acc=1,
        k_vel=1,
        r=0,
        mode=0,
    ) -> None:
        """Circular move through ``pose_via`` to an XYZ+RPY target pose."""
        via_rpy = self._vector6(pose_via, "pose_via")
        target_rpy = self._vector6(tool_configuration, "tool_configuration")
        via_ur_pose = self._pose_rpy_to_ur_pose(via_rpy)
        target_ur_pose = self._pose_rpy_to_ur_pose(target_rpy)
        acceleration = self.TOOL_ACCELERATION * self._motion_parameter(
            k_acc, "k_acc", allow_zero=False
        )
        velocity = self.TOOL_VELOCITY * self._motion_parameter(
            k_vel, "k_vel", allow_zero=False
        )
        blend_radius = self._motion_parameter(r, "r", allow_zero=True)
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 or 1, got {mode}")
        self._send_script(
            f"movec(p[{self._format_vector(via_ur_pose)}],"
            f"p[{self._format_vector(target_ur_pose)}],"
            f"a={acceleration:.10f},v={velocity:.10f},"
            f"r={blend_radius:.10f},mode={mode})"
        )
        self._wait_for_tcp_pose(target_rpy)

    def stop_j(self, acceleration: float = 2.0) -> None:
        """Decelerate and stop joint-space motion."""
        acceleration = self._motion_parameter(
            acceleration, "acceleration", allow_zero=False
        )
        self._send_script(f"stopj({acceleration:.10f})")

    def stop_l(self, acceleration: float = 1.0) -> None:
        """Decelerate and stop Cartesian motion."""
        acceleration = self._motion_parameter(
            acceleration, "acceleration", allow_zero=False
        )
        self._send_script(f"stopl({acceleration:.10f})")


# Cleaner alias for new code while preserving the existing import name.
URRobot = UR_Robot
