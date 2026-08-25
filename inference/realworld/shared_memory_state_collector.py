"""Multiprocess real-world collection with diffusion_policy shared ring buffers."""

from __future__ import annotations

import _thread
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Mapping

import numpy as np




from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from inference.client import RemoteUR5EInferenceClient
from inference.realworld.UR_Robot import UR_Robot
from inference.realworld.hand_2 import (
    closeSerial,
    get_position,
    openSerial,
    write,
)
import pyrealsense2 as rs


EXTERIOR_IMAGE_KEY = "observation.images.exterior_image_1_left"
WRIST_IMAGE_KEY = "observation.images.wrist_image_left"
ROBOT_SOURCE_KEY = "robot"
GRIPPER_SOURCE_KEY = "gripper"
HARDWARE_TIMEOUT_S = 2.0
COLLECTOR_STARTUP_TIMEOUT_S = HARDWARE_TIMEOUT_S + 1.0
REALWORLD_ROBOT_IP = "192.168.1.9"
SERIAL_GRIPPER_PORT = "/dev/ttyUSB0"
SERIAL_GRIPPER_BAUDRATE = 115200
SERIAL_GRIPPER_OPEN_POSITION = 16000
SERIAL_GRIPPER_CLOSED_POSITION = 500
SERIAL_GRIPPER_INITIAL_POSITION = 10000
SERIAL_GRIPPER_CLOSED_THRESHOLD = (500 + 16000) / 2
REALWORLD_PROMPT = (
    "Pick and place the yellow cylinder on the left side of the black rectangular "
    "paper. Transfer the red cube over the yellow cylinder."
)
CAMERA_FREQUENCY_HZ = 60.0
ROBOT_FREQUENCY_HZ = 125.0
GRIPPER_FREQUENCY_HZ = 20.0
BUFFER_SECONDS = 1.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
PROCESS_CHECK_INTERVAL_S = 0.05


@dataclass(frozen=True)
class SourceTiming:
    sample_count: int
    target_frequency_hz: float
    measured_frequency_hz: float
    mean_interval_s: float
    min_interval_s: float
    max_interval_s: float


@dataclass(frozen=True)
class AlignedStateSample:
    state: np.ndarray
    images: dict[str, np.ndarray]
    timestamp: float
    source_timestamps: dict[str, float]
    source_ages: dict[str, float]


def _next_tick(next_tick: float, period: float) -> float:
    next_tick += period
    delay = next_tick - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    else:
        next_tick = time.monotonic()
    return next_tick


def _recent_slots(buffer: SharedMemoryRingBuffer) -> np.ndarray:
    count = buffer.count
    k = min(count, buffer.get_max_k)
    return np.arange(count - k, count, dtype=np.int64) % buffer.buffer_size


def _timestamps(buffer: SharedMemoryRingBuffer) -> np.ndarray:
    slots = _recent_slots(buffer)
    return buffer.shared_arrays["receive_timestamp"].get()[slots].copy()


def _latest_not_after(
    buffer: SharedMemoryRingBuffer,
    anchor_timestamp: float,
) -> tuple[np.ndarray, float]:
    slots = _recent_slots(buffer)
    timestamps = buffer.shared_arrays["receive_timestamp"].get()[slots]
    idx = np.nonzero(timestamps <= anchor_timestamp)[0][-1]
    slot = slots[idx]
    value = np.array(buffer.shared_arrays["value"].get()[slot], copy=True)
    return value, float(timestamps[idx])


class _RealSenseProcess(mp.Process):
    def __init__(
        self,
        serial_number: str,
        ring_buffer: SharedMemoryRingBuffer,
        ready_event: mp.Event,
        stop_event: mp.Event,
    ) -> None:
        super().__init__(name=f"realsense-{serial_number}", daemon=True)
        self.serial_number = serial_number
        self.ring_buffer = ring_buffer
        self.ready_event = ready_event
        self.stop_event = stop_event

    def run(self) -> None:
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.color,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rs.format.rgb8,
            int(CAMERA_FREQUENCY_HZ),
        )
        pipeline = rs.pipeline()
        pipeline.start(config)

        while not self.stop_event.is_set():
            frameset = pipeline.wait_for_frames(
                timeout_ms=round(HARDWARE_TIMEOUT_S * 1000)
            )
            receive_timestamp = time.time()
            color_frame = frameset.get_color_frame()
            self.ring_buffer.put({
                "value": np.asanyarray(color_frame.get_data()),
                "receive_timestamp": receive_timestamp,
            })
            self.ready_event.set()

        pipeline.stop()


class _RobotStateProcess(mp.Process):
    def __init__(
        self,
        ring_buffer: SharedMemoryRingBuffer,
        ready_event: mp.Event,
        stop_event: mp.Event,
    ) -> None:
        super().__init__(name="ur-state", daemon=True)
        self.ring_buffer = ring_buffer
        self.ready_event = ready_event
        self.stop_event = stop_event

    def run(self) -> None:
        period = 1.0 / ROBOT_FREQUENCY_HZ
        next_publish = time.monotonic()
        with UR_Robot(tcp_host_ip=REALWORLD_ROBOT_IP) as robot:
            while not self.stop_event.is_set():
                value = robot.get_tcp_pose_rpy()
                now = time.monotonic()
                if now < next_publish:
                    continue
                self.ring_buffer.put({
                    "value": value,
                    "receive_timestamp": time.time(),
                })
                self.ready_event.set()
                next_publish = now + period


class _GripperStateProcess(mp.Process):
    def __init__(
        self,
        ring_buffer: SharedMemoryRingBuffer,
        ready_event: mp.Event,
        stop_event: mp.Event,
        target_position,
    ) -> None:
        super().__init__(name="gripper-state", daemon=True)
        self.ring_buffer = ring_buffer
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.target_position = target_position

    def run(self) -> None:
        serial_port = openSerial(SERIAL_GRIPPER_PORT, SERIAL_GRIPPER_BAUDRATE)
        write(serial_port, "speedSet", 1, [12000])
        time.sleep(0.1)
        write(serial_port, "forceSet", 1, [3000])
        time.sleep(0.1)
        commanded_position = self.target_position.value
        write(serial_port, "angleSet", 1, [commanded_position])
        time.sleep(0.1)

        period = 1.0 / GRIPPER_FREQUENCY_HZ
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            target_position = self.target_position.value
            if target_position != commanded_position:
                write(serial_port, "angleSet", 1, [target_position])
                commanded_position = target_position
            position = get_position(serial_port)
            self.ring_buffer.put({
                "value": np.float32(
                    position <= SERIAL_GRIPPER_CLOSED_THRESHOLD
                ),
                "receive_timestamp": time.time(),
            })
            self.ready_event.set()
            next_tick = _next_tick(next_tick, period)

        closeSerial(serial_port)


def _create_ring_buffer(
    manager: SharedMemoryManager,
    value: np.ndarray,
    frequency_hz: float,
) -> SharedMemoryRingBuffer:
    return SharedMemoryRingBuffer.create_from_examples(
        shm_manager=manager,
        examples={
            "value": value,
            "receive_timestamp": 0.0,
        },
        get_max_k=max(2, round(frequency_hz * BUFFER_SECONDS)),
        get_time_budget=0.2,
        put_desired_frequency=frequency_hz,
    )


class SharedMemoryRealWorldStateCollector:
    def __init__(self, camera_serial_numbers: Mapping[str, str]) -> None:
        self._manager = SharedMemoryManager()
        self._manager.start()

        self._camera_names = tuple(camera_serial_numbers)
        self._buffers = {
            name: _create_ring_buffer(
                self._manager,
                np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
                CAMERA_FREQUENCY_HZ,
            )
            for name in self._camera_names
        }
        self._buffers[ROBOT_SOURCE_KEY] = _create_ring_buffer(
            self._manager,
            np.empty(6, dtype=np.float32),
            ROBOT_FREQUENCY_HZ,
        )
        self._buffers[GRIPPER_SOURCE_KEY] = _create_ring_buffer(
            self._manager,
            np.empty((), dtype=np.float32),
            GRIPPER_FREQUENCY_HZ,
        )

        self._ready_events = {
            name: mp.Event() for name in self._buffers
        }
        self._stop_events = {
            name: mp.Event() for name in self._buffers
        }
        self._gripper_target_position = mp.Value(
            "i", SERIAL_GRIPPER_INITIAL_POSITION
        )
        self._processes: dict[str, mp.Process] = {
            name: _RealSenseProcess(
                serial_number,
                self._buffers[name],
                self._ready_events[name],
                self._stop_events[name],
            )
            for name, serial_number in camera_serial_numbers.items()
        }
        self._processes[ROBOT_SOURCE_KEY] = _RobotStateProcess(
            self._buffers[ROBOT_SOURCE_KEY],
            self._ready_events[ROBOT_SOURCE_KEY],
            self._stop_events[ROBOT_SOURCE_KEY],
        )
        self._processes[GRIPPER_SOURCE_KEY] = _GripperStateProcess(
            self._buffers[GRIPPER_SOURCE_KEY],
            self._ready_events[GRIPPER_SOURCE_KEY],
            self._stop_events[GRIPPER_SOURCE_KEY],
            self._gripper_target_position,
        )
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._child_error = ""

    def start(self) -> None:
        for process in self._processes.values():
            process.start()
        self._watchdog = threading.Thread(
            target=self._watch_processes,
            name="realworld-process-watchdog",
            daemon=True,
        )
        self._watchdog.start()

        try:
            deadline = time.monotonic() + COLLECTOR_STARTUP_TIMEOUT_S
            for name, event in self._ready_events.items():
                if not event.wait(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"timed out waiting for {name}")
        except BaseException:
            self.stop()
            if self._child_error:
                raise ChildProcessError(self._child_error) from None
            raise

    def stop(self) -> None:
        self._watchdog_stop.set()
        self._watchdog.join()
        for event in self._stop_events.values():
            event.set()
        for process in self._processes.values():
            process.join(HARDWARE_TIMEOUT_S + 0.5)
        self._manager.shutdown()

    def check_processes(self) -> None:
        for name, process in self._processes.items():
            if process.exitcode is not None:
                raise ChildProcessError(
                    f"child process {name!r} exited: "
                    f"pid={process.pid}, exitcode={process.exitcode}"
                )

    def _watch_processes(self) -> None:
        while not self._watchdog_stop.wait(PROCESS_CHECK_INTERVAL_S):
            try:
                self.check_processes()
            except ChildProcessError as exc:
                self._child_error = str(exc)
                _thread.interrupt_main()
                return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if self._child_error:
            raise ChildProcessError(self._child_error) from None

    def get_observation(
        self,
        anchor_timestamp: float | None = None,
    ) -> AlignedStateSample:
        self.check_processes()
        if anchor_timestamp is None:
            anchor_timestamp = time.time()

        selected = {
            name: _latest_not_after(buffer, anchor_timestamp)
            for name, buffer in self._buffers.items()
        }
        source_timestamps = {
            name: timestamp for name, (_, timestamp) in selected.items()
        }
        robot_state = selected[ROBOT_SOURCE_KEY][0].astype(np.float32)
        gripper_state = selected[GRIPPER_SOURCE_KEY][0].reshape(1).astype(np.float32)

        return AlignedStateSample(
            state=np.concatenate([robot_state, gripper_state]),
            images={name: selected[name][0] for name in self._camera_names},
            timestamp=anchor_timestamp,
            source_timestamps=source_timestamps,
            source_ages={
                name: anchor_timestamp - timestamp
                for name, timestamp in source_timestamps.items()
            },
        )

    def set_gripper(self, closed: bool) -> None:
        self._gripper_target_position.value = (
            SERIAL_GRIPPER_CLOSED_POSITION
            if closed
            else SERIAL_GRIPPER_OPEN_POSITION
        )

    def timing_report(self) -> dict[str, SourceTiming]:
        frequencies = {
            **{name: CAMERA_FREQUENCY_HZ for name in self._camera_names},
            ROBOT_SOURCE_KEY: ROBOT_FREQUENCY_HZ,
            GRIPPER_SOURCE_KEY: GRIPPER_FREQUENCY_HZ,
        }
        report = {}
        for name, buffer in self._buffers.items():
            timestamps = _timestamps(buffer)
            intervals = np.diff(timestamps)
            mean_interval = float(np.mean(intervals))
            report[name] = SourceTiming(
                sample_count=len(timestamps),
                target_frequency_hz=frequencies[name],
                measured_frequency_hz=1.0 / mean_interval,
                mean_interval_s=mean_interval,
                min_interval_s=float(np.min(intervals)),
                max_interval_s=float(np.max(intervals)),
            )
        return report


def discover_realsense_serial_numbers() -> tuple[str, ...]:
    return tuple(sorted(
        device.get_info(rs.camera_info.serial_number)
        for device in rs.context().query_devices()
    ))


def main() -> None:
    serial_numbers = discover_realsense_serial_numbers()
    camera_serial_numbers = {
        EXTERIOR_IMAGE_KEY: serial_numbers[0],
        WRIST_IMAGE_KEY: serial_numbers[1],
    }
    print(f"camera mapping: {camera_serial_numbers}")

    with SharedMemoryRealWorldStateCollector(camera_serial_numbers) as collector:
        time.sleep(0.2)
        anchor = time.time()
        sample = collector.get_observation(anchor)

        with RemoteUR5EInferenceClient() as client:
            actions = client.infer_action_chunk_from_sample(
                sample,
                prompt=REALWORLD_PROMPT,
            )

        print("source timing (host receive timestamps):")
        for name, timing in collector.timing_report().items():
            print(
                f"  {name}: n={timing.sample_count}, "
                f"target={timing.target_frequency_hz:.1f}Hz, "
                f"measured={timing.measured_frequency_hz:.1f}Hz, "
                f"mean_dt={timing.mean_interval_s * 1000:.2f}ms, "
                f"range=[{timing.min_interval_s * 1000:.2f}, "
                f"{timing.max_interval_s * 1000:.2f}]ms"
            )
        print(
            f"causal_alignment=ok, "
            f"max_selected_age={max(sample.source_ages.values()) * 1000:.2f}ms, "
            f"server_inference=ok, action_shape={actions.shape}, "
            f"actions_executed=0"
        )


if __name__ == "__main__":
    main()
