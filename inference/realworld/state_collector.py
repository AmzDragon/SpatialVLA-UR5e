"""Asynchronous single-step real-world observation collection for pi0.5."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXTERIOR_IMAGE_KEY = "observation.images.exterior_image_1_left"
WRIST_IMAGE_KEY = "observation.images.wrist_image_left"
ROBOT_SOURCE_KEY = "robot"
GRIPPER_SOURCE_KEY = "gripper"
HARDWARE_TIMEOUT_S = 2.0
REALWORLD_ROBOT_IP = "192.168.1.9"
SERIAL_GRIPPER_PORT = "/dev/ttyUSB0"
SERIAL_GRIPPER_BAUDRATE = 115200
SERIAL_GRIPPER_CLOSED_THRESHOLD = (500 + 16000) / 2
REALWORLD_PROMPT = (
    "Pick and place the yellow cylinder on the left side of the black rectangular "
    "paper. Transfer the red cube over the yellow cylinder."
)


@dataclass(frozen=True)
class TimestampedReading:
    value: Any
    receive_timestamp: float
    device_timestamp: float | None = None


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
    device_timestamps: dict[str, float]


class TimestampedRingBuffer:
    def __init__(self, maxlen: int) -> None:
        self._samples: deque[TimestampedReading] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, sample: TimestampedReading) -> None:
        with self._lock:
            self._samples.append(sample)

    def latest_not_after(self, anchor_timestamp: float) -> TimestampedReading:
        with self._lock:
            for sample in reversed(self._samples):
                if sample.receive_timestamp <= anchor_timestamp:
                    return sample
        raise RuntimeError(f"no sample at or before {anchor_timestamp:.6f}")

    def snapshot(self) -> tuple[TimestampedReading, ...]:
        with self._lock:
            return tuple(self._samples)


class _SourceSampler:
    def __init__(
        self,
        *,
        name: str,
        source: Any,
        frequency_hz: float,
        buffer_size: int,
    ) -> None:
        self.name = name
        self.source = source
        self.frequency_hz = float(frequency_hz)
        self.buffer = TimestampedRingBuffer(buffer_size)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def open_source(self) -> None:
        start = getattr(self.source, "start", None)
        if start:
            start()

    def close_source(self) -> None:
        stop = getattr(self.source, "stop", None)
        if stop:
            stop()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"state-source-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        self._thread.join(HARDWARE_TIMEOUT_S + 0.1)

    def wait_ready(self, timeout: float) -> bool:
        return self._ready_event.wait(timeout)

    def _read(self) -> Any:
        read = getattr(self.source, "read", None)
        return read() if read else self.source()

    def _run(self) -> None:
        period = 1.0 / self.frequency_hz
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                raw = self._read()
                sample = (
                    raw
                    if isinstance(raw, TimestampedReading)
                    else TimestampedReading(raw, time.time())
                )
                self.buffer.put(sample)
                self._ready_event.set()

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self.error = exc
            self._ready_event.set()


class RealSenseCameraSource:
    def __init__(
        self,
        serial_number: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
    ) -> None:
        self.serial_number = str(serial_number)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._pipeline = None

    @staticmethod
    def discover_serial_numbers() -> tuple[str, ...]:
        import pyrealsense2 as rs

        return tuple(sorted(
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ))

    def start(self) -> None:
        import pyrealsense2 as rs

        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        profile.get_device().first_color_sensor().set_option(
            rs.option.global_time_enabled, 1
        )

    def read(self) -> TimestampedReading:
        frames = self._pipeline.wait_for_frames(
            timeout_ms=round(HARDWARE_TIMEOUT_S * 1000)
        )
        receive_timestamp = time.time()
        color_frame = frames.get_color_frame()
        return TimestampedReading(
            value=np.asanyarray(color_frame.get_data()).copy(),
            receive_timestamp=receive_timestamp,
            device_timestamp=float(color_frame.get_timestamp()) / 1000.0,
        )

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.stop()


class RealWorldStateCollector:
    def __init__(
        self,
        *,
        cameras: Mapping[str, Any],
        robot_state_reader: Any,
        gripper_state_reader: Any,
        camera_frequencies_hz: float | Mapping[str, float] = 60.0,
        robot_frequency_hz: float = 125.0,
        gripper_frequency_hz: float = 20.0,
        buffer_size: int = 512,
    ) -> None:
        self._samplers: dict[str, _SourceSampler] = {}
        self._camera_names = tuple(cameras)

        for camera_name, source in cameras.items():
            frequency = (
                camera_frequencies_hz[camera_name]
                if isinstance(camera_frequencies_hz, Mapping)
                else camera_frequencies_hz
            )
            self._samplers[camera_name] = _SourceSampler(
                name=camera_name,
                source=source,
                frequency_hz=frequency,
                buffer_size=buffer_size,
            )

        self._samplers[ROBOT_SOURCE_KEY] = _SourceSampler(
            name=ROBOT_SOURCE_KEY,
            source=robot_state_reader,
            frequency_hz=robot_frequency_hz,
            buffer_size=buffer_size,
        )
        self._samplers[GRIPPER_SOURCE_KEY] = _SourceSampler(
            name=GRIPPER_SOURCE_KEY,
            source=gripper_state_reader,
            frequency_hz=gripper_frequency_hz,
            buffer_size=buffer_size,
        )
        self._started = False

    def start(self, timeout: float = HARDWARE_TIMEOUT_S) -> None:
        for sampler in self._samplers.values():
            sampler.open_source()
        for sampler in self._samplers.values():
            sampler.start()
        self._started = True
        try:
            self.wait_until_ready(timeout)
        except Exception:
            self.stop()
            raise

    def wait_until_ready(self, timeout: float = HARDWARE_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        for name, sampler in self._samplers.items():
            remaining = max(0.0, deadline - time.monotonic())
            if not sampler.wait_ready(remaining):
                raise TimeoutError(f"timed out waiting for source {name!r}")
            if sampler.error:
                raise sampler.error

    def stop(self) -> None:
        if not self._started:
            return
        for sampler in self._samplers.values():
            sampler.stop()
        for sampler in self._samplers.values():
            sampler.join()
        for sampler in reversed(tuple(self._samplers.values())):
            sampler.close_source()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get_observation(
        self,
        anchor_timestamp: float | None = None,
    ) -> AlignedStateSample:
        if anchor_timestamp is None:
            anchor_timestamp = time.time()

        selected = {
            name: sampler.buffer.latest_not_after(anchor_timestamp)
            for name, sampler in self._samplers.items()
        }

        robot_state = np.asarray(selected[ROBOT_SOURCE_KEY].value, dtype=np.float32)
        gripper_state = np.asarray([selected[GRIPPER_SOURCE_KEY].value], dtype=np.float32)
        source_timestamps = {
            name: sample.receive_timestamp for name, sample in selected.items()
        }
        source_ages = {
            name: anchor_timestamp - timestamp
            for name, timestamp in source_timestamps.items()
        }

        return AlignedStateSample(
            state=np.concatenate([robot_state, gripper_state]),
            images={name: selected[name].value for name in self._camera_names},
            timestamp=anchor_timestamp,
            source_timestamps=source_timestamps,
            source_ages=source_ages,
            device_timestamps={
                name: sample.device_timestamp
                for name, sample in selected.items()
                if sample.device_timestamp is not None
            },
        )

    def timing_report(self) -> dict[str, SourceTiming]:
        report = {}
        for name, sampler in self._samplers.items():
            samples = sampler.buffer.snapshot()
            timestamps = np.asarray(
                [sample.receive_timestamp for sample in samples],
                dtype=np.float64,
            )
            intervals = np.diff(timestamps)
            mean_interval = float(np.mean(intervals)) if intervals.size else float("nan")
            report[name] = SourceTiming(
                sample_count=len(samples),
                target_frequency_hz=sampler.frequency_hz,
                measured_frequency_hz=(
                    1.0 / mean_interval
                    if intervals.size and mean_interval > 0
                    else float("nan")
                ),
                mean_interval_s=mean_interval,
                min_interval_s=(float(np.min(intervals)) if intervals.size else float("nan")),
                max_interval_s=(float(np.max(intervals)) if intervals.size else float("nan")),
            )
        return report


def create_pi05_collector(
    ur_robot: Any,
    gripper_state_reader: Any,
    camera_serial_numbers: Mapping[str, str],
    *,
    camera_frequencies_hz: float | Mapping[str, float] = 60.0,
    robot_frequency_hz: float = 125.0,
    gripper_frequency_hz: float = 20.0,
    image_width: int = 640,
    image_height: int = 480,
) -> RealWorldStateCollector:
    cameras = {
        name: RealSenseCameraSource(
            serial_number,
            width=image_width,
            height=image_height,
            fps=int(
                camera_frequencies_hz[name]
                if isinstance(camera_frequencies_hz, Mapping)
                else camera_frequencies_hz
            ),
        )
        for name, serial_number in camera_serial_numbers.items()
    }
    return RealWorldStateCollector(
        cameras=cameras,
        robot_state_reader=ur_robot.get_tcp_pose_rpy,
        gripper_state_reader=gripper_state_reader,
        camera_frequencies_hz=camera_frequencies_hz,
        robot_frequency_hz=robot_frequency_hz,
        gripper_frequency_hz=gripper_frequency_hz,
    )


def main() -> None:
    from inference.client import RemoteUR5EInferenceClient
    from inference.realworld.UR_Robot import UR_Robot
    from inference.realworld.hand_2 import closeSerial, get_position, openSerial, write

    serial_numbers = RealSenseCameraSource.discover_serial_numbers()
    camera_serial_numbers = {
        EXTERIOR_IMAGE_KEY: serial_numbers[0],
        WRIST_IMAGE_KEY: serial_numbers[1],
    }
    print(f"camera mapping: {camera_serial_numbers}")

    gripper_serial = openSerial(SERIAL_GRIPPER_PORT, SERIAL_GRIPPER_BAUDRATE)
    try:
        write(gripper_serial, "speedSet", 1, [12000])
        time.sleep(0.1)
        write(gripper_serial, "forceSet", 1, [3000])
        time.sleep(0.1)
        write(gripper_serial, "angleSet", 1, [10000])
        time.sleep(0.1)

        def read_gripper_state() -> np.float32:
            position = get_position(gripper_serial)
            return np.float32(position <= SERIAL_GRIPPER_CLOSED_THRESHOLD)

        with UR_Robot(tcp_host_ip=REALWORLD_ROBOT_IP) as ur_robot:
            collector = create_pi05_collector(
                ur_robot,
                read_gripper_state,
                camera_serial_numbers,
                camera_frequencies_hz=60.0,
                robot_frequency_hz=125.0,
                gripper_frequency_hz=20.0,
            )
            with collector:
                time.sleep(0.5)
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
                    f"max_selected_age="
                    f"{max(sample.source_ages.values()) * 1000:.2f}ms, "
                    f"server_inference=ok, action_shape={actions.shape}, "
                    f"actions_executed=0"
                )
    finally:
        closeSerial(gripper_serial)


if __name__ == "__main__":
    main()
