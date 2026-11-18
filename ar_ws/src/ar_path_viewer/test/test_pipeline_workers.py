import threading
import time

import cv2
import numpy as np
import pytest

from ar_path_viewer.async_video_writer import AsyncPacedVideoWriter
from ar_path_viewer.latest_image_source import LatestImageSource
from ar_path_viewer.v4l2_timestamp_source import V4L2ReadMetrics


class ControlledSource:
    def __init__(self):
        self.condition = threading.Condition()
        self.items = []
        self.closed = False
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ""
        self.last_read_metrics = None
        self.requested_capture_fps = 30.0
        self.effective_capture_fps = 30.0
        self.capture_rate_reason = "matched"

    def offer(self, value, stamp):
        with self.condition:
            self.items.append((value, stamp))
            self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.items or self.closed, 0.1)
            if not self.items:
                return None
            value, stamp = self.items.pop(0)
        self.frame_timestamp_monotonic = stamp
        self.frame_timestamp_kind = "v4l2_monotonic_soe"
        self.last_read_metrics = V4L2ReadMetrics(
            frame_timestamp_monotonic_sec=stamp,
            sequence=int(value),
        )
        return np.full((3, 4, 3), value, np.uint8)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def test_latest_camera_handoff_is_bounded_and_keeps_image_timestamp_pair():
    source = ControlledSource()
    latest = LatestImageSource(source)
    try:
        source.offer(1, 10.01)
        source.offer(2, 10.02)
        source.offer(3, 10.03)
        assert _wait_until(lambda: latest.handoff_discarded_total >= 2)
        image = latest.read()
        assert int(image[0, 0, 0]) == 3
        assert latest.frame_timestamp_monotonic == 10.03
        assert latest.frame_timestamp_kind == "v4l2_monotonic_soe"
        assert latest.last_read_metrics.capture_total == 3
        assert latest.last_read_metrics.handoff_discarded_total == 2
        assert latest.last_read_metrics.source_interval_sec == pytest.approx(0.01)
        assert latest.read() is None
        assert latest.temporary_empty
    finally:
        latest.close()
    assert not latest.worker_alive
    assert source.closed


def test_slow_camera_does_not_block_consumer_and_recovers_with_new_stamp():
    source = ControlledSource()
    latest = LatestImageSource(source, unavailable_after_sec=0.2)
    try:
        started = time.monotonic()
        assert latest.read() is None
        assert time.monotonic() - started < 0.02
        source.offer(7, 20.07)
        assert _wait_until(lambda: latest._generation == 1)
        assert int(latest.read()[0, 0, 0]) == 7
        assert latest.frame_timestamp_monotonic == 20.07
    finally:
        latest.close()


def test_camera_disconnect_becomes_unavailable_and_worker_closes():
    source = ControlledSource()
    latest = LatestImageSource(source, unavailable_after_sec=0.02)
    try:
        assert latest.read() is None
        assert _wait_until(lambda: not latest.temporary_empty, timeout=0.2)
        assert latest.read() is None
    finally:
        latest.close()

    assert not latest.worker_alive
    assert source.closed


class SlowWriter:
    def __init__(self, delay=0.0, fail=False, delay_writes=0):
        self.delay = delay
        self.fail = fail
        self.delay_writes = delay_writes
        self.frames = []
        self.released = False

    def write(self, frame):
        if self.delay_writes > 0:
            self.delay_writes -= 1
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("encoder failure")
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def test_one_second_encoder_stall_never_blocks_camera_render_submitter():
    output = SlowWriter(delay=1.0, delay_writes=1)
    writer = AsyncPacedVideoWriter(output, 30.0)
    durations = []
    for value in range(10):
        started = time.monotonic()
        assert writer.write(np.full((2, 2, 3), value, np.uint8))
        durations.append(time.monotonic() - started)
        time.sleep(0.005)
    assert max(durations) < 0.05
    assert writer.replaced_frames > 0
    metrics_started = time.monotonic()
    assert writer.frames_written >= 0
    assert writer.duplicate_frames_written >= 0
    assert writer.max_encode_duration_sec >= 0.0
    assert time.monotonic() - metrics_started < 0.05
    writer.release()
    assert not writer.worker_alive
    assert output.released
    assert writer.frames_written >= 1


def test_encoder_failure_isolated_and_future_frames_are_rejected():
    output = SlowWriter(fail=True)
    writer = AsyncPacedVideoWriter(output, 30.0)
    assert writer.write(np.zeros((2, 2, 3), np.uint8))
    assert _wait_until(lambda: bool(writer.error_reason))
    started = time.monotonic()
    assert writer.write(np.ones((2, 2, 3), np.uint8)) is False
    assert time.monotonic() - started < 0.02
    writer.release()
    assert output.released


def test_encoder_metrics_are_safe_during_concurrent_diagnostics_reads():
    output = SlowWriter(delay=0.002, delay_writes=30)
    writer = AsyncPacedVideoWriter(output, 30.0)
    stop = threading.Event()
    failures = []

    def read_metrics():
        try:
            while not stop.is_set():
                assert writer.encode_duration_sec >= 0.0
                assert writer.max_encode_duration_sec >= 0.0
        except BaseException as exc:
            failures.append(exc)

    reader = threading.Thread(target=read_metrics)
    reader.start()
    try:
        for value in range(30):
            assert writer.write(np.full((2, 2, 3), value, np.uint8))
            time.sleep(0.001)
    finally:
        writer.release()
        stop.set()
        reader.join(timeout=1.0)

    assert not reader.is_alive()
    assert not failures
    assert not writer.worker_alive


@pytest.mark.parametrize('fps', (20.0, 30.0))
def test_camera_progress_continues_during_bounded_encoder_stall(fps):
    """Exercise real waits separately from pure throughput unit tests."""
    source = ControlledSource()
    latest = LatestImageSource(source)
    output = SlowWriter(delay=0.20, delay_writes=1)
    writer = AsyncPacedVideoWriter(output, fps)
    stop = threading.Event()

    def produce():
        period = 1.0 / fps
        sequence = 0
        deadline = time.monotonic()
        while not stop.is_set():
            sequence += 1
            deadline += period
            source.offer(sequence, deadline)
            stop.wait(max(0.0, deadline - time.monotonic()))

    producer = threading.Thread(target=produce)
    producer.start()
    consumed = 0
    deadline = time.monotonic() + 0.6
    try:
        while time.monotonic() < deadline:
            frame = latest.read()
            if frame is not None:
                consumed += 1
                assert writer.write(frame)
            time.sleep(0.002)
    finally:
        stop.set()
        producer.join(timeout=1.0)
        latest.close()
        writer.release()

    assert consumed >= int(fps * 0.45)
    assert writer.submitted_frames == consumed
    assert writer.replaced_frames > 0
    assert not latest.worker_alive
    assert not writer.worker_alive


def test_async_writer_closes_playable_fixed_rate_mp4(tmp_path):
    destination = tmp_path / 'async.mp4'
    raw = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (32, 24)
    )
    assert raw.isOpened()
    started = time.monotonic()
    writer = AsyncPacedVideoWriter(raw, 20.0)
    for value in range(6):
        assert writer.write(np.full((24, 32, 3), value * 30, np.uint8))
        time.sleep(0.05)
    elapsed = time.monotonic() - started
    writer.release()

    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened()
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    assert decoded == writer.frames_written
    assert decoded >= 5
    assert abs(decoded / 20.0 - elapsed) <= 2.0 / 20.0
    assert not writer.worker_alive
