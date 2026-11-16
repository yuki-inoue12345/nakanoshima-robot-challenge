"""ROS-free P1-1 tests: reject buffered OpenCV frames, trust kernel metadata conditionally."""

import ctypes
import errno

import pytest

from ar_path_viewer.costmap_overlay import (
    CostmapCameraSyncSettings, estimate_image_frame_timing,
)
from ar_path_viewer.image_source import make_image_source
from ar_path_viewer.v4l2_timestamp_source import (
    _Buffer, _Format, _RequestBuffers, _S_FMT,
    _TIMESTAMP_MONOTONIC, _TIMESTAMP_SOURCE_SOE,
    timestamp_from_v4l2_buffer,
)


def _timing(*, source_kind="camera_timestamped", source_monotonic=100.00,
            start_ros=20.0, end_ros=20.01, start_mono=100.0,
            end_mono=100.01, timestamp_source="v4l2_monotonic_soe",
            previous=None):
    return estimate_image_frame_timing(
        ros_started_sec=start_ros, ros_finished_sec=end_ros,
        monotonic_started_sec=start_mono,
        monotonic_finished_sec=end_mono,
        source_kind=source_kind,
        source_timestamp_monotonic_sec=source_monotonic,
        timestamp_source=timestamp_source,
        previous_timing=previous,
        settings=CostmapCameraSyncSettings(),
    )


def test_fast_opencv_read_of_an_old_buffer_never_authorizes_ar():
    # The OpenCV read operation is fast, but the image could be old.
    value = _timing(source_kind="camera", source_monotonic=None)
    assert not value.usable
    assert value.reason == "camera_timestamp_unverified_opencv"


@pytest.mark.parametrize("lag", (0.0, 0.04, 0.09))
def test_v4l2_kernel_timestamp_converts_monotonic_to_ros(lag):
    value = _timing(source_monotonic=100.01 - lag)
    assert value.usable
    assert value.stamp_sec == pytest.approx(20.01 - lag)


def test_v4l2_eof_is_not_evidence_of_exposure_time():
    value = _timing(timestamp_source="v4l2_monotonic_eof")
    assert not value.usable
    assert value.reason == "camera_eof_exposure_offset_unverified"


def test_old_kernel_buffer_stays_hidden_even_when_read_is_fast():
    value = _timing(source_monotonic=99.6)
    assert not value.usable
    assert value.reason == "camera_buffer_stale"


def test_bogus_or_missing_kernel_timestamp_stays_hidden():
    assert _timing(source_monotonic=None).reason == "camera_timestamp_missing"
    assert _timing(timestamp_source="user_injected").reason == (
        "camera_timestamp_source_unverified"
    )
    assert _timing(source_monotonic=100.03).reason == (
        "camera_timestamp_from_future"
    )


def test_kernel_timestamp_must_advance_between_accepted_frames():
    first = _timing()
    second = _timing(
        source_monotonic=100.0, start_ros=20.05, end_ros=20.06,
        start_mono=100.05, end_mono=100.06, previous=first,
    )
    assert not second.usable
    assert second.reason == "camera_timestamp_not_advancing"


def test_clock_forward_jump_80ms_during_read_is_rejected():
    value = _timing(end_ros=20.09)
    assert not value.usable
    assert value.reason.startswith("ros_monotonic_clock_mismatch")


def test_clock_forward_jump_between_frames_is_rejected():
    first = _timing()
    second = _timing(
        start_ros=20.14, end_ros=20.15,
        start_mono=100.05, end_mono=100.06,
        source_monotonic=100.06, previous=first,
    )
    assert not second.usable
    assert second.reason == "ros_monotonic_clock_jump_between_frames"


def test_v4l2_abi_matches_linux_x86_64_headers():
    assert ctypes.sizeof(_Buffer) == 88
    assert ctypes.sizeof(_Format) == 208
    assert ctypes.sizeof(_RequestBuffers) == 20
    assert _S_FMT == 0xC0D05605


def test_v4l2_buffer_monotonic_provenance_is_required():
    buf = _Buffer()
    buf.timestamp.tv_sec = 100
    buf.timestamp.tv_usec = 40000
    buf.flags = _TIMESTAMP_MONOTONIC
    assert timestamp_from_v4l2_buffer(buf, monotonic_now=100.05) == (
        100.04, "v4l2_monotonic_eof"
    )
    buf.flags |= _TIMESTAMP_SOURCE_SOE
    assert timestamp_from_v4l2_buffer(buf, monotonic_now=100.05)[1] == (
        "v4l2_monotonic_soe"
    )
    buf.flags = 0
    with pytest.raises(ValueError, match="not_monotonic"):
        timestamp_from_v4l2_buffer(buf, monotonic_now=100.05)
    buf.flags = _TIMESTAMP_MONOTONIC | 0x00020000
    with pytest.raises(ValueError, match="unknown_timestamp_source"):
        timestamp_from_v4l2_buffer(buf, monotonic_now=100.05)


def test_timestamped_camera_uses_dedicated_source_only(monkeypatch):
    from ar_path_viewer import v4l2_timestamp_source
    marker = object()
    monkeypatch.setattr(
        v4l2_timestamp_source, "V4L2TimestampCameraSource",
        lambda *args: marker,
    )
    assert make_image_source("camera_timestamped", "/dev/video0", 640, 480) is marker


def test_v4l2_mock_dequeue_keeps_image_and_stamp_from_same_buffer(monkeypatch):
    """Exercise the actual ioctl/mmap/decode path without using a real C920."""
    import cv2
    import numpy as np
    from ar_path_viewer import v4l2_timestamp_source as v

    ok, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))
    assert ok
    jpeg_bytes = jpeg.tobytes()
    qbuf_calls = []
    dq_calls = []
    closed = []

    class FakeMap:
        def __len__(self):
            return len(jpeg_bytes)

        def __getitem__(self, part):
            return jpeg_bytes[part]

        def close(self):
            closed.append('map')

    def ioctl(fd, request, payload, mutate=True):
        assert fd == 123
        if request == v._REQBUFS:
            payload.count = 2
        elif request == v._QUERYBUF:
            payload.length = len(jpeg_bytes)
            payload.m.offset = payload.index * 4096
        elif request == v._QBUF:
            qbuf_calls.append(payload.index)
        elif request == v._DQBUF:
            if dq_calls:
                raise BlockingIOError(errno.EAGAIN, 'no more ready buffers')
            dq_calls.append(1)
            payload.index = 0
            payload.bytesused = len(jpeg_bytes)
            payload.flags = v._TIMESTAMP_MONOTONIC
            payload.timestamp.tv_sec = 100
            payload.timestamp.tv_usec = 40_000
            payload.sequence = 1
        return 0

    monkeypatch.setattr(v.os, 'open', lambda *args: 123)
    monkeypatch.setattr(v.os, 'close', lambda fd: closed.append(fd))
    monkeypatch.setattr(v.fcntl, 'ioctl', ioctl)
    monkeypatch.setattr(v.mmap, 'mmap', lambda *args, **kwargs: FakeMap())
    monkeypatch.setattr(v.select, 'select', lambda *args: ([123], [], []))
    monkeypatch.setattr(v.time, 'monotonic', lambda: 100.05)

    camera = v.V4L2TimestampCameraSource('/dev/video0', 640, 480)
    try:
        frame = camera.read()
        assert frame.shape == (480, 640, 3)
        assert camera.frame_timestamp_monotonic == pytest.approx(100.04)
        assert camera.frame_timestamp_kind == 'v4l2_monotonic_eof'
        assert qbuf_calls == [0, 1, 0]
    finally:
        camera.close()
    assert closed.count('map') == 2
    assert 123 in closed


def test_v4l2_backlog_drains_to_freshest_matching_image_and_soe_stamp(
    monkeypatch,
):
    """A 4-buffer backlog must not leave the Viewer three frames behind."""
    import cv2
    import numpy as np
    from ar_path_viewer import v4l2_timestamp_source as v

    jpeg_by_index = []
    for value in (20, 60, 100, 180):
        ok, jpeg = cv2.imencode(
            '.jpg', np.full((480, 640, 3), value, np.uint8)
        )
        assert ok
        jpeg_by_index.append(jpeg.tobytes())
    pending = [
        (0, 1, 100.80),
        (1, 2, 100.85),
        (2, 3, 100.90),
        (3, 4, 100.97),
    ]
    qbuf_calls = []

    class FakeMap:
        def __init__(self, index):
            self.data = jpeg_by_index[index]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, part):
            return self.data[part]

        def close(self):
            pass

    def ioctl(_fd, request, payload, mutate=True):
        if request == v._REQBUFS:
            payload.count = 4
        elif request == v._QUERYBUF:
            payload.length = len(jpeg_by_index[payload.index])
            payload.m.offset = payload.index * 4096
        elif request == v._QBUF:
            qbuf_calls.append(payload.index)
        elif request == v._DQBUF:
            if not pending:
                raise BlockingIOError(errno.EAGAIN, 'queue drained')
            index, sequence, stamp = pending.pop(0)
            payload.index = index
            payload.bytesused = len(jpeg_by_index[index])
            payload.flags = v._TIMESTAMP_MONOTONIC | v._TIMESTAMP_SOURCE_SOE
            payload.timestamp.tv_sec = int(stamp)
            payload.timestamp.tv_usec = int(round((stamp - int(stamp)) * 1e6))
            payload.sequence = sequence
        return 0

    monkeypatch.setattr(v.os, 'open', lambda *args: 123)
    monkeypatch.setattr(v.os, 'close', lambda _fd: None)
    monkeypatch.setattr(v.fcntl, 'ioctl', ioctl)
    monkeypatch.setattr(
        v.mmap, 'mmap',
        lambda *args, **kwargs: FakeMap(kwargs['offset'] // 4096),
    )
    monkeypatch.setattr(v.select, 'select', lambda *args: ([123], [], []))
    monkeypatch.setattr(v.time, 'monotonic', lambda: 101.0)

    camera = v.V4L2TimestampCameraSource('/dev/video0', 640, 480)
    try:
        frame = camera.read()
        assert float(frame.mean()) == pytest.approx(180.0, abs=3.0)
        assert camera.frame_timestamp_monotonic == pytest.approx(100.97)
        assert camera.frame_timestamp_kind == 'v4l2_monotonic_soe'
        assert camera.last_read_metrics.sequence == 4
        assert camera.last_read_metrics.dequeued_buffers == 4
        assert camera.last_read_metrics.discarded_buffers == 3
        assert camera.last_read_metrics.total_discarded_buffers == 3
        assert camera.last_read_metrics.transport_age_sec == pytest.approx(0.03)
        # Four initial QBUFs, three intentional stale drops, one selected frame.
        assert qbuf_calls == [0, 1, 2, 3, 0, 1, 2, 3]
    finally:
        camera.close()

    fresh = _timing(source_monotonic=100.97, start_mono=100.99,
                    end_mono=101.0)
    stale = _timing(source_monotonic=100.80, start_mono=100.99,
                    end_mono=101.0)
    assert fresh.usable
    assert stale.reason == 'camera_buffer_stale'


def test_v4l2_twenty_hz_backlog_is_bounded_under_processing_jitter(monkeypatch):
    """600 loaded ticks select a fresh frame instead of accumulating latency."""
    import cv2
    import numpy as np
    from ar_path_viewer import v4l2_timestamp_source as v

    ok, jpeg = cv2.imencode(
        '.jpg', np.full((480, 640, 3), 90, np.uint8)
    )
    assert ok
    jpeg_bytes = jpeg.tobytes()
    clock = [100.0]
    next_sequence = [1]
    pending = []

    class FakeMap:
        def __len__(self):
            return len(jpeg_bytes)

        def __getitem__(self, part):
            return jpeg_bytes[part]

        def close(self):
            pass

    def select_ready(*_args):
        base = clock[0]
        for offset in (-0.15, -0.10, -0.05, 0.0):
            pending.append((next_sequence[0] % 4, next_sequence[0], base + offset))
            next_sequence[0] += 1
        return [123], [], []

    def ioctl(_fd, request, payload, mutate=True):
        if request == v._REQBUFS:
            payload.count = 4
        elif request == v._QUERYBUF:
            payload.length = len(jpeg_bytes)
            payload.m.offset = payload.index * 4096
        elif request == v._DQBUF:
            if not pending:
                raise BlockingIOError(errno.EAGAIN, 'queue drained')
            index, sequence, stamp = pending.pop(0)
            payload.index = index
            payload.bytesused = len(jpeg_bytes)
            payload.flags = v._TIMESTAMP_MONOTONIC | v._TIMESTAMP_SOURCE_SOE
            payload.timestamp.tv_sec = int(stamp)
            payload.timestamp.tv_usec = int(round((stamp - int(stamp)) * 1e6))
            payload.sequence = sequence
        return 0

    monkeypatch.setattr(v.os, 'open', lambda *args: 123)
    monkeypatch.setattr(v.os, 'close', lambda _fd: None)
    monkeypatch.setattr(v.fcntl, 'ioctl', ioctl)
    monkeypatch.setattr(v.mmap, 'mmap', lambda *args, **kwargs: FakeMap())
    monkeypatch.setattr(v.select, 'select', select_ready)
    monkeypatch.setattr(v.time, 'monotonic', lambda: clock[0])

    camera = v.V4L2TimestampCameraSource('/dev/video0', 640, 480)
    try:
        for index in range(600):
            assert camera.read() is not None
            metrics = camera.last_read_metrics
            assert metrics.dequeued_buffers == 4
            assert metrics.discarded_buffers == 3
            assert metrics.transport_age_sec == pytest.approx(0.0, abs=1e-6)
            timing = _timing(
                source_monotonic=camera.frame_timestamp_monotonic,
                start_mono=clock[0], end_mono=clock[0],
            )
            assert timing.usable
            # Nominal 20 Hz, with periodic processing-load excursions.
            clock[0] += 0.11 if index and index % 50 == 0 else 0.05
        assert camera.last_read_metrics.total_discarded_buffers == 1800
        assert camera.last_read_metrics.sequence_gap == 3
    finally:
        camera.close()


def test_v4l2_dequeue_failure_is_explicit_and_never_reuses_old_timestamp(
    monkeypatch,
):
    from ar_path_viewer import v4l2_timestamp_source as v

    camera = object.__new__(v.V4L2TimestampCameraSource)
    camera._fd = 123
    camera._mappings = [bytearray(32), bytearray(32)]
    camera._running = True
    camera._last_sequence = 7
    camera._discarded_buffer_total = 0
    camera.frame_timestamp_monotonic = 99.0
    camera.frame_timestamp_kind = 'v4l2_monotonic_soe'
    camera.last_read_metrics = v.V4L2ReadMetrics()

    monkeypatch.setattr(v.select, 'select', lambda *args: ([123], [], []))
    monkeypatch.setattr(v.time, 'monotonic', lambda: 100.0)

    def failed_ioctl(_fd, request, _payload, mutate=True):
        assert request == v._DQBUF
        raise OSError(errno.EIO, 'camera disconnected')

    monkeypatch.setattr(v.fcntl, 'ioctl', failed_ioctl)

    assert camera.read() is None
    assert camera.frame_timestamp_monotonic is None
    assert camera.frame_timestamp_kind == ''
    assert camera.last_read_metrics.error_reason == (
        f'v4l2_dequeue_failed_errno_{errno.EIO}'
    )
