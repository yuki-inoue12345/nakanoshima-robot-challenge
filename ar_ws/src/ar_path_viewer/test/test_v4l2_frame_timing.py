"""ROS-free P1-1 tests: reject buffered OpenCV frames, trust kernel metadata conditionally."""

import ctypes

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
