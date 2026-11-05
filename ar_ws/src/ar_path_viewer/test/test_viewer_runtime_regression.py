from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.logging import get_logger

from ar_path_viewer.diagnostics import DiagnosticReporter
from ar_path_viewer.localization_monitor import (
    AmclObservation,
    LocalizationMonitor,
    LocalizationThresholds,
)
from ar_path_viewer.path_adapter import PathStore
from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer import viewer_node


class RecordingLogger:
    def __init__(self):
        self.records = []

    def info(self, message):
        self.records.append(("info", message))

    def warning(self, message):
        self.records.append(("warning", message))


class ImageSource:
    def __init__(self, width=640, height=480):
        self.image = np.zeros((height, width, 3), np.uint8)
        self.close_count = 0

    def read(self):
        return self.image.copy()

    def close(self):
        self.close_count += 1


class RecordingWriter:
    def __init__(self):
        self.frames = []
        self.release_count = 0

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.release_count += 1


class FakeViewer:
    pass


def make_viewer(clock=None):
    fake = FakeViewer()
    active_clock = clock or __import__("time").monotonic
    fake._shutdown_requested = False
    fake._resources_released = False
    fake._intrinsics = CameraIntrinsics()
    fake._mount = CameraMount()
    fake._near = 0.1
    fake._path_store = PathStore(clock=active_clock)
    fake._image_source = ImageSource()
    fake._camera_error = ""
    fake._config_error = ""
    fake._base_frame = "map"
    fake._color = (0, 255, 0)
    fake._thickness = 5
    fake._display = False
    fake._writer = RecordingWriter()
    fake._points_in_base = lambda points, frame: points
    fake._check_localization_tf = lambda: (True, "", (0.0, 0.0, 0.0))
    fake.logger = RecordingLogger()
    fake.get_logger = lambda: fake.logger
    fake._diagnostics = DiagnosticReporter(True, fake.logger, clock=active_clock)
    fake._localization = LocalizationMonitor(
        LocalizationThresholds(recovery_samples=1, recovery_sec=0.0),
        clock=active_clock,
    )
    fake._localization.observe(
        AmclObservation(
            0.0, 0.0, 0.0, tuple(np.eye(6).ravel() * 0.01), "map", 100.0
        ),
        100.0,
    )
    return fake


def path_message(points, frame="map", stamp=1.0):
    poses = [
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=float(x), y=float(y), z=float(z))
            )
        )
        for x, y, z in points
    ]
    seconds = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame,
            stamp=SimpleNamespace(sec=seconds, nanosec=int((stamp - seconds) * 1e9)),
        ),
        poses=poses,
    )


def capture_states(monkeypatch):
    calls = []

    def render(image, segments, state, *args, **kwargs):
        calls.append((state, list(segments)))
        return image.copy()

    monkeypatch.setattr(viewer_node, "render_frame", render)
    return calls


def test_start_before_path_then_adopt_and_update_route_while_recording(monkeypatch):
    states = capture_states(monkeypatch)
    fake = make_viewer()

    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("NO PATH", [])

    first = path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    viewer_node.ArPathViewerNode._on_path(fake, first)
    viewer_node.ArPathViewerNode._tick(fake)
    first_segments = states[-1][1]
    assert states[-1][0] == "ACTIVE"
    assert first_segments

    second = path_message(((1.0, 0.0, 0.0), (4.0, -1.0, 0.0)), stamp=2.0)
    viewer_node.ArPathViewerNode._on_path(fake, second)
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1][0] == "ACTIVE"
    assert states[-1][1] != first_segments
    assert fake._path_store.snapshot().generation == 2

    viewer_node.ArPathViewerNode._on_path(fake, path_message((), stamp=3.0))
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("NO PATH", [])
    assert len(fake._writer.frames) == 4


def test_plan_is_adopted_and_processing_continues_with_real_rclpy_logger(monkeypatch):
    states = capture_states(monkeypatch)
    fake = make_viewer()
    fake.logger = get_logger("ar_path_viewer_plan_reception_regression")
    fake._diagnostics = DiagnosticReporter(True, fake.logger)
    fake._diagnostics.state_changed("NO PATH", "waiting for /plan")

    first = path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    viewer_node.ArPathViewerNode._on_path(fake, first)
    first_snapshot = fake._path_store.snapshot()
    assert first_snapshot is not None
    assert first_snapshot.generation == 1
    assert np.array_equal(first_snapshot.points, ((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1][0] == "ACTIVE"
    assert states[-1][1]

    second = path_message(((2.0, 0.5, 0.0), (6.0, 1.0, 0.0)), stamp=2.0)
    viewer_node.ArPathViewerNode._on_path(fake, second)
    second_snapshot = fake._path_store.snapshot()
    assert second_snapshot is not None
    assert second_snapshot.generation == 2
    assert fake._diagnostics.path_receive_count == 2
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1][0] == "ACTIVE"
    assert states[-1][1]


def test_invalid_path_and_tf_failure_never_draw_green_segments(monkeypatch):
    states = capture_states(monkeypatch)
    fake = make_viewer()
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((float("nan"), 0.0, 0.0),), stamp=2.0)
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("NO PATH", [])

    viewer_node.ArPathViewerNode._on_path(
        fake,
        path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)), frame="map", stamp=3.0),
    )
    fake._base_frame = "robot/base_footprint"
    fake._points_in_base = lambda points, frame: (_ for _ in ()).throw(
        RuntimeError("TF lookup timed out")
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("TF UNAVAILABLE", [])
    assert any("TF lookup timed out" in message for _, message in fake.logger.records)


def test_stale_and_out_of_view_paths_draw_no_segments(monkeypatch):
    states = capture_states(monkeypatch)
    now = [10.0]
    fake = make_viewer(clock=lambda: now[0])
    fake._path_store = PathStore(timeout_sec=6.5, clock=lambda: now[0])
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((-5.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("PATH OUT OF VIEW", [])
    now[0] = 16.6
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("PATH STALE", [])


def test_no_path_and_path_stale_take_priority_over_localization_state(monkeypatch):
    states = capture_states(monkeypatch)
    now = [10.0]
    fake = make_viewer(clock=lambda: now[0])
    fake._localization = LocalizationMonitor(clock=lambda: now[0])
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("NO PATH", [])

    fake._path_store = PathStore(timeout_sec=1.0, clock=lambda: now[0])
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    now[0] = 11.1
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("PATH STALE", [])


def test_amcl_missing_and_degraded_localization_hide_segments(monkeypatch):
    states = capture_states(monkeypatch)
    now = [10.0]
    fake = make_viewer(clock=lambda: now[0])
    fake._localization = LocalizationMonitor(clock=lambda: now[0])
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("LOCALIZING", [])

    fake._localization = LocalizationMonitor(
        LocalizationThresholds(recovery_samples=1, recovery_sec=0.0),
        clock=lambda: now[0],
    )
    fake._localization.observe(
        AmclObservation(
            0.0, 0.0, 0.0, tuple(np.eye(6).ravel() * 0.01), "map", 100.0
        ),
        100.0,
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1][0] == "ACTIVE"
    fake._localization.reject("test localization degradation")
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1] == ("LOCALIZATION UNSTABLE", [])


def test_path_to_first_waypoint_remains_visible_when_localization_is_stable(monkeypatch):
    states = capture_states(monkeypatch)
    fake = make_viewer()
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((2.0, 0.0, 0.0), (6.0, 0.5, 0.0)))
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert states[-1][0] == "ACTIVE"
    assert states[-1][1]


def test_recorded_frame_removes_green_path_when_localization_degrades():
    fake = make_viewer()
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    viewer_node.ArPathViewerNode._tick(fake)
    active_frame = fake._writer.frames[-1]
    active_green = np.count_nonzero(
        (active_frame[80:, :, 1] > 180)
        & (active_frame[80:, :, 0] < 80)
        & (active_frame[80:, :, 2] < 80)
    )
    assert active_green > 0

    fake._localization.reject("test localization degradation")
    viewer_node.ArPathViewerNode._tick(fake)
    unstable_frame = fake._writer.frames[-1]
    unstable_green = np.count_nonzero(
        (unstable_frame[80:, :, 1] > 180)
        & (unstable_frame[80:, :, 0] < 80)
        & (unstable_frame[80:, :, 2] < 80)
    )
    assert unstable_green == 0


@pytest.mark.parametrize(
    "key_code",
    (ord("q"), ord("Q"), 0x00790000, 0x0000FFC7, 0x01000039),
)
def test_q_and_backend_specific_f10_codes_request_exit(key_code):
    assert viewer_node.is_exit_key(key_code)


def test_unrelated_or_missing_keys_do_not_request_exit():
    assert not viewer_node.is_exit_key(-1)
    assert not viewer_node.is_exit_key(ord("y"))


def test_keyboard_exit_records_frame_and_releases_resources_once(monkeypatch):
    fake = make_viewer()
    writer = fake._writer
    source = fake._image_source
    fake._display = True
    fake._release_resources = lambda: viewer_node.ArPathViewerNode._release_resources(fake)
    fake._request_shutdown = lambda reason: viewer_node.ArPathViewerNode._request_shutdown(
        fake, reason
    )
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    shutdown_calls = []
    monkeypatch.setattr(viewer_node.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(viewer_node.cv2, "waitKeyEx", lambda delay: ord("q"))
    monkeypatch.setattr(viewer_node.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        viewer_node, "shutdown_ros_context", lambda: shutdown_calls.append(True)
    )

    viewer_node.ArPathViewerNode._tick(fake)
    assert len(writer.frames) == 1
    assert writer.release_count == 1
    assert source.close_count == 1
    assert shutdown_calls == [True]
    fake._request_shutdown("duplicate")
    fake._release_resources()
    assert writer.release_count == 1
    assert source.close_count == 1
    assert shutdown_calls == [True]


def test_localization_tf_requires_a_fresh_timestamp():
    fake = make_viewer()
    fake._base_frame = "robot/base_footprint"
    fake._ros_now_sec = lambda: 100.0

    class Buffer:
        stamp = 100.0

        def lookup_transform(self, *args, **kwargs):
            seconds = int(self.stamp)
            return SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(
                        sec=seconds,
                        nanosec=int((self.stamp - seconds) * 1e9),
                    )
                ),
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=0.0, y=0.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )

    fake._tf_buffer = Buffer()
    available, detail, pose = viewer_node.ArPathViewerNode._check_localization_tf(fake)
    assert available
    assert detail == ""
    assert pose == (0.0, 0.0, 0.0)

    fake._tf_buffer.stamp = 90.0
    available, detail, pose = viewer_node.ArPathViewerNode._check_localization_tf(fake)
    assert not available
    assert "TF stale" in detail
    assert pose is None


def test_shutdown_uses_try_shutdown(monkeypatch):
    calls = []
    monkeypatch.setattr(viewer_node.rclpy, "try_shutdown", lambda: calls.append(True))
    viewer_node.shutdown_ros_context()
    assert calls == [True]
