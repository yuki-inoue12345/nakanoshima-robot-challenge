from types import SimpleNamespace

import numpy as np

from ar_path_viewer.diagnostics import DiagnosticReporter
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

    def read(self):
        return self.image.copy()


class RecordingWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class FakeViewer:
    pass


def make_viewer(clock=None):
    fake = FakeViewer()
    fake._intrinsics = CameraIntrinsics()
    fake._mount = CameraMount()
    fake._near = 0.1
    fake._path_store = PathStore(clock=(clock or __import__("time").monotonic))
    fake._image_source = ImageSource()
    fake._camera_error = ""
    fake._config_error = ""
    fake._base_frame = "map"
    fake._color = (0, 255, 0)
    fake._thickness = 5
    fake._display = False
    fake._writer = RecordingWriter()
    fake._points_in_base = lambda points, frame: points
    fake.logger = RecordingLogger()
    fake._diagnostics = DiagnosticReporter(True, fake.logger, clock=(clock or __import__("time").monotonic))
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


def test_shutdown_uses_try_shutdown(monkeypatch):
    calls = []
    monkeypatch.setattr(viewer_node.rclpy, "try_shutdown", lambda: calls.append(True))
    viewer_node.shutdown_ros_context()
    assert calls == [True]
