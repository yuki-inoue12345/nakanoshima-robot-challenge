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
from ar_path_viewer.offscreen_guidance import GuidanceSettings
from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.reference_path import ReferencePathGate
from ar_path_viewer import viewer_node
from ar_path_viewer.viewer_logic import PathDisplayGeometry
from ar_path_viewer.waypoint_progress import WaypointProgressGate


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


class RecordingDisplayWindow:
    def __init__(self, key_code=-1, actions=()):
        self.key_code = key_code
        self.actions = list(actions)
        self.frames = []
        self.statuses = []
        self.close_count = 0

    def show_and_poll(self, frame):
        self.frames.append(frame.copy())
        return self.key_code

    def consume_actions(self):
        actions = self.actions
        self.actions = []
        return actions

    def set_status(self, text, success):
        self.statuses.append((text, success))

    def close(self):
        self.close_count += 1


class FakeViewer:
    pass


class RecordingLogGui:
    def __init__(self, copy_success=True):
        self.copy_success = copy_success
        self.shown = []
        self.copied = []
        self.poll_count = 0
        self.close_count = 0

    def show(self, content, refresh_source=None):
        self.shown.append((content, refresh_source))
        return True, "shown"

    def copy(self, content):
        self.copied.append(content)
        if self.copy_success:
            return True, "copied"
        return False, "clipboard unavailable"

    def poll(self):
        self.poll_count += 1

    def close(self):
        self.close_count += 1


class MemorySessionLog:
    def __init__(self):
        self.records = []
        self.closed = False

    def text(self):
        return "camera_configuration test\npath_projection test\n"

    def copy_text(self, max_chars):
        return self.text()[:max_chars]

    def record(self, level, message):
        self.records.append((level, message))

    def close(self):
        self.closed = True


def make_viewer(clock=None):
    fake = FakeViewer()
    active_clock = clock or __import__("time").monotonic
    fake._shutdown_requested = False
    fake._resources_released = False
    fake._intrinsics = CameraIntrinsics()
    fake._source_intrinsics = fake._intrinsics
    fake._undistorter = None
    fake._mount = CameraMount()
    fake._near = 0.1
    # Existing runtime tests retain the former ground-plane baseline. Dedicated
    # tests below exercise the new elevated mode explicitly.
    fake._path_display_geometry = PathDisplayGeometry("ground", 0.5)
    fake._calibration_profile_label = ""
    fake._guidance_settings = GuidanceSettings()
    fake._guidance_color = (0, 165, 255)
    fake._path_store = PathStore(clock=active_clock)
    fake._reference_enabled = False
    fake._waypoint_distance_enabled = True
    fake._waypoint_progress_gate = WaypointProgressGate()
    fake._image_source = ImageSource()
    fake._camera_error = ""
    fake._config_error = ""
    fake._base_frame = "map"
    fake._color = (0, 255, 0)
    fake._thickness = 5
    fake._display = False
    fake._display_window = None
    fake._writer = RecordingWriter()
    fake._points_in_base = lambda points, frame: points
    fake._check_localization_tf = lambda: (True, "", (0.0, 0.0, 0.0))
    fake._ros_now_sec = lambda: 100.0
    fake.logger = RecordingLogger()
    fake.get_logger = lambda: fake.logger
    fake._log_info = lambda message: fake.logger.info(message)
    fake._log_warning = lambda message: fake.logger.warning(message)
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
            header=SimpleNamespace(frame_id=frame),
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


def reference_follower_message(
    *, stamp=100.0, route="a" * 64, index=1, generation=4,
    activity=1, allowed=True, mode="to_pose", number=None,
):
    seconds = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="map",
            stamp=SimpleNamespace(
                sec=seconds, nanosec=int((stamp - seconds) * 1e9)
            ),
        ),
        schema_version=1,
        route_sha256=route,
        next_waypoint_index=index,
        next_waypoint_number=20 + index if number is None else number,
        navigation_mode=mode,
        goal_generation=generation,
        activity=activity,
        reference_allowed=allowed,
        safe_section_end_index=7,
        waypoint_count=8,
    )


def reference_path_message(
    points, *, stamp=100.0, route="a" * 64, index=1, generation=4,
    valid=True, reason="",
):
    seconds = int(stamp)
    stamp_message = SimpleNamespace(
        sec=seconds, nanosec=int((stamp - seconds) * 1e9)
    )
    planar_length = sum(
        np.linalg.norm(np.asarray(b[:2]) - np.asarray(a[:2]))
        for a, b in zip(points, points[1:])
    )
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map", stamp=stamp_message),
        schema_version=1,
        valid=valid,
        invalid_reason=reason,
        route_sha256=route,
        next_waypoint_index=index,
        next_waypoint_number=20 + index,
        navigation_mode="to_pose",
        goal_generation=generation,
        safe_section_end_index=7,
        waypoint_count=8,
        planner_generation=5,
        follower_state_stamp=stamp_message,
        path_length_m=planar_length,
        path=path_message(points, stamp=stamp),
    )


def enable_reference(fake, now=100.0):
    fake._reference_enabled = True
    fake._reference_gate = ReferencePathGate(clock=lambda: 10.0)
    fake._reference_color = (255, 255, 0)
    fake._reference_thickness = 3
    fake._reference_dash_px = 14.0
    fake._reference_gap_px = 9.0
    fake._ros_now_sec = lambda: now
    follower = reference_follower_message(stamp=now)
    viewer_node.ArPathViewerNode._on_reference_follower_state(fake, follower)
    return follower


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
    log_text = "\n".join(message for _, message in fake.logger.records)
    assert "path_length_m=" in log_text
    assert "endpoint_distance_m=" in log_text
    assert "camera_forward_path_length_m=" in log_text
    assert "visible_path_length_m=" in log_text
    assert "nonvisible_forward_path_length_m=" in log_text
    assert "visible_depth_max_m=" in log_text
    assert "visible_pixel_length=" in log_text

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


def test_plan_remaining_distance_tracks_only_fresh_follower_goal(monkeypatch):
    rendered = []

    def render(image, segments, state, *args, **kwargs):
        rendered.append((state, kwargs.get("waypoint_status", "")))
        return image.copy()

    monkeypatch.setattr(viewer_node, "render_frame", render)
    fake = make_viewer()

    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        reference_follower_message(index=2, number=12, generation=4),
    )
    viewer_node.ArPathViewerNode._on_path(
        fake,
        path_message(((0.1, 0.0, 0.0), (2.5, 0.0, 0.0))),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert rendered[-1][1] == "PLAN LEFT: 2.4 m"

    # A changed Follower goal invalidates the old plan immediately.
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        reference_follower_message(index=3, number=13, generation=5),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert rendered[-1][1] == ""

    viewer_node.ArPathViewerNode._on_path(
        fake,
        path_message(((0.1, 0.0, 0.0), (0.92, 0.0, 0.0)), stamp=2.0),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert rendered[-1][1] == "PLAN LEFT: 82 cm"

    # Pause/person-search states clear the read-only display immediately.
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        reference_follower_message(
            index=3, number=13, generation=5, activity=4, allowed=False
        ),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert rendered[-1][1] == ""


def test_through_poses_shows_only_plan_remaining_distance(monkeypatch):
    waypoint_statuses = []

    def render(image, segments, state, *args, **kwargs):
        waypoint_statuses.append(kwargs.get("waypoint_status", ""))
        return image.copy()

    monkeypatch.setattr(viewer_node, "render_frame", render)
    fake = make_viewer()
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake, reference_follower_message(mode="through_poses")
    )
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((0.1, 0.0, 0.0), (4.0, 0.0, 0.0)))
    )
    viewer_node.ArPathViewerNode._tick(fake)
    assert waypoint_statuses[-1] == "PLAN LEFT: 3.9 m"


def test_elevated_mode_records_more_of_a_short_real_path_than_ground_mode():
    points = tuple(
        (float(x), 0.2 * float(x), 0.0) for x in np.linspace(0.4, 2.8, 49)
    )
    ground = make_viewer()
    viewer_node.ArPathViewerNode._on_path(ground, path_message(points))
    viewer_node.ArPathViewerNode._tick(ground)
    ground_green = np.count_nonzero(
        (ground._writer.frames[-1][:, :, 1] > 180)
        & (ground._writer.frames[-1][:, :, 0] < 80)
        & (ground._writer.frames[-1][:, :, 2] < 80)
    )

    elevated = make_viewer()
    elevated._path_display_geometry = PathDisplayGeometry("elevated", 0.5)
    viewer_node.ArPathViewerNode._on_path(elevated, path_message(points))
    viewer_node.ArPathViewerNode._tick(elevated)
    elevated_green = np.count_nonzero(
        (elevated._writer.frames[-1][:, :, 1] > 180)
        & (elevated._writer.frames[-1][:, :, 0] < 80)
        & (elevated._writer.frames[-1][:, :, 2] < 80)
    )

    assert elevated_green > ground_green


def test_reference_path_is_cyan_in_recording_and_does_not_replace_green_plan():
    fake = make_viewer()
    enable_reference(fake)
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        reference_path_message(
            ((1.0, 0.6, 0.0), (4.0, 0.6, 0.0), (10.0, 0.6, 0.0))
        ),
    )
    viewer_node.ArPathViewerNode._on_path(
        fake,
        path_message(((1.0, -0.4, 0.0), (5.0, -0.4, 0.0))),
    )

    viewer_node.ArPathViewerNode._tick(fake)

    frame = fake._writer.frames[-1]
    cyan = (
        (frame[:, :, 0] > 150)
        & (frame[:, :, 1] > 150)
        & (frame[:, :, 2] < 100)
    )
    green = (
        (frame[:, :, 0] < 100)
        & (frame[:, :, 1] > 150)
        & (frame[:, :, 2] < 100)
    )
    assert np.count_nonzero(cyan) > 20
    assert np.count_nonzero(green) > 20
    assert fake._path_store.snapshot() is not None
    assert fake._reference_gate.path_store.snapshot() is not None


def test_pause_and_planner_invalidation_remove_reference_from_next_recorded_frame():
    fake = make_viewer()
    enable_reference(fake)
    points = ((1.0, 0.5, 0.0), (5.0, 0.5, 0.0), (10.0, 0.5, 0.0))
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake, reference_path_message(points)
    )
    viewer_node.ArPathViewerNode._tick(fake)
    first = fake._writer.frames[-1]
    assert np.count_nonzero(
        (first[:, :, 0] > 150)
        & (first[:, :, 1] > 150)
        & (first[:, :, 2] < 100)
    ) > 20

    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        reference_follower_message(activity=4, allowed=False),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    paused = fake._writer.frames[-1]
    assert np.count_nonzero(
        (paused[:, :, 0] > 150)
        & (paused[:, :, 1] > 150)
        & (paused[:, :, 2] < 100)
    ) == 0

    enable_reference(fake)
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake, reference_path_message(points)
    )
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        reference_path_message(
            (), valid=False, reason="reference_planner_not_active"
        ),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    stopped = fake._writer.frames[-1]
    assert np.count_nonzero(
        (stopped[:, :, 0] > 150)
        & (stopped[:, :, 1] > 150)
        & (stopped[:, :, 2] < 100)
    ) == 0


def test_reference_path_obeys_localization_gate_and_does_not_create_orange_guidance():
    fake = make_viewer(clock=lambda: 100.0)
    enable_reference(fake)
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        reference_path_message(
            ((1.0, 0.0, 0.0), (8.0, 0.0, 0.0))
        ),
    )
    fake._localization = LocalizationMonitor(clock=lambda: 100.0)

    viewer_node.ArPathViewerNode._tick(fake)

    frame = fake._writer.frames[-1]
    cyan = (
        (frame[:, :, 0] > 150)
        & (frame[:, :, 1] > 150)
        & (frame[:, :, 2] < 100)
    )
    orange = (
        (frame[:, :, 0] < 80)
        & (frame[:, :, 1] > 80)
        & (frame[:, :, 2] > 180)
    )
    assert np.count_nonzero(cyan) == 0
    assert np.count_nonzero(orange[100:]) == 0


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


def test_offscreen_guidance_records_only_for_fresh_stable_real_path():
    now = [10.0]
    fake = make_viewer(clock=lambda: now[0])
    fake._path_store = PathStore(timeout_sec=6.5, clock=lambda: now[0])
    viewer_node.ArPathViewerNode._on_path(
        fake,
        path_message(((0.1, 0.0, 0.0), (1.8, 0.0, 0.0))),
    )
    viewer_node.ArPathViewerNode._tick(fake)
    guided = fake._writer.frames[-1]
    guided_orange = (
        (guided[120:, :, 2] > 180)
        & (guided[120:, :, 1] > 80)
        & (guided[120:, :, 0] < 80)
    )
    assert np.count_nonzero(guided_orange) > 100

    fake._localization.reject("test localization degradation")
    viewer_node.ArPathViewerNode._tick(fake)
    unstable = fake._writer.frames[-1]
    unstable_orange = (
        (unstable[120:, :, 2] > 180)
        & (unstable[120:, :, 1] > 80)
        & (unstable[120:, :, 0] < 80)
    )
    assert np.count_nonzero(unstable_orange) == 0

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
    now[0] = 16.6
    viewer_node.ArPathViewerNode._tick(fake)
    stale = fake._writer.frames[-1]
    stale_orange = (
        (stale[120:, :, 2] > 180)
        & (stale[120:, :, 1] > 80)
        & (stale[120:, :, 0] < 80)
    )
    assert np.count_nonzero(stale_orange) == 0


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
    fake._display_window = RecordingDisplayWindow(ord("q"))
    display_window = fake._display_window
    fake._release_resources = lambda: viewer_node.ArPathViewerNode._release_resources(fake)
    fake._request_shutdown = lambda reason: viewer_node.ArPathViewerNode._request_shutdown(
        fake, reason
    )
    viewer_node.ArPathViewerNode._on_path(
        fake, path_message(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    )
    shutdown_calls = []
    monkeypatch.setattr(
        viewer_node, "shutdown_ros_context", lambda: shutdown_calls.append(True)
    )

    viewer_node.ArPathViewerNode._tick(fake)
    assert len(writer.frames) == 1
    assert writer.release_count == 1
    assert source.close_count == 1
    assert len(display_window.frames) == 1
    assert display_window.frames[0].shape == (480, 640, 3)
    assert display_window.close_count == 1
    assert shutdown_calls == [True]
    fake._request_shutdown("duplicate")
    fake._release_resources()
    assert writer.release_count == 1
    assert source.close_count == 1
    assert display_window.close_count == 1
    assert shutdown_calls == [True]


def test_log_view_copy_failure_and_control_bar_never_enter_recording(monkeypatch):
    fake = make_viewer()
    fake._display = True
    fake._display_window = RecordingDisplayWindow(
        actions=(viewer_node.ACTION_VIEW_LOG, viewer_node.ACTION_COPY_LOG)
    )
    fake._log_gui = RecordingLogGui(copy_success=False)
    fake._session_log = MemorySessionLog()
    fake._log_copy_max_chars = 100000
    fake._handle_display_action = lambda action: (
        viewer_node.ArPathViewerNode._handle_display_action(fake, action)
    )
    fake._log_info = lambda message: (
        fake._session_log.record("INFO", message), fake.logger.info(message)
    )
    fake._log_warning = lambda message: (
        fake._session_log.record("WARNING", message), fake.logger.warning(message)
    )

    viewer_node.ArPathViewerNode._tick(fake)

    assert len(fake._writer.frames) == 1
    assert fake._writer.frames[0].shape == (480, 640, 3)
    assert len(fake._display_window.frames) == 1
    assert fake._display_window.frames[0].shape == (480, 640, 3)
    assert len(fake._log_gui.shown) == 1
    assert callable(fake._log_gui.shown[0][1])
    assert fake._log_gui.copied == [fake._session_log.text()]
    assert fake._display_window.statuses == [
        ("Log window opened", True),
        ("Log copy failed", False),
    ]
    assert not fake._shutdown_requested
    assert fake._log_gui.poll_count == 1
    assert any(
        level == "WARNING" and "clipboard unavailable" in message
        for level, message in fake._session_log.records
    )


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
