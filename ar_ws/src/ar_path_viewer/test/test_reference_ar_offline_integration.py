"""
Offline integration coverage for the validated reference-path AR pipeline.

The tests deliberately use the generated ROS message classes, a deterministic
time-stamped TF buffer, AMCL messages, and the real viewer tick/rendering path.
No ROS graph, camera device, Nav2 process, or robot command topic is used.
"""

from types import MethodType

import cv2
import numpy as np

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Path
from robotbase_waypoint_interfaces.msg import ReferencePathState, WaypointFollowerState

from ar_path_viewer.diagnostics import DiagnosticReporter
from ar_path_viewer.guidance_stability import GuidanceStabilityGate
from ar_path_viewer.image_source import SimulationSource
from ar_path_viewer.localization_monitor import LocalizationMonitor, LocalizationThresholds
from ar_path_viewer.offscreen_guidance import GuidanceSettings
from ar_path_viewer.path_adapter import PathStore
from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.reference_path import ReferencePathGate
from ar_path_viewer.viewer_logic import PathDisplayGeometry
from ar_path_viewer.waypoint_progress import WaypointProgressGate
from ar_path_viewer import viewer_node


ROUTE_A = "a" * 64
ROUTE_B = "b" * 64


class MutableClock:
    def __init__(self, now: float = 100.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


class RecordingLogger:
    def __init__(self):
        self.records = []

    def _record(self, level, message):
        self.records.append((level, str(message)))

    def debug(self, message):
        self._record("debug", message)

    def info(self, message):
        self._record("info", message)

    def warning(self, message):
        self._record("warning", message)

    def error(self, message):
        self._record("error", message)

    def fatal(self, message):
        self._record("fatal", message)


class MemoryWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class TimedTfBuffer:
    """Return identity transforms with controllable availability and age."""

    def __init__(self, clock: MutableClock):
        self.clock = clock
        self.available = True
        self.stamp_offset_sec = 0.0
        self.zero_stamp = False

    def lookup_transform(self, target_frame, source_frame, *_args, **_kwargs):
        if not self.available:
            raise LookupError(f"no transform {source_frame!r} -> {target_frame!r}")
        message = TransformStamped()
        message.header.frame_id = str(target_frame)
        message.child_frame_id = str(source_frame)
        stamp = 0.0 if self.zero_stamp else self.clock.now + self.stamp_offset_sec
        _set_stamp(message.header.stamp, stamp)
        message.transform.rotation.w = 1.0
        return message


class OfflineViewer:
    pass


def _set_stamp(stamp, seconds: float) -> None:
    whole = int(seconds)
    stamp.sec = whole
    stamp.nanosec = int(round((seconds - whole) * 1_000_000_000))


def _path(points, stamp: float, frame: str = "map") -> Path:
    message = Path()
    message.header.frame_id = frame
    _set_stamp(message.header.stamp, stamp)
    for x, y, z in points:
        pose = PoseStamped()
        pose.header.frame_id = frame
        _set_stamp(pose.header.stamp, stamp)
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def _follower(
    stamp: float,
    *,
    route: str = ROUTE_A,
    index: int = 1,
    generation: int = 4,
    activity: int = WaypointFollowerState.ACTIVITY_NAVIGATING,
    allowed: bool = True,
) -> WaypointFollowerState:
    message = WaypointFollowerState()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.schema_version = WaypointFollowerState.SCHEMA_VERSION
    message.route_sha256 = route
    message.next_waypoint_index = index
    message.next_waypoint_number = 20 + index
    message.navigation_mode = "to_pose"
    message.goal_generation = generation
    message.activity = activity
    message.reference_allowed = allowed
    message.safe_section_end_index = 7
    message.waypoint_count = 8
    return message


def _reference(
    points,
    stamp: float,
    *,
    route: str = ROUTE_A,
    index: int = 1,
    generation: int = 4,
    valid: bool = True,
    reason: str = "",
) -> ReferencePathState:
    message = ReferencePathState()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.schema_version = ReferencePathState.SCHEMA_VERSION
    message.valid = valid
    message.invalid_reason = reason
    message.route_sha256 = route
    message.next_waypoint_index = index
    message.next_waypoint_number = 20 + index
    message.navigation_mode = "to_pose"
    message.goal_generation = generation
    message.safe_section_end_index = 7
    message.waypoint_count = 8
    message.planner_generation = 5
    _set_stamp(message.follower_state_stamp, stamp)
    message.path = _path(points, stamp)
    if len(points) >= 2:
        array = np.asarray(points, dtype=float)
        message.path_length_m = float(
            np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1).sum()
        )
    return message


def _amcl(stamp: float, *, position_variance: float = 0.01):
    message = PoseWithCovarianceStamped()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance[0] = position_variance
    message.pose.covariance[7] = position_variance
    message.pose.covariance[35] = 0.01
    return message


def _make_viewer(clock: MutableClock, writer=None) -> OfflineViewer:
    fake = OfflineViewer()
    fake._shutdown_requested = False
    fake._intrinsics = CameraIntrinsics()
    fake._source_intrinsics = fake._intrinsics
    fake._undistorter = None
    fake._mount = CameraMount()
    fake._near = 0.1
    fake._path_display_geometry = PathDisplayGeometry("elevated", 0.50)
    fake._calibration_profile_label = ""
    fake._guidance_settings = GuidanceSettings()
    fake._guidance_color = (0, 165, 255)
    fake._guidance_stability = GuidanceStabilityGate(0.0)
    fake._guidance_require_follower = True
    fake._debug_overlay = False
    fake._path_store = PathStore(timeout_sec=6.5, clock=clock)
    fake._image_source = SimulationSource(640, 480)
    fake._camera_error = ""
    fake._config_error = ""
    fake._base_frame = "robot/base_footprint"
    fake._color = (0, 255, 0)
    fake._thickness = 5
    fake._display = False
    fake._display_window = None
    fake._writer = writer if writer is not None else MemoryWriter()
    fake._log_gui = None
    fake._reference_enabled = True
    fake._waypoint_distance_enabled = True
    fake._waypoint_progress_gate = WaypointProgressGate(
        timeout_sec=2.5, expected_frame="map"
    )
    fake._reference_gate = ReferencePathGate(
        timeout_sec=6.5,
        follower_timeout_sec=2.5,
        expected_frame="map",
        clock=clock,
    )
    fake._reference_color = (255, 255, 0)
    fake._reference_thickness = 3
    fake._reference_dash_px = 14.0
    fake._reference_gap_px = 9.0
    fake._tf_buffer = TimedTfBuffer(clock)
    fake._ros_now_sec = lambda: clock.now
    fake._points_in_base = MethodType(viewer_node.ArPathViewerNode._points_in_base, fake)
    fake._check_localization_tf = MethodType(
        viewer_node.ArPathViewerNode._check_localization_tf, fake
    )
    fake.logger = RecordingLogger()
    fake.get_logger = lambda: fake.logger
    fake._log_info = fake.logger.info
    fake._log_warning = fake.logger.warning
    fake._diagnostics = DiagnosticReporter(
        True, fake.logger, min_interval_sec=0.01, clock=clock
    )
    fake._localization = LocalizationMonitor(
        LocalizationThresholds(recovery_samples=1, recovery_sec=0.0),
        clock=clock,
    )
    return fake


def _send_good_localization(fake, clock: MutableClock) -> None:
    viewer_node.ArPathViewerNode._on_amcl_pose(fake, _amcl(clock.now))


def _send_authorized_reference(
    fake,
    clock: MutableClock,
    points,
    *,
    route: str = ROUTE_A,
    index: int = 1,
    generation: int = 4,
) -> None:
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        _follower(
            clock.now, route=route, index=index, generation=generation
        ),
    )
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        _reference(
            points,
            clock.now,
            route=route,
            index=index,
            generation=generation,
        ),
    )


def _cyan_pixels(frame) -> int:
    # Ignore the status/REFERENCE label and count only route pixels.
    image = frame[90:]
    return int(
        np.count_nonzero(
            (image[:, :, 0] > 150)
            & (image[:, :, 1] > 150)
            & (image[:, :, 2] < 120)
        )
    )


def _green_pixels(frame) -> int:
    image = frame[90:]
    return int(
        np.count_nonzero(
            (image[:, :, 0] < 120)
            & (image[:, :, 1] > 150)
            & (image[:, :, 2] < 120)
        )
    )


def _tick_frame(fake):
    viewer_node.ArPathViewerNode._tick(fake)
    return fake._writer.frames[-1]


REFERENCE_POINTS = tuple(
    (float(x), 0.55 + 0.10 * np.sin(float(x) / 2.0), 0.0)
    for x in np.linspace(0.7, 12.0, 80)
)
PLAN_POINTS = tuple(
    (float(x), -0.45 + 0.03 * float(x), 0.0)
    for x in np.linspace(0.7, 6.0, 45)
)


def test_real_messages_fresh_amcl_and_timed_tf_draw_independent_paths():
    clock = MutableClock()
    fake = _make_viewer(clock)
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    viewer_node.ArPathViewerNode._on_path(
        fake, _path(PLAN_POINTS, clock.now)
    )

    frame = _tick_frame(fake)

    assert fake._diagnostics.reference_receive_count == 1
    assert fake._reference_gate.state(clock.now)[0] == "ACTIVE"
    assert _cyan_pixels(frame) > 100
    assert _green_pixels(frame) > 100
    # The configured gaps remain visible; the reference is not rendered solid.
    cyan_mask = (
        (frame[90:, :, 0] > 150)
        & (frame[90:, :, 1] > 150)
        & (frame[90:, :, 2] < 120)
    ).astype(np.uint8)
    components, _, stats, _ = cv2.connectedComponentsWithStats(cyan_mask, 8)
    substantial_dashes = stats[1:, cv2.CC_STAT_AREA] >= 4
    assert components > 2
    assert np.count_nonzero(substantial_dashes) >= 3


def test_all_safety_stops_hide_reference_and_fresh_state_recovers():
    clock = MutableClock()
    fake = _make_viewer(clock)
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    assert _cyan_pixels(_tick_frame(fake)) > 100

    def assert_hidden():
        assert _cyan_pixels(_tick_frame(fake)) == 0

    def restore(route=ROUTE_A, index=1, generation=4):
        _send_good_localization(fake, clock)
        _send_authorized_reference(
            fake,
            clock,
            REFERENCE_POINTS,
            route=route,
            index=index,
            generation=generation,
        )
        assert _cyan_pixels(_tick_frame(fake)) > 100

    # Pause/person-search style authorization removal.
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake,
        _follower(
            clock.now,
            activity=WaypointFollowerState.ACTIVITY_MANUAL_PAUSED,
            allowed=False,
        ),
    )
    assert_hidden()
    restore()

    # Route/generation/index identity changes invalidate an old atomic result.
    viewer_node.ArPathViewerNode._on_reference_follower_state(
        fake, _follower(clock.now, route=ROUTE_B, index=2, generation=5)
    )
    assert_hidden()
    restore(route=ROUTE_B, index=2, generation=5)

    # A fail-closed planner result represents failed Keepout verification.
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        _reference(
            (),
            clock.now,
            route=ROUTE_B,
            index=2,
            generation=5,
            valid=False,
            reason="keepout_validation_failed map_mask_geometry_mismatch",
        ),
    )
    assert_hidden()
    restore(route=ROUTE_B, index=2, generation=5)

    # Invalid path data clears a formerly valid route rather than retaining it.
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        _reference(
            ((2.0, 0.5, 0.0),),
            clock.now,
            route=ROUTE_B,
            index=2,
            generation=5,
        ),
    )
    assert_hidden()
    restore(route=ROUTE_B, index=2, generation=5)

    # Follower/reference expiry is evaluated against both ROS and monotonic time.
    clock.now += 7.0
    assert_hidden()
    restore(route=ROUTE_B, index=2, generation=5)

    # Bad AMCL covariance suppresses drawing; a new good observation recovers.
    clock.now += 0.1
    viewer_node.ArPathViewerNode._on_amcl_pose(
        fake, _amcl(clock.now, position_variance=4.0)
    )
    assert_hidden()
    restore(route=ROUTE_B, index=2, generation=5)

    # Reproduce the field-test time-zero static TF failure, then restore freshness.
    fake._tf_buffer.zero_stamp = True
    assert_hidden()
    fake._tf_buffer.zero_stamp = False
    restore(route=ROUTE_B, index=2, generation=5)

    # A stale or absent TF independently suppresses reference projection.
    fake._tf_buffer.stamp_offset_sec = -10.0
    assert_hidden()
    fake._tf_buffer.stamp_offset_sec = 0.0
    restore(route=ROUTE_B, index=2, generation=5)
    fake._tf_buffer.available = False
    assert_hidden()
    fake._tf_buffer.available = True
    restore(route=ROUTE_B, index=2, generation=5)

    # A valid but physically out-of-view path is not forced into the image.
    _send_authorized_reference(
        fake,
        clock,
        tuple((-float(x), 0.0, 0.0) for x in np.linspace(1.0, 8.0, 20)),
        route=ROUTE_B,
        index=2,
        generation=5,
    )
    assert_hidden()
    assert any(
        "reason=out_of_view" in message
        for _, message in fake.logger.records
    )
    restore(route=ROUTE_B, index=2, generation=5)


def test_recorded_mp4_contains_cyan_dashes_and_removes_them_when_invalid(tmp_path):
    output = tmp_path / "reference_ar_offline_integration.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480)
    )
    assert writer.isOpened()
    clock = MutableClock()
    fake = _make_viewer(clock, writer=writer)
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    viewer_node.ArPathViewerNode._on_path(fake, _path(PLAN_POINTS, clock.now))

    for _ in range(4):
        viewer_node.ArPathViewerNode._tick(fake)
    viewer_node.ArPathViewerNode._on_reference_path_state(
        fake,
        _reference(
            (),
            clock.now,
            valid=False,
            reason="keepout_validation_failed mask_missing",
        ),
    )
    for _ in range(4):
        viewer_node.ArPathViewerNode._tick(fake)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    for _ in range(4):
        viewer_node.ArPathViewerNode._tick(fake)
    writer.release()

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()

    assert output.stat().st_size > 0
    assert len(decoded) == 12
    cyan_counts = [_cyan_pixels(frame) for frame in decoded]
    green_counts = [_green_pixels(frame) for frame in decoded]
    assert min(cyan_counts[:4]) > 50
    # mp4v is lossy and can create an isolated color-classification pixel at
    # a sharp green/background edge.  Reject any route-sized cyan residue while
    # allowing that bounded codec artifact.
    assert max(cyan_counts[4:8]) <= 2
    assert min(cyan_counts[8:]) > 50
    # The real /plan remains recorded throughout reference invalidation.
    assert min(green_counts) > 50
