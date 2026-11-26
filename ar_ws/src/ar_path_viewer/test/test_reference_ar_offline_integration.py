"""
Offline integration coverage for the validated reference-path AR pipeline.

The tests deliberately use the generated ROS message classes, a deterministic
time-stamped TF buffer, AMCL messages, and the real viewer tick/rendering path.
No ROS graph, camera device, Nav2 process, or robot command topic is used.
"""

from types import MethodType
import threading

import cv2
import numpy as np
import pytest

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from robotbase_waypoint_interfaces.msg import ReferencePathState, WaypointFollowerState

from ar_path_viewer.diagnostics import DiagnosticReporter
from ar_path_viewer.costmap_alerts import CostmapAlertGate, CostmapAlertSettings
from ar_path_viewer.costmap_explanation import CostmapExplanationSettings
from ar_path_viewer.costmap_overlay import (
    CostmapCameraSyncSettings,
    CostmapOverlaySettings,
    CostmapStore,
    CostmapVisualHold,
)
from ar_path_viewer.guidance_stability import GuidanceStabilityGate
from ar_path_viewer.image_source import SimulationSource
from ar_path_viewer.localization_monitor import LocalizationMonitor, LocalizationThresholds
from ar_path_viewer.offscreen_guidance import GuidanceSettings
from ar_path_viewer.odometry_distance import SessionOdomTravelDistanceMeter
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


class MissingImageTimeTfBuffer(TimedTfBuffer):
    """Allow localization/Costmap TF, then fail the image-time lookup."""

    def __init__(self, clock: MutableClock):
        super().__init__(clock)
        self.explicit_queries = 0

    def lookup_transform(self, target_frame, source_frame, *args, **kwargs):
        query = args[0] if args else None
        if query is not None and getattr(query, "nanoseconds", 0) > 0:
            self.explicit_queries += 1
            if self.explicit_queries == 2:
                raise LookupError("synthetic image-time TF unavailable")
        return super().lookup_transform(target_frame, source_frame, *args, **kwargs)


class ExactQueryTfBuffer(TimedTfBuffer):
    """Return identity transforms stamped at the requested ROS time."""

    def lookup_transform(self, target_frame, source_frame, *args, **kwargs):
        message = super().lookup_transform(
            target_frame, source_frame, *args, **kwargs
        )
        query = args[0] if args else None
        if query is not None and getattr(query, "nanoseconds", 0) > 0:
            _set_stamp(message.header.stamp, query.nanoseconds / 1e9)
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
    fake._travel_session_lock = threading.Lock()
    fake._session_distance_meter = SessionOdomTravelDistanceMeter()
    fake._autonomous_session_identity = None
    fake._route_completion = None
    fake._last_verified_final_goal = None
    fake._final_waypoint_success = None
    fake._update_autonomous_travel_session = MethodType(
        viewer_node.ArPathViewerNode._update_autonomous_travel_session, fake
    )
    fake._accept_verified_route_completion = MethodType(
        viewer_node.ArPathViewerNode._accept_verified_route_completion, fake
    )
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
    fake._image_input_kind = "simulation"
    fake._last_image_frame_timing = None
    fake._costmap_camera_sync_settings = CostmapCameraSyncSettings()
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
    fake._perception_costmap_enabled = False
    fake._perception_explanation_enabled = False
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
    fake._costmap_arrays_at_image_time = MethodType(
        viewer_node.ArPathViewerNode._costmap_arrays_at_image_time, fake
    )
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


def _costmap(stamp: float) -> OccupancyGrid:
    message = OccupancyGrid()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.info.width = 30
    message.info.height = 20
    message.info.resolution = 0.10
    message.info.origin.position.x = 2.5
    message.info.origin.position.y = -1.0
    message.info.origin.orientation.w = 1.0
    data = np.zeros(message.info.width * message.info.height, dtype=np.int8)
    # Three separate Nav2 semantics near the projected route.
    data[8 * message.info.width + 8:8 * message.info.width + 14] = 70
    data[10 * message.info.width + 11:10 * message.info.width + 17] = 100
    data[12 * message.info.width + 14:12 * message.info.width + 20] = -1
    message.data = data.tolist()
    return message


def _near_robot_lethal_costmap(stamp: float) -> OccupancyGrid:
    message = OccupancyGrid()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.info.width = 10
    message.info.height = 10
    message.info.resolution = 0.10
    message.info.origin.position.x = 0.0
    message.info.origin.position.y = -0.5
    message.info.origin.orientation.w = 1.0
    data = np.zeros(100, dtype=np.int8)
    data[5 * 10 + 4] = 100
    message.data = data.tolist()
    return message


def _explanation_costmap(stamp: float) -> OccupancyGrid:
    message = OccupancyGrid()
    message.header.frame_id = "map"
    _set_stamp(message.header.stamp, stamp)
    message.info.width = 60
    message.info.height = 60
    message.info.resolution = 0.05
    message.info.origin.position.x = 0.0
    message.info.origin.position.y = -1.5
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (message.info.width * message.info.height)
    return message


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


def _largest_cyan_component(frame) -> int:
    """Distinguish a retained route from isolated lossy-codec color noise."""
    image = frame[90:]
    mask = (
        (image[:, :, 0] > 150)
        & (image[:, :, 1] > 150)
        & (image[:, :, 2] < 120)
    ).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


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


@pytest.mark.parametrize(
    ('direction', 'lateral_sign'),
    (('LEFT', 1.0), ('STRAIGHT', 0.0), ('RIGHT', -1.0)),
)
def test_live_input_contract_draws_both_paths_and_direction_card(
    monkeypatch, direction, lateral_sign
):
    """Exercise the full Viewer tick, not just the card drawing helper."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    captured = []
    real_render = viewer_node.render_frame

    def recording_render(*args, **kwargs):
        captured.append((
            kwargs.get('guidance'),
            kwargs.get('waypoint_status'),
        ))
        return real_render(*args, **kwargs)

    monkeypatch.setattr(viewer_node, 'render_frame', recording_render)
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    plan = tuple(
        (float(x), lateral_sign * 0.45 * float(x), 0.0)
        for x in np.linspace(0.15, 4.0, 50)
    )
    viewer_node.ArPathViewerNode._on_path(fake, _path(plan, clock.now))

    frame = _tick_frame(fake)

    guidance, distance = captured[-1]
    assert guidance is not None
    assert guidance.direction == direction
    assert distance.startswith('PLAN LEFT:')
    assert _cyan_pixels(frame) > 100
    assert _green_pixels(frame) > 100
    orange = (
        (frame[48:130, :, 2] > 180)
        & (frame[48:130, :, 1] > 80)
        & (frame[48:130, :, 0] < 80)
    )
    assert np.count_nonzero(orange) > 250


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


def test_local_costmap_paths_guidance_and_recording_are_independent():
    """Exercise the opt-in perception layer through the full Viewer tick."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_alpha = 0.55
    fake._costmap_radius = 2
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    viewer_node.ArPathViewerNode._on_path(
        fake,
        _path(
            tuple(
                (float(x), 0.32 * float(x), 0.0)
                for x in np.linspace(0.2, 4.0, 50)
            ),
            clock.now,
        ),
    )
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))

    visible = _tick_frame(fake)
    magenta = (
        (visible[:, :, 0] > 120)
        & (visible[:, :, 1] < 130)
        & (visible[:, :, 2] > 140)
    )
    assert np.count_nonzero(magenta) > 10
    assert _cyan_pixels(visible) > 100
    assert _green_pixels(visible) > 100
    assert any(
        "perception_costmap_projection" in message and "reason=visible" in message
        for _, message in fake.logger.records
    )

    # Fresh paths remain, while stale perception data disappears fail-closed.
    clock.now += 2.0
    stale = _tick_frame(fake)
    stale_magenta = (
        (stale[:, :, 0] > 120)
        & (stale[:, :, 1] < 130)
        & (stale[:, :, 2] > 140)
    )
    assert np.count_nonzero(stale_magenta) == 0
    assert _cyan_pixels(stale) > 100
    assert _green_pixels(stale) > 100

    # A new costmap recovers, but an unavailable transform suppresses only it.
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))
    fake._costmap_arrays_in_base = (
        lambda _snapshot: (_ for _ in ()).throw(
            ValueError("synthetic costmap TF unavailable")
        )
    )
    no_tf = _tick_frame(fake)
    no_tf_magenta = (
        (no_tf[:, :, 0] > 120)
        & (no_tf[:, :, 1] < 130)
        & (no_tf[:, :, 2] > 140)
    )
    assert np.count_nonzero(no_tf_magenta) == 0
    assert _cyan_pixels(no_tf) > 100
    assert _green_pixels(no_tf) > 100


def test_exact_costmap_visual_hold_bridges_only_short_publish_gap():
    """A short stale visual is reprojected at image time, not pixel-reused."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_camera_sync_settings = CostmapCameraSyncSettings(
        max_costmap_age_sec=0.40
    )
    fake._costmap_visual_hold = CostmapVisualHold(0.30, clock=clock)
    fake._last_costmap_visual_context = ""
    fake._costmap_alpha = 0.55
    fake._costmap_radius = 2
    fake._perception_alert_enabled = True
    fake._costmap_alert_settings = CostmapAlertSettings(
        confirm_sec=0.0, release_sec=0.0
    )
    fake._costmap_alert_gate = CostmapAlertGate(0.0, 0.0, clock=clock)
    fake._tf_buffer = ExactQueryTfBuffer(clock)
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    viewer_node.ArPathViewerNode._on_path(fake, _path(PLAN_POINTS, clock.now))
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))

    # Establish one exact-time synchronized projection.
    live = _tick_frame(fake)
    assert np.count_nonzero(
        (live[:, :, 0] > 120) & (live[:, :, 1] < 130)
        & (live[:, :, 2] > 140)
    ) > 10
    projected = fake._costmap_visual_hold._projection
    assert projected is not None and len(projected.lethal_cost_pixels) > 0
    marker_x, marker_y = projected.lethal_cost_pixels[0]

    # The measured median update gap is about 0.62 s. At age 0.60 s the 0.40 s
    # semantic limit has expired, while exact transforms at both Costmap and
    # image time still authorize a dim, explicitly labelled display-only
    # reprojection. It cannot enter alerts or explanations.
    clock.now += 0.60
    # The synthetic ROS clock advances independently of real monotonic time;
    # reset only this test harness' cross-clock sample before the next frame.
    fake._last_image_frame_timing = None
    held = _tick_frame(fake)
    assert np.count_nonzero(
        (held[:, :, 0] > 90) & (held[:, :, 1] < 150)
        & (held[:, :, 2] > 105)
    ) > 10
    assert fake._diagnostics.costmap_hold_display_count == 1
    assert fake._costmap_visual_hold._projection is None
    assert any(
        "perception_costmap_alert visible=false "
        "reason=synchronized_inputs_unavailable" in message
        for _, message in fake.logger.records
    )

    # A new exact-time Costmap immediately replaces the visual-only state.
    viewer_node.ArPathViewerNode._on_local_costmap(
        fake, _costmap(clock.now)
    )
    fake._last_image_frame_timing = None
    refreshed = _tick_frame(fake)
    assert fake._diagnostics.costmap_live_display_count == 2
    assert np.count_nonzero(
        (refreshed[:, :, 0] > 120) & (refreshed[:, :, 1] < 130)
        & (refreshed[:, :, 2] > 140)
    ) > 10

    # The display-only deadline is semantic limit + configured visual bridge,
    # and remains independent of the longer Costmap store timeout.
    clock.now += 0.71
    fake._last_image_frame_timing = None
    expired = _tick_frame(fake)
    x0, x1 = max(0, marker_x - 12), min(expired.shape[1], marker_x + 13)
    y0, y1 = max(0, marker_y - 12), min(expired.shape[0], marker_y + 13)
    assert np.sum(np.abs(
        held[y0:y1, x0:x1].astype(np.int16)
        - expired[y0:y1, x0:x1].astype(np.int16)
    )) > 500
    assert fake._diagnostics.costmap_hidden_display_count >= 1


def test_near_robot_lethal_alert_remains_available_without_real_plan():
    """A missing /plan does not suppress fresh exact-time proximity evidence."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_camera_sync_settings = CostmapCameraSyncSettings(
        max_costmap_age_sec=0.40
    )
    fake._costmap_visual_hold = CostmapVisualHold(0.30, clock=clock)
    fake._costmap_alpha = 0.55
    fake._costmap_radius = 2
    fake._perception_alert_enabled = True
    fake._costmap_alert_settings = CostmapAlertSettings(
        confirm_sec=0.0, release_sec=0.0
    )
    fake._costmap_alert_gate = CostmapAlertGate(0.0, 0.0, clock=clock)
    fake._tf_buffer = ExactQueryTfBuffer(clock)
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    viewer_node.ArPathViewerNode._on_local_costmap(
        fake, _near_robot_lethal_costmap(clock.now)
    )

    frame = _tick_frame(fake)

    assert np.count_nonzero(frame[109:201]) > 100
    assert any(
        "perception_costmap_alert visible=true "
        "reason=lethal_cost_near_robot" in message
        for _, message in fake.logger.records
    )


def test_path_updated_event_does_not_require_costmap_perception():
    """The existing /plan-generation event is not a Costmap claim."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._perception_alert_enabled = True
    fake._path_updated_until = clock.now + 1.0
    _send_good_localization(fake, clock)
    viewer_node.ArPathViewerNode._on_path(
        fake, _path(PLAN_POINTS, clock.now)
    )

    frame = _tick_frame(fake)

    assert np.count_nonzero(frame[109:146]) > 100
    assert any(
        "perception_costmap_alert visible=true "
        "reason=confirmed_real_plan_update" in message
        for _, message in fake.logger.records
    )


def test_untrusted_camera_time_hides_costmap_then_recovers():
    """Camera-time failure hides perception without hiding either path."""
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_alpha = 0.55
    fake._costmap_radius = 2
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    viewer_node.ArPathViewerNode._on_path(
        fake, _path(PLAN_POINTS, clock.now)
    )
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))

    fake._image_input_kind = "video"
    hidden = _tick_frame(fake)
    hidden_magenta = (
        (hidden[:, :, 0] > 120)
        & (hidden[:, :, 1] < 130)
        & (hidden[:, :, 2] > 140)
    )
    assert np.count_nonzero(hidden_magenta) == 0
    assert _cyan_pixels(hidden) > 100
    assert _green_pixels(hidden) > 100
    assert any(
        "camera_time_alignment_unavailable" in message
        and "untrusted_source" in message
        for _, message in fake.logger.records
    )

    fake._image_input_kind = "simulation"
    recovered = _tick_frame(fake)
    recovered_magenta = (
        (recovered[:, :, 0] > 120)
        & (recovered[:, :, 1] < 130)
        & (recovered[:, :, 2] > 140)
    )
    assert np.count_nonzero(recovered_magenta) > 10
    assert _cyan_pixels(recovered) > 100
    assert _green_pixels(recovered) > 100


def test_missing_image_time_tf_hides_costmap_without_hiding_paths():
    clock = MutableClock()
    fake = _make_viewer(clock)
    fake._tf_buffer = MissingImageTimeTfBuffer(clock)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_alpha = 0.55
    fake._costmap_radius = 2
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    viewer_node.ArPathViewerNode._on_path(fake, _path(PLAN_POINTS, clock.now))
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))

    frame = _tick_frame(fake)
    magenta = (
        (frame[:, :, 0] > 120)
        & (frame[:, :, 1] < 130)
        & (frame[:, :, 2] > 140)
    )
    assert np.count_nonzero(magenta) == 0
    assert _cyan_pixels(frame) > 100
    assert _green_pixels(frame) > 100
    assert any(
        "camera_time_alignment_unavailable" in message
        and "image-time TF unavailable" in message
        for _, message in fake.logger.records
    )


def test_recent_minimap_snapshot_is_explicitly_stale_and_expires():
    class Dummy:
        pass

    fake = Dummy()
    token = object()
    call = viewer_node.ArPathViewerNode._display_minimap_snapshot
    current, label = call(fake, token, now=10.0, allow_recent=True)
    assert current is token and label == ""
    recent, label = call(fake, None, now=10.65, allow_recent=True)
    assert recent is token and label.startswith("LAST SNAPSHOT 0.7s")
    recent, label = call(fake, None, now=11.01, allow_recent=True)
    assert recent is None and label == ""
    call(fake, token, now=20.0, allow_recent=True)
    recent, label = call(fake, None, now=20.1, allow_recent=False)
    assert recent is None and label == ""


def test_fresh_costmap_and_authorized_real_path_draw_explanation_card():
    clock = MutableClock()
    fake = _make_viewer(clock)
    # The explanation card is available in diagnostic display mode.
    fake._debug_overlay = True
    fake._perception_costmap_enabled = True
    fake._perception_explanation_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_explanation_settings = CostmapExplanationSettings()
    fake._costmap_alpha = 0.42
    fake._costmap_radius = 2
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    _send_authorized_reference(fake, clock, REFERENCE_POINTS)
    straight = tuple((float(x), 0.0, 0.0) for x in np.linspace(0.2, 4.0, 50))
    viewer_node.ArPathViewerNode._on_path(fake, _path(straight, clock.now))
    viewer_node.ArPathViewerNode._on_local_costmap(
        fake, _explanation_costmap(clock.now)
    )

    frame = _tick_frame(fake)

    assert np.count_nonzero(frame[50:104, 10:380]) > 500
    assert _green_pixels(frame) > 100
    assert _cyan_pixels(frame) > 100
    assert any(
        "perception_path_explanation" in message
        and "low_cost_front_with_straight_path" in message
        for _, message in fake.logger.records
    )

    clock.now += 2.0
    stale = _tick_frame(fake)
    assert not np.array_equal(frame[50:104, 10:380], stale[50:104, 10:380])
    assert any(
        "perception_path_explanation" in message
        and "visible=false" in message
        for _, message in fake.logger.records
    )


def test_recorded_mp4_contains_only_fresh_local_costmap_markers(tmp_path):
    output = tmp_path / "perception_costmap_offline_integration.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480)
    )
    assert writer.isOpened()
    clock = MutableClock()
    fake = _make_viewer(clock, writer=writer)
    fake._perception_costmap_enabled = True
    fake._costmap_settings = CostmapOverlaySettings(
        timeout_sec=1.5,
        stamp_max_age_sec=1.5,
        max_points_per_class=30,
    )
    fake._costmap_store = CostmapStore(fake._costmap_settings, clock=clock)
    fake._costmap_alpha = 0.60
    fake._costmap_radius = 3
    fake._costmap_arrays_in_base = lambda snapshot: (
        snapshot.points.copy(), snapshot.analysis_points.copy()
    )
    _send_good_localization(fake, clock)
    viewer_node.ArPathViewerNode._on_path(fake, _path(PLAN_POINTS, clock.now))
    viewer_node.ArPathViewerNode._on_local_costmap(fake, _costmap(clock.now))

    for _ in range(4):
        viewer_node.ArPathViewerNode._tick(fake)
    clock.now += 2.0
    for _ in range(4):
        viewer_node.ArPathViewerNode._tick(fake)
    writer.release()

    capture = cv2.VideoCapture(str(output))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()

    def largest_marker_component(frame):
        mask = (
            (frame[:, :, 0] > 105)
            & (frame[:, :, 1] < 140)
            & (frame[:, :, 2] > 125)
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        if count <= 1:
            return 0
        return int(stats[1:, cv2.CC_STAT_AREA].max())

    assert output.stat().st_size > 0
    assert len(decoded) == 8
    counts = [largest_marker_component(frame) for frame in decoded]
    assert min(counts[:4]) > 100
    assert max(counts[4:]) <= 10
    assert min(_green_pixels(frame) for frame in decoded) > 50


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
    assert min(_largest_cyan_component(frame) for frame in decoded[:4]) > 50
    # mp4v is lossy and creates a few isolated color-classification pixels at
    # sharp green/dark-halo/background edges.  Reject any connected route-sized
    # cyan residue while allowing a small, bounded codec artifact.
    assert max(cyan_counts[4:8]) <= 20
    assert max(_largest_cyan_component(frame) for frame in decoded[4:8]) <= 8
    assert min(cyan_counts[8:]) > 50
    assert min(_largest_cyan_component(frame) for frame in decoded[8:]) > 50
    # The real /plan remains recorded throughout reference invalidation.
    assert min(green_counts) > 50
