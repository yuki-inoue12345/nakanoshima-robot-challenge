"""Follow YAML routes continuously, stopping at task boundaries."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import signal
from time import monotonic
from typing import Optional
from uuid import uuid4

from action_msgs.msg import GoalInfo, GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from .waypoint_io import load_waypoints
from .waypoint_visualize import RouteMarkerPublisher


TERMINAL_STATUSES = {
    GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED,
}
SHUTDOWN_TIMEOUT_SEC = 1.5
SHUTDOWN_ZERO_SETTLE_SEC = 0.5
SHUTDOWN_SPIN_SEC = 0.05


@dataclass
class ShutdownRequest:
    """
    Order shutdown and small commits on the single executor/main thread.

    A successful commit check precedes its one operation. A signal arriving
    inside that operation is recorded immediately, but latched on return; it
    cannot retract an already committed operation. Subsequent commits reject
    even a recorded, not-yet-latched request. This is not a multithreaded lock:
    never spin an executor or group multiple mission operations in a commit.
    """

    _requests: list = field(default_factory=list)
    _latched: Optional[tuple] = None
    _committing: bool = False

    @property
    def started_at(self):
        return self._latched[0] if self._latched else None

    @property
    def reason(self):
        return self._latched[1] if self._latched else ''

    @property
    def signum(self):
        return self._latched[2] if self._latched else None

    @property
    def requested(self):
        return bool(self._requests)

    def request(self, reason: str, signum: Optional[int] = None) -> None:
        if not self._requests:
            # append is the request ordering point, including nested signals.
            # Keep the first timestamp even when latch waits for a commit.
            self._requests.append((monotonic(), reason, signum))
        if not self._committing:
            self._latched = self._requests[0]

    def commit(self, operation, *args, **kwargs) -> bool:
        if self._committing:
            raise RuntimeError('mission commits must not nest')
        self._committing = True
        try:
            if self.requested:
                return False
            # Linearization point: this single operation wins over later
            # requests. Its transport may run for an unbounded duration.
            operation(*args, **kwargs)
            return True
        finally:
            self._committing = False
            if self._requests:
                self._latched = self._requests[0]


@dataclass
class GoalRecord:
    """Keep remote goal ownership separate from the current mission generation."""

    goal_id: object = None
    send_future: object = None
    accepted: Optional[bool] = None
    handle: object = None
    result_future: object = None
    terminal_status: Optional[int] = None
    cancel_attempted: bool = False
    cancel_future: object = None
    cancel_confirmed: bool = False
    error: str = ''
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    allow_early_extension: bool = False
    verify_endpoint_on_success: bool = False

    @property
    def resolved(self) -> bool:
        return self.accepted is False or self.terminal_status in TERMINAL_STATUSES


def yaw_to_quaternion(yaw: float):
    """Return z and w for a planar yaw quaternion."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def goal_uuids_equal(left, right) -> bool:
    """Compare ROS goal UUID sequences without array truth-value coercion."""
    try:
        return bytes(int(value) for value in left) == bytes(
            int(value) for value in right)
    except (TypeError, ValueError):
        return False


def person_area_arrival_qos() -> QoSProfile:
    """Return the durable arrival edge contract consumed by Camera V6."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def person_search_response_qos() -> QoSProfile:
    """Return the volatile response contract published by Camera V6."""
    return QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def waypoint_uses_person_search(waypoint, navigation_only: bool = False) -> bool:
    """Return whether a YAML person area is active for this node run."""
    return bool(waypoint.person_area and not navigation_only)


def route_uses_person_search(waypoints, navigation_only: bool = False) -> bool:
    """Return whether this run needs the Camera person-search ROS contract."""
    return any(
        waypoint_uses_person_search(waypoint, navigation_only)
        for waypoint in waypoints
    )


def validate_person_search_route(
        waypoints, start_index: int, loop: bool,
        navigation_only: bool = False) -> None:
    """Reject route configurations that cannot preserve Camera's area order."""
    if navigation_only:
        return
    person_indices = [
        index for index, waypoint in enumerate(waypoints) if waypoint.person_area
    ]
    if len(person_indices) > 2:
        raise ValueError('person_area may appear at most twice (target_A then target_B)')
    for index in person_indices:
        waypoint = waypoints[index]
        if waypoint.stop or waypoint.wait_time > 0.0:
            raise ValueError(
                f'waypoint {waypoint.number}: person_area cannot be combined with '
                'stop or wait_time')
    if not person_indices:
        return
    if loop:
        raise ValueError('loop=true is not supported for routes containing person_area')
    if start_index - 1 > person_indices[0]:
        raise ValueError(
            'start_index must not skip the first person_area; Camera maps the first '
            'arrival to target_A')


def waypoint_arrival_threshold(
        waypoint, default_threshold: float, precise_threshold: float,
        navigation_only: bool = False) -> float:
    """Select the existing precise threshold for any waypoint that must pause."""
    if (
            waypoint.stop
            or waypoint.wait_time > 0.0
            or waypoint_uses_person_search(waypoint, navigation_only)
    ):
        return precise_threshold
    if waypoint.threshold > 0.0:
        return waypoint.threshold
    return default_threshold


def navigation_segment_end(waypoints, start_index: int, navigation_only: bool = False) -> int:
    """Include the first task/stop waypoint as this continuous section's endpoint."""
    for index in range(start_index, len(waypoints)):
        waypoint = waypoints[index]
        if (waypoint.stop or waypoint.wait_time > 0.0
                or waypoint_uses_person_search(waypoint, navigation_only)):
            return index
    return len(waypoints) - 1


def limit_section_length(waypoints, start: int, end: int, position, limit: float) -> int:
    """Bound polyline distance from the robot without dropping waypoint tasks."""
    distance = 0.0
    previous = position
    for index in range(start, end + 1):
        point = (waypoints[index].x, waypoints[index].y)
        distance += math.hypot(point[0] - previous[0], point[1] - previous[1])
        if distance > limit:
            # An indivisible first leg still needs a goal, even if it is long.
            return max(start, index - 1)
        previous = point
    return end


def split_nearby_endpoint(waypoints, start: int, end: int, position) -> int:
    """Avoid immediate goal success for closed routes or at the start of a lap."""
    def distance(index):
        return math.hypot(waypoints[index].x - position[0], waypoints[index].y - position[1])

    if end > start and distance(end) < 0.5:
        farthest = max(range(start, end), key=distance)
        if distance(farthest) >= 0.5:
            return farthest
    return end


def split_ambiguous_section(waypoints, start: int, end: int) -> int:
    """Keep hairpins and an early revisit of the section goal out of one path."""
    # MPPI's local lookahead can contain both sides of a hairpin. End the
    # section at that turn, then continue only after its action has completed.
    for index in range(max(1, start), end):
        a, b, c = waypoints[index - 1:index + 2]
        incoming = (b.x - a.x, b.y - a.y)
        outgoing = (c.x - b.x, c.y - b.y)
        lengths = math.hypot(*incoming) * math.hypot(*outgoing)
        if lengths > 1e-9 and sum(x * y for x, y in zip(incoming, outgoing)) <= 0.0:
            return index  # A change of direction of at least 90 degrees.

    goal = waypoints[end]
    remaining = 0.0
    for index in range(end - 1, start - 1, -1):
        a, b = waypoints[index:index + 2]
        # Exclude the final approach: naturally every path ends at its goal.
        if remaining > 3.0:
            dx, dy = b.x - a.x, b.y - a.y
            length2 = dx * dx + dy * dy
            projection = (((goal.x - a.x) * dx + (goal.y - a.y) * dy) / length2
                          if length2 > 1e-12 else 0.0)
            t = min(1.0, max(0.0, projection))
            if math.hypot(goal.x - a.x - t * dx, goal.y - a.y - t * dy) < 1.0:
                # This is a batching guard, not a passing radius: no point is
                # discarded. Send both resulting sections in their YAML order.
                farthest = max(range(start, end), key=lambda i: math.hypot(
                    waypoints[i].x - goal.x, waypoints[i].y - goal.y))
                if math.hypot(waypoints[farthest].x - goal.x,
                              waypoints[farthest].y - goal.y) > 1.0:
                    return farthest
        remaining += math.hypot(b.x - a.x, b.y - a.y)
    return end


class WaypointFollower(Node):
    """Send continuous route sections and preserve task-specific stop handshakes."""

    def __init__(self, *, shutdown_request=None) -> None:
        super().__init__('koko_waypoint_follower')
        self._init_shutdown_state(shutdown_request)
        default_prefix = os.environ.get('ROBOTBASE_TF_PREFIX', 'robot')
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('start_index', 1)
        self.declare_parameter('loop', False)
        self.declare_parameter('default_threshold', 1.0)
        self.declare_parameter('precise_threshold', 0.35)
        self.declare_parameter('stop_duration', 5.0)
        self.declare_parameter('stop_on_failure', False)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', f'{default_prefix}/base_footprint')
        self.declare_parameter('safety_cmd_vel_topic', '/cmd_vel_direct')
        self.declare_parameter('person_search_timeout_sec', 60.0)
        self.declare_parameter('person_search_ack_timeout_sec', 5.0)
        self.declare_parameter('person_area_cancel_timeout_sec', 3.0)
        self.declare_parameter('person_area_zero_settle_sec', 0.5)
        self.declare_parameter('navigation_only', False)
        self.declare_parameter('navigation_mode', 'to_pose')
        self.declare_parameter('max_section_length', 40.0)
        self.declare_parameter('section_switch_distance', 6.0)

        waypoint_file = str(self.get_parameter('waypoint_file').value)
        if not waypoint_file:
            raise ValueError('waypoint_file parameter is required')
        self.waypoints = load_waypoints(waypoint_file)
        requested_index = int(self.get_parameter('start_index').value)
        if requested_index < 1 or requested_index > len(self.waypoints):
            raise ValueError(
                f'start_index must be between 1 and {len(self.waypoints)}; '
                f'got {requested_index}')

        self.index = requested_index - 1
        self.navigation_mode = str(self.get_parameter('navigation_mode').value)
        if self.navigation_mode not in ('through_poses', 'to_pose'):
            raise ValueError('navigation_mode must be through_poses or to_pose')
        self.single_goal_recovery = False
        self.manual_paused = False
        self.manual_pause_started_at = None
        self.loop = bool(self.get_parameter('loop').value)
        self.navigation_only = bool(self.get_parameter('navigation_only').value)
        self.max_section_length = float(self.get_parameter('max_section_length').value)
        if not math.isfinite(self.max_section_length) or self.max_section_length <= 0.0:
            raise ValueError('max_section_length must be finite and positive')
        self.section_switch_distance = float(self.get_parameter('section_switch_distance').value)
        if not math.isfinite(self.section_switch_distance) or self.section_switch_distance < 0.0:
            raise ValueError('section_switch_distance must be finite and non-negative')
        validate_person_search_route(
            self.waypoints, requested_index, self.loop, self.navigation_only)
        self.default_threshold = float(self.get_parameter('default_threshold').value)
        self.precise_threshold = float(self.get_parameter('precise_threshold').value)
        self.stop_duration = float(self.get_parameter('stop_duration').value)
        self.stop_on_failure = bool(self.get_parameter('stop_on_failure').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        safety_cmd_vel_topic = str(
            self.get_parameter('safety_cmd_vel_topic').value)
        self.person_search_timeout_sec = float(
            self.get_parameter('person_search_timeout_sec').value)
        self.person_search_ack_timeout_sec = float(
            self.get_parameter('person_search_ack_timeout_sec').value)
        self.person_area_cancel_timeout_sec = float(
            self.get_parameter('person_area_cancel_timeout_sec').value)
        self.person_area_zero_settle_sec = float(
            self.get_parameter('person_area_zero_settle_sec').value)
        for name in (
                'default_threshold', 'precise_threshold', 'stop_duration',
                'person_search_timeout_sec', 'person_search_ack_timeout_sec',
                'person_area_cancel_timeout_sec', 'person_area_zero_settle_sec'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.stop_duration < 0.0:
            raise ValueError('stop_duration must be non-negative')
        if self.default_threshold <= 0.0 or self.precise_threshold <= 0.0:
            raise ValueError('threshold parameters must be positive')
        if self.person_search_timeout_sec <= 0.0:
            raise ValueError('person_search_timeout_sec must be positive')
        if self.person_search_ack_timeout_sec <= 0.0:
            raise ValueError('person_search_ack_timeout_sec must be positive')
        if self.person_area_cancel_timeout_sec <= 0.0:
            raise ValueError('person_area_cancel_timeout_sec must be positive')
        if self.person_area_zero_settle_sec <= 0.0:
            raise ValueError('person_area_zero_settle_sec must be positive')

        through = self.navigation_mode == 'through_poses'
        action_type = NavigateThroughPoses if through else NavigateToPose
        self.action_name = 'navigate_through_poses' if through else 'navigate_to_pose'
        self.action_client = ActionClient(self, action_type, self.action_name)
        self.pause_service = self.create_service(SetBool, '~/set_paused', self._set_paused)
        self.stop_publisher = self.create_publisher(Bool, '/stop', 10)
        self.safety_cmd_vel_publisher = self.create_publisher(
            Twist, safety_cmd_vel_topic, 10)
        self.person_area_arrived_publisher = None
        self.person_search_done_subscription = None
        self.person_search_detail_subscription = None
        self.person_search_ack_subscription = None
        if route_uses_person_search(self.waypoints, self.navigation_only):
            self.person_area_arrived_publisher = self.create_publisher(
                Bool, '/person_area_arrived', person_area_arrival_qos())
            response_qos = person_search_response_qos()
            self.person_search_done_subscription = self.create_subscription(
                Bool, '/person_search_done', self._person_search_done_callback,
                response_qos)
            self.person_search_detail_subscription = self.create_subscription(
                String, '/person_search_done_detail',
                self._person_search_detail_callback, response_qos)
            self.person_search_ack_subscription = self.create_subscription(
                Bool, '/person_search/arrival_received',
                self._person_search_ack_callback, response_qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        self.goal_result_future = None
        self.goal_serial = 0
        self.transition_in_progress = False
        self.cancel_transition_generation = 0
        self.active_cancel_transition_generation = None
        self.person_area_cancel_started_at = None
        self.person_area_stop_pending = False
        self.person_area_stop_started_at = None
        self.person_area_stop_starts_search = False
        self.holding_stop = False
        self.waiting_until = None
        self.waiting_finishes_route = False
        self.person_search_waiting = False
        self.person_search_faulted = False
        self.person_search_fault_reason = ''
        self.person_search_goal_index = None
        self.person_search_started_at = None
        self.person_search_session_id = 0
        self.person_search_arrival_acknowledged = False
        self.person_search_ack_received_at = None
        self.person_search_last_detail = ''
        self.finished = False
        self.timer = self.create_timer(0.1, self._timer_callback)
        self.route_markers = RouteMarkerPublisher(self, self.waypoints, self.map_frame)

        if not self._is_shutting_down():
            self._publish_stop(False)
        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints from '{waypoint_file}', "
            f'start={requested_index}, loop={self.loop}, frame={self.base_frame}, '
            f'stop_on_failure={self.stop_on_failure}, '
            f'navigation_only={self.navigation_only}, navigation_mode={self.navigation_mode}')
        if self.navigation_only:
            self.get_logger().warning(
                'Navigation-only mode: person_area markers are ignored and Camera '
                'topics are not used')

    def start(self) -> None:
        """Wait for Nav2 and dispatch the first waypoint."""
        while rclpy.ok() and not self._is_shutting_down():
            if self.action_client.wait_for_server(timeout_sec=SHUTDOWN_SPIN_SEC):
                self._send_current_goal()
                return
            self.get_logger().info(
                f'Waiting for /{self.action_name} action server...', throttle_duration_sec=1.0)

    def _init_shutdown_state(self, shutdown_request=None) -> None:
        self.shutdown_request = shutdown_request or ShutdownRequest()
        self.shutdown_state = 'RUNNING'
        self.goal_records = {}
        self.cleanup_confirmed = None
        self.cleanup_io_failed = False
        self._cleanup_release_allowed = False

    def _is_shutting_down(self) -> bool:
        return self.shutdown_state != 'RUNNING' or self.shutdown_request.requested

    def _commit_mission(self, operation, *args, **kwargs) -> bool:
        if self.shutdown_state != 'RUNNING':
            return False
        return self.shutdown_request.commit(operation, *args, **kwargs)

    def _commit_state(self, **changes) -> bool:
        """Prepare values first, then atomically commit one mission transition."""
        return self._commit_mission(self.__dict__.update, changes)

    def request_shutdown(self, reason: str) -> None:
        self.shutdown_request.request(reason)

    def _record_for_handle(self, handle) -> GoalRecord:
        for record in self.goal_records.values():
            if record.handle is handle:
                return record
        raise RuntimeError('Nav2 goal handle has no ownership record')

    def _observe_goal_result(self, future, record: GoalRecord) -> None:
        try:
            status = future.result().status
            if status in TERMINAL_STATUSES:
                record.terminal_status = status
            else:
                record.error = f'non-terminal goal result status {status}'
        except Exception as error:  # noqa: BLE001 - transport failure is not goal termination
            record.error = f'goal result failed: {error}'

    def _ensure_cancel_requested(self, record: GoalRecord):
        """Attempt at most one cancel per goal, including synchronous failures."""
        if (self.shutdown_state == 'CLOSED' or record.resolved
                or record.handle is None or record.cancel_attempted):
            return record.cancel_future
        if (self.shutdown_state == 'STOPPING'
                and monotonic() >= self.shutdown_request.started_at + SHUTDOWN_TIMEOUT_SEC):
            return record.cancel_future
        record.cancel_attempted = True
        try:
            record.cancel_future = record.handle.cancel_goal_async()
            record.cancel_future.add_done_callback(
                lambda future: self._observe_cancel_response(future, record))
        except Exception as error:  # noqa: BLE001
            record.error = f'goal cancel failed: {error}'
            self.get_logger().warning(record.error)
        return record.cancel_future

    def _observe_cancel_response(self, future, record: GoalRecord) -> None:
        try:
            response = future.result()
            record.cancel_confirmed = response.return_code == 0 and any(
                goal_uuids_equal(item.goal_id.uuid, record.goal_id.uuid)
                for item in response.goals_canceling)
            if not record.cancel_confirmed:
                record.error = 'Nav2 did not confirm cancellation of the owned goal'
        except Exception as error:  # noqa: BLE001
            record.error = f'goal cancel response failed: {error}'

    def _publish_stop(self, value: bool) -> bool:
        message = Bool()
        message.data = value
        publish = self.stop_publisher.publish
        if value:
            publish(message)
            return True
        if any(not record.resolved for record in self.goal_records.values()):
            return False
        if self._is_shutting_down():
            if not self._cleanup_release_allowed or self.cleanup_io_failed:
                return False
            deadline = self.shutdown_request.started_at + SHUTDOWN_TIMEOUT_SEC
            # Last authorization check immediately before transport. A publish
            # already authorized/started cannot be preempted or retracted by
            # this Python deadline; elapsed transport time is checked in cleanup.
            now = monotonic()
            if now >= deadline or deadline - now < SHUTDOWN_ZERO_SETTLE_SEC:
                return False
            publish(message)
            return True
        return self._commit_mission(publish, message)

    def _publish_safety_zero(self) -> None:
        """Stop drivers that retain the last velocity when mux output goes quiet."""
        self.safety_cmd_vel_publisher.publish(Twist())

    def _begin_holding_stop(self) -> None:
        """Hold position by commanding zero velocity without engaging hardware emergency stop."""
        if self._is_shutting_down():
            return
        if not self._commit_state(holding_stop=True):
            return
        self._publish_stop(False)
        self._publish_safety_zero()

    def _send_current_goal(self, *, extend_after: Optional[int] = None) -> bool:
        if any((
                self._is_shutting_down(),
                self.finished,
                self.waiting_until is not None,
                self.person_area_stop_pending,
                self.person_search_waiting,
                self.person_search_faulted,
                self.manual_paused,
        )):
            return False
        waypoint = self.waypoints[self.index]
        through = self.navigation_mode == 'through_poses'
        end_index = self.index
        allow_early_extension = False
        if through and not self.single_goal_recovery:
            end_index = navigation_segment_end(self.waypoints, self.index, self.navigation_only)
            protected_end = split_ambiguous_section(self.waypoints, self.index, end_index)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rclpy.time.Time())
                position = (transform.transform.translation.x, transform.transform.translation.y)
            except TransformException:
                position = (waypoint.x, waypoint.y)
            end_index = limit_section_length(
                self.waypoints, self.index, protected_end, position, self.max_section_length)
            length_end = end_index
            end_index = split_ambiguous_section(self.waypoints, self.index, end_index)
            end_index = split_nearby_endpoint(self.waypoints, self.index, end_index, position)
            allow_early_extension = end_index == length_end and end_index < protected_end
        if extend_after is not None and end_index <= extend_after:
            return False
        poses = []
        stamp = self.get_clock().now().to_msg()
        for item in self.waypoints[self.index:end_index + 1]:
            pose = PoseStamped()
            pose.header.frame_id, pose.header.stamp = self.map_frame, stamp
            pose.pose.position.x, pose.pose.position.y = item.x, item.y
            pose.pose.orientation.z, pose.pose.orientation.w = (
                yaw_to_quaternion(item.angle_radians))
            poses.append(pose)
        if through:
            goal = NavigateThroughPoses.Goal()
            goal.poses = poses
            goal.behavior_tree = self._through_poses_tree()
        else:
            goal = NavigateToPose.Goal()
            goal.pose = poses[0]

        serial = self.goal_serial + 1
        self.get_logger().info(
            f'[WP:{waypoint.number}] {self.index + 1}/{len(self.waypoints)} '
            f'goal=({waypoint.x:.2f}, {waypoint.y:.2f}, '
            f'{waypoint.angle_radians:.2f} rad), '
            f'section_end=WP:{self.waypoints[end_index].number}, poses={len(poses)}')
        # Register ownership before dispatch: an exception is not proof that the
        # server never received the request. Never discard an unresolved record.
        record = GoalRecord(goal_id=GoalInfo().goal_id,
                            start_index=self.index if through else None,
                            end_index=end_index if through else None,
                            allow_early_extension=allow_early_extension,
                            verify_endpoint_on_success=extend_after is not None)
        record.goal_id.uuid = list(uuid4().bytes)
        send = self.action_client.send_goal_async

        def dispatch():
            # Ownership registration is part of this one send commit, so even
            # an interrupted/failed transport remains tracked for cleanup.
            self.goal_serial = serial
            self.goal_records = {
                key: item for key, item in self.goal_records.items() if not item.resolved
            }
            self.goal_records[serial] = record
            try:
                kwargs = {'goal_uuid': record.goal_id}
                if through:
                    kwargs['feedback_callback'] = (
                        lambda message: self._goal_feedback(message, serial))
                record.send_future = send(goal, **kwargs)
            except Exception as error:  # noqa: BLE001
                record.error = f'goal send failed: {error}'
                self.request_shutdown(record.error)
                raise

        if not self._commit_mission(dispatch):
            return False
        record.send_future.add_done_callback(
            lambda completed, sent_serial=serial: self._goal_response(completed, sent_serial))
        return True

    def _try_extend_section(self, record) -> bool:
        """Extend a length-limited horizon while retaining every unpassed pose."""
        if (record is None or not record.allow_early_extension
                or record.accepted is not True or record.terminal_status is not None
                or self.section_switch_distance <= 0.0):
            return False
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException:
            return False
        endpoint = self.waypoints[record.end_index]
        distance = math.hypot(endpoint.x - transform.transform.translation.x,
                              endpoint.y - transform.transform.translation.y)
        if distance > self.section_switch_distance:
            return False
        if not self._send_current_goal(extend_after=record.end_index):
            return False
        self.get_logger().info(
            f'Extended continuous section {distance:.2f} m before WP:{endpoint.number}; '
            'unpassed waypoints retained')
        return True

    def _through_poses_tree(self) -> str:
        """Use traversal progress without an early arrival distance threshold."""
        return str(Path(get_package_share_directory('robotbase_bringup')) / 'behavior_trees' / (
            'navigate_through_poses_w_replanning_goal_patience_and_recovery_no_spin.xml'))

    def _goal_feedback(self, message, serial: int) -> None:
        """Track Nav2's remaining ordered goals without sending another action."""
        record = self.goal_records.get(serial)
        if (record is None or record.end_index is None or self._is_shutting_down()
                or serial != self.goal_serial or self.manual_paused or self.finished):
            return
        remaining = int(message.feedback.number_of_poses_remaining)
        count = record.end_index - record.start_index + 1
        if not 1 <= remaining <= count:
            return
        next_index = record.end_index + 1 - remaining
        if next_index > self.index and self._commit_state(index=next_index):
            self.get_logger().info(
                f'[WP:{self.waypoints[self.index].number}] next unpassed waypoint '
                f'(Nav2 remaining={remaining})')

    def _set_paused(self, request, response):
        """Cancel and retain the next waypoint; resume only after confirmed stop."""
        response.success = False
        if self._is_shutting_down() or self.finished:
            response.message = 'Waypoint follower is stopping or finished'
            return response
        if request.data:
            if self.manual_paused:
                response.success, response.message = True, 'Already paused'
                return response
            if (self.waiting_until is not None or self.transition_in_progress
                    or self.person_area_stop_pending or self.person_search_waiting
                    or self.person_search_faulted):
                response.message = (
                    'Already stopping/waiting at a task waypoint; task state is unchanged')
                return response
            if not self._commit_state(
                    manual_paused=True, manual_pause_started_at=monotonic(),
                    goal_serial=self.goal_serial + 1, holding_stop=True):
                return response
            self._publish_safety_zero()
            for record in self.goal_records.values():
                self._ensure_cancel_requested(record)
            response.success = True
            response.message = f'Pause requested; resume from waypoint index {self.index + 1}'
            self.get_logger().info(response.message)
            return response
        if not self.manual_paused:
            response.success, response.message = True, 'Already running'
            return response
        if any(not record.resolved for record in self.goal_records.values()):
            response.message = 'Waiting for Nav2 cancellation; retry resume after confirmation'
            return response
        if monotonic() - self.manual_pause_started_at < SHUTDOWN_ZERO_SETTLE_SEC:
            response.message = 'Waiting for zero-velocity settling; retry resume shortly'
            return response
        if not self._publish_stop(False):
            response.message = 'Stop lock could not be released'
            return response
        if not self._commit_state(
                manual_paused=False, manual_pause_started_at=None, holding_stop=False,
                goal_handle=None, goal_result_future=None, single_goal_recovery=False):
            return response
        self._send_current_goal()
        response.success, response.message = True, f'Resumed from waypoint index {self.index + 1}'
        return response

    def _continue_manual_pause(self) -> None:
        """Drain cancellation while holding zero, without progressing route/tasks."""
        self._publish_safety_zero()
        elapsed = monotonic() - self.manual_pause_started_at
        if elapsed >= SHUTDOWN_ZERO_SETTLE_SEC:
            self._publish_stop(True)
        for record in self.goal_records.values():
            self._ensure_cancel_requested(record)
        if (elapsed > self.person_area_cancel_timeout_sec
                and any(not record.resolved for record in self.goal_records.values())):
            self.request_shutdown('Manual pause cancellation could not be confirmed')

    def _goal_response(self, future, serial: int) -> None:
        record = self.goal_records.get(serial)
        if record is None:
            return
        try:
            goal_handle = future.result()
        except Exception as error:  # noqa: BLE001 - ROS future can surface transport errors
            record.error = f'goal request failed: {error}'
            self._handle_failure(record.error, serial, unconfirmed=True)
            return
        record.accepted = goal_handle.accepted
        if not goal_handle.accepted:
            self._handle_failure('goal was rejected by Nav2', serial)
            return
        record.handle = goal_handle
        record.goal_id = goal_handle.goal_id
        if serial == self.goal_serial:
            self.goal_handle = goal_handle
        try:
            result_future = goal_handle.get_result_async()
            record.result_future = result_future
            if serial == self.goal_serial:
                self.goal_result_future = result_future
            result_future.add_done_callback(
                lambda completed, sent_serial=serial: self._goal_result(completed, sent_serial))
        except Exception as error:  # noqa: BLE001
            record.error = f'goal result request failed: {error}'
            self._handle_failure(record.error, serial, unconfirmed=True)
        # A stale goal is normally being preempted by a newer NavigateToPose
        # request. Do not cancel it here: an explicit cancel makes Nav2 publish
        # a stop between ordinary pass-through waypoints. Shutdown still owns
        # and cancels every unresolved goal through its GoalRecord.
        if self._is_shutting_down():
            self._ensure_cancel_requested(record)

    def _goal_result(self, future, serial: int) -> None:
        record = self.goal_records.get(serial)
        if record is None:
            return
        self._observe_goal_result(future, record)
        if (self._is_shutting_down() or serial != self.goal_serial
                or self.finished or self.waiting_until is not None):
            return
        try:
            wrapped_result = future.result()
        except Exception as error:  # noqa: BLE001
            self._handle_failure(f'goal result failed: {error}', serial, unconfirmed=True)
            return
        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            if record.verify_endpoint_on_success:
                # During preemption, the old FollowPath can finish before the
                # new plan arrives. Nav2 may then succeed the replacement
                # action while the robot is still at the OLD endpoint.
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame, self.base_frame, rclpy.time.Time())
                    endpoint = self.waypoints[record.end_index]
                    endpoint_distance = math.hypot(
                        endpoint.x - transform.transform.translation.x,
                        endpoint.y - transform.transform.translation.y)
                except TransformException:
                    endpoint_distance = math.inf
                if endpoint_distance > self.precise_threshold:
                    self.get_logger().warning(
                        'Nav2 succeeded the updated section before reaching its endpoint; '
                        f'distance={endpoint_distance:.2f} m. Retrying unpassed waypoints')
                    if self._commit_state(goal_handle=None, goal_result_future=None):
                        self._send_current_goal()
                    return
            if record.end_index is not None:
                if not self._commit_state(index=record.end_index):
                    return
            waypoint = self.waypoints[self.index]
            self.get_logger().info(f'[WP:{waypoint.number}] reached by Nav2')
            self._arrive_at_waypoint(exact_arrival=True)
        elif wrapped_result.status == GoalStatus.STATUS_ABORTED:
            self._handle_failure('Nav2 aborted the waypoint', serial)
        elif wrapped_result.status == GoalStatus.STATUS_CANCELED:
            self._handle_failure('Nav2 canceled the waypoint', serial)
        else:
            self._handle_failure(
                f'Nav2 returned status {wrapped_result.status}', serial, unconfirmed=True)

    def _timer_callback(self) -> None:
        if self._is_shutting_down():
            return
        if self.manual_paused:
            self._continue_manual_pause()
            return
        if self.person_area_stop_pending:
            self._continue_person_area_stop()
            return
        if self.holding_stop:
            self._publish_safety_zero()
            if self.person_search_waiting or self.person_search_faulted:
                self._publish_stop(True)
        if self.finished:
            return
        if self.person_search_waiting:
            self._check_person_search_timeout()
            return
        if self.person_search_faulted:
            return
        if self.transition_in_progress:
            self._check_person_area_cancel_timeout()
            return
        if self.goal_handle is None and self.waiting_until is None:
            return
        if self.waiting_until is not None:
            if self.get_clock().now() >= self.waiting_until:
                if not self._commit_state(holding_stop=False, waiting_until=None):
                    return
                if self.waiting_finishes_route:
                    self._complete_or_loop()
                else:
                    self._advance()
            return

        if self.navigation_mode == 'through_poses':
            record = self.goal_records.get(self.goal_serial)
            if self._try_extend_section(record):
                return
            # Intermediate progress comes from the same BT that removes passed
            # goals. Only a section endpoint can trigger a stop/task locally.
            if record is None or self.index != record.end_index:
                return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as error:
            self.get_logger().warning(
                f'TF {self.map_frame} -> {self.base_frame} unavailable: {error}',
                throttle_duration_sec=2.0)
            return

        waypoint = self.waypoints[self.index]
        dx = waypoint.x - transform.transform.translation.x
        dy = waypoint.y - transform.transform.translation.y
        distance = math.hypot(dx, dy)
        self.get_logger().info(
            f'[WP:{waypoint.number}] distance={distance:.2f} m',
            throttle_duration_sec=1.0)

        uses_person_search = waypoint_uses_person_search(
            waypoint, self.navigation_only)
        needs_pause = waypoint.stop or waypoint.wait_time > 0.0 or uses_person_search
        is_last = self.index == len(self.waypoints) - 1
        if self.navigation_mode == 'through_poses' and not needs_pause:
            # A short recovery/loop section is also one action: wait for its
            # terminal result before changing BT files or starting another.
            return
        if is_last and not needs_pause:
            return
        threshold = waypoint_arrival_threshold(
            waypoint, self.default_threshold, self.precise_threshold,
            self.navigation_only)
        if distance <= threshold:
            self.get_logger().info(
                f'[WP:{waypoint.number}] threshold reached at {distance:.2f} m')
            if not needs_pause and not is_last:
                # Sending a newer NavigateToPose goal preempts the active goal
                # inside Nav2 without injecting a direct zero-velocity command.
                self._advance()
            else:
                self._arrive_at_waypoint(exact_arrival=False)

    def _arrive_at_waypoint(self, exact_arrival: bool) -> None:
        if self._is_shutting_down():
            return
        waypoint = self.waypoints[self.index]
        is_last = self.index == len(self.waypoints) - 1
        wait_seconds = waypoint.wait_time
        if waypoint.stop and wait_seconds <= 0.0:
            wait_seconds = self.stop_duration

        if not exact_arrival and self.goal_handle is not None:
            transition_generation = self.cancel_transition_generation + 1
            started_at = (
                self.get_clock().now()
                if waypoint_uses_person_search(waypoint, self.navigation_only)
                else None)
            if not self._commit_state(
                    transition_in_progress=True, goal_serial=self.goal_serial + 1,
                    cancel_transition_generation=transition_generation,
                    active_cancel_transition_generation=transition_generation,
                    person_area_cancel_started_at=started_at):
                return
            record = self._record_for_handle(self.goal_handle)
            cancel_future = self._ensure_cancel_requested(record)
            if cancel_future is None:
                if self._fail_person_area_cancel(
                        record.error or 'Nav2 cancel request unavailable',
                        transition_generation):
                    return
                self.request_shutdown(record.error or 'Nav2 cancel request unavailable')
                return
            cancel_future.add_done_callback(
                lambda completed: self._cancel_finished(
                    completed, wait_seconds, is_last, transition_generation))
            return
        self._finish_arrival(wait_seconds, is_last)

    def _cancel_transition_is_current(self, transition_generation: int) -> bool:
        return bool(
            not self._is_shutting_down()
            and self.transition_in_progress
            and transition_generation == self.active_cancel_transition_generation
            and not self.finished
            and not self.person_search_faulted
        )

    def _invalidate_cancel_transition(self) -> None:
        self.transition_in_progress = False
        self.active_cancel_transition_generation = None
        self.person_area_cancel_started_at = None

    def _cancel_finished(
            self, future, wait_seconds: float, is_last: bool,
            transition_generation: int) -> None:
        if not self._cancel_transition_is_current(transition_generation):
            self.get_logger().info('stale Nav2 cancel request callback ignored')
            return
        wait_for_result = False
        try:
            response = future.result()
            record = self._record_for_handle(self.goal_handle)
            wait_for_result = response.return_code == 0 and any(
                goal_uuids_equal(item.goal_id.uuid, record.goal_id.uuid)
                for item in response.goals_canceling)
            if not wait_for_result:
                self.get_logger().warning(
                    'Nav2 did not report the previous waypoint as canceling')
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'Waypoint cancel request failed: {error}')
        if wait_for_result:
            if self.goal_result_future is not None:
                self.goal_result_future.add_done_callback(
                    lambda completed: self._cancel_result_finished(
                        completed, wait_seconds, is_last,
                        transition_generation))
                return
            if self._fail_person_area_cancel(
                    'Nav2 cancel result future is unavailable',
                    transition_generation):
                return
        elif self._fail_person_area_cancel(
                'Nav2 did not confirm waypoint cancellation',
                transition_generation):
            return
        self.request_shutdown('Nav2 waypoint cancellation is unconfirmed')

    def _cancel_result_finished(
            self, future, wait_seconds: float, is_last: bool,
            transition_generation: int) -> None:
        if not self._cancel_transition_is_current(transition_generation):
            self.get_logger().info('stale Nav2 cancel result callback ignored')
            return
        try:
            if future.result().status not in TERMINAL_STATUSES:
                raise RuntimeError('Nav2 returned a non-terminal cancel result')
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'Canceled goal result failed: {error}')
            if self._fail_person_area_cancel(
                    f'Nav2 canceled goal result failed: {error}',
                    transition_generation):
                return
            self.request_shutdown(f'Nav2 canceled goal result failed: {error}')
            return
        self._finish_cancel_transition(
            wait_seconds, is_last, transition_generation)

    def _check_person_area_cancel_timeout(self) -> bool:
        if self.person_area_cancel_started_at is None:
            return False
        elapsed = self.get_clock().now() - self.person_area_cancel_started_at
        if elapsed.nanoseconds <= int(self.person_area_cancel_timeout_sec * 1e9):
            return False
        transition_generation = self.active_cancel_transition_generation
        if transition_generation is None:
            return False
        return self._fail_person_area_cancel(
            f'Nav2 cancel confirmation timed out after '
            f'{self.person_area_cancel_timeout_sec:.1f}s',
            transition_generation)

    def _fail_person_area_cancel(
            self, reason: str, transition_generation: int) -> bool:
        """Fail closed if a person-area goal cannot be confirmed canceled."""
        if not self._cancel_transition_is_current(transition_generation):
            return True
        if (
                self.index >= len(self.waypoints)
                or not waypoint_uses_person_search(
                    self.waypoints[self.index], self.navigation_only)
        ):
            return False
        waypoint = self.waypoints[self.index]
        if not self._commit_state(
                transition_in_progress=False, active_cancel_transition_generation=None,
                person_area_cancel_started_at=None, person_search_waiting=False,
                person_search_faulted=True, person_search_fault_reason='nav2_cancel_not_confirmed',
                person_search_goal_index=None, person_search_started_at=None,
                person_search_arrival_acknowledged=False,
                person_search_ack_received_at=None, holding_stop=True):
            return True
        self._begin_person_area_stop(starts_search=False)
        self.get_logger().error(
            f'[WP:{waypoint.number}] {reason}; person search was not started and '
            'navigation is stopping fail-closed')
        return True

    def _finish_cancel_transition(
            self, wait_seconds: float, is_last: bool,
            transition_generation: int) -> None:
        if not self._cancel_transition_is_current(transition_generation):
            self.get_logger().info('stale Nav2 cancel transition ignored')
            return
        if not self._commit_state(
                transition_in_progress=False, active_cancel_transition_generation=None,
                person_area_cancel_started_at=None):
            return
        self._publish_safety_zero()
        self._finish_arrival(wait_seconds, is_last)

    def _finish_arrival(self, wait_seconds: float, is_last: bool) -> None:
        if self._is_shutting_down():
            return
        if not self._commit_state(
                transition_in_progress=False, active_cancel_transition_generation=None,
                person_area_cancel_started_at=None, goal_handle=None, goal_result_future=None):
            return
        waypoint = self.waypoints[self.index]
        if waypoint_uses_person_search(waypoint, self.navigation_only):
            self._start_person_search()
        elif wait_seconds > 0.0:
            self._begin_holding_stop()
            if not self._commit_state(
                    waiting_until=self.get_clock().now() + Duration(seconds=wait_seconds),
                    waiting_finishes_route=is_last):
                return
            self.get_logger().info(
                f'[WP:{waypoint.number}] waiting for {wait_seconds:.1f} seconds')
        elif is_last:
            self._complete_or_loop()
        else:
            self._advance()

    def _publish_person_area_arrival(self, arrived: bool) -> None:
        if arrived and self._is_shutting_down():
            return
        message = Bool()
        message.data = bool(arrived)
        publish = self.person_area_arrived_publisher.publish
        if arrived:
            if not self._commit_mission(publish, message):
                return
        else:
            publish(message)
        waypoint = self.waypoints[self.index]
        self.get_logger().info(
            f'[WP:{waypoint.number}] /person_area_arrived='
            f'{str(bool(arrived)).lower()}')

    def _start_person_search(self) -> None:
        """Pass zero through the mux before latching the person-search stop lock."""
        if self._is_shutting_down():
            return
        waypoint = self.waypoints[self.index]
        if (
                self.person_area_stop_pending
                or self.person_search_waiting
                or self.person_search_faulted
        ):
            self.get_logger().warning(
                f'[WP:{waypoint.number}] duplicate person search start ignored')
            return
        self._begin_person_area_stop(starts_search=True)

    def _begin_person_area_stop(self, *, starts_search: bool) -> None:
        if self._is_shutting_down():
            return
        if not self._commit_state(
                holding_stop=True, person_area_stop_pending=True,
                person_area_stop_started_at=self.get_clock().now(),
                person_area_stop_starts_search=bool(starts_search)):
            return
        if starts_search:
            self._publish_stop(False)
        self._publish_safety_zero()

    def _continue_person_area_stop(self) -> bool:
        """Publish zero through the unlocked mux, then latch the stop lock."""
        if self._is_shutting_down():
            return False
        if not self.person_area_stop_pending or self.person_area_stop_started_at is None:
            return False
        if self.person_area_stop_starts_search:
            self._publish_stop(False)
        self._publish_safety_zero()
        elapsed = self.get_clock().now() - self.person_area_stop_started_at
        if elapsed.nanoseconds <= int(self.person_area_zero_settle_sec * 1e9):
            return False
        starts_search = self.person_area_stop_starts_search
        if not self._commit_state(
                person_area_stop_pending=False, person_area_stop_started_at=None,
                person_area_stop_starts_search=False):
            return False
        self._publish_stop(True)
        if starts_search and not self.person_search_faulted and not self.finished:
            self._activate_person_search()
        return True

    def _invalidate_person_area_stop(self) -> None:
        self.person_area_stop_pending = False
        self.person_area_stop_started_at = None
        self.person_area_stop_starts_search = False

    def _activate_person_search(self) -> None:
        """Publish arrival after the downstream zero command and lock are established."""
        if self._is_shutting_down():
            return
        waypoint = self.waypoints[self.index]
        if not self._commit_state(
                person_search_waiting=True, person_search_faulted=False,
                person_search_fault_reason='', person_search_goal_index=self.index,
                person_search_started_at=self.get_clock().now(),
                person_search_session_id=self.person_search_session_id + 1,
                person_search_arrival_acknowledged=False,
                person_search_ack_received_at=None, person_search_last_detail=''):
            return
        self._publish_person_area_arrival(True)
        self.get_logger().info(
            f'[WP:{waypoint.number}] waiting for Camera arrival ACK '
            f'(session={self.person_search_session_id}, '
            f'ack_timeout={self.person_search_ack_timeout_sec:.1f}s)')

    def _check_person_search_timeout(self) -> bool:
        if self._is_shutting_down():
            return False
        if not self.person_search_waiting or self.person_search_started_at is None:
            return False
        if not self.person_search_arrival_acknowledged:
            elapsed = self.get_clock().now() - self.person_search_started_at
            if elapsed.nanoseconds <= int(self.person_search_ack_timeout_sec * 1e9):
                return False
            self._finish_person_search(
                False, fault_reason='person_search_ack_timeout')
            return True
        if self.person_search_ack_received_at is None:
            return False
        elapsed = self.get_clock().now() - self.person_search_ack_received_at
        if elapsed.nanoseconds <= int(self.person_search_timeout_sec * 1e9):
            return False
        self._finish_person_search(False, timeout=True)
        return True

    def _person_search_done_callback(self, message: Bool) -> None:
        if self._is_shutting_down() or not self.person_search_waiting:
            self.get_logger().info(
                'late/duplicate /person_search_done ignored because no active '
                'person search is waiting')
            return
        if not self.person_search_arrival_acknowledged:
            self.get_logger().warning(
                f'/person_search_done ignored before Camera arrival ACK '
                f'(session={self.person_search_session_id})')
            return
        self._finish_person_search(bool(message.data), timeout=False)

    def _person_search_detail_callback(self, message: String) -> None:
        if self._is_shutting_down() or not self.person_search_waiting:
            return
        self._commit_state(person_search_last_detail=str(message.data))

    def _person_search_ack_callback(self, message: Bool) -> None:
        if self._is_shutting_down() or not self.person_search_waiting or not bool(message.data):
            return
        if self.person_search_arrival_acknowledged:
            self.get_logger().info(
                f'duplicate Camera arrival ACK ignored '
                f'(session={self.person_search_session_id})')
            return
        if not self._commit_state(
                person_search_arrival_acknowledged=True,
                person_search_ack_received_at=self.get_clock().now()):
            return
        waypoint = self.waypoints[self.index]
        self.get_logger().info(
            f'[WP:{waypoint.number}] Camera arrival ACK received '
            f'(session={self.person_search_session_id})')
        self.get_logger().info(
            f'[WP:{waypoint.number}] waiting for Camera safe-to-advance '
            f'(session={self.person_search_session_id}, '
            f'search_timeout={self.person_search_timeout_sec:.1f}s)')

    def _finish_person_search(
            self, safe_to_advance: bool, *, timeout: bool = False,
            fault_reason: str = '') -> None:
        if self._is_shutting_down():
            return
        if not self.person_search_waiting:
            self.get_logger().info(
                'person search completion ignored because no active session is waiting')
            return
        goal_index = self.person_search_goal_index
        waypoint = self.waypoints[self.index]
        self._publish_person_area_arrival(False)
        if not self._commit_state(
                person_search_waiting=False, person_search_started_at=None,
                person_search_arrival_acknowledged=False,
                person_search_ack_received_at=None):
            return

        if goal_index != self.index:
            safe_to_advance = False
            timeout = False
            fault_reason = 'person_search_waypoint_changed'
        elif fault_reason:
            safe_to_advance = False
        elif timeout:
            fault_reason = 'person_search_timeout'
        else:
            fault_reason = 'camera_reported_unsafe_completion'

        if not safe_to_advance:
            if not self._commit_state(
                    person_search_faulted=True, person_search_fault_reason=fault_reason,
                    person_search_goal_index=None, holding_stop=True):
                return
            self._publish_safety_zero()
            self._publish_stop(True)
            self.get_logger().error(
                f'[WP:{waypoint.number}] person search fault: {fault_reason}; '
                'navigation remains stopped')
            return

        if not self._commit_state(
                person_search_faulted=False, person_search_fault_reason='',
                person_search_goal_index=None, holding_stop=False):
            return
        self.get_logger().info(
            f'[WP:{waypoint.number}] Camera reported safe-to-advance')
        if self.index == len(self.waypoints) - 1:
            self._complete_or_loop()
        else:
            if not self._publish_stop(False):
                if not self._commit_state(
                        person_search_faulted=True,
                        person_search_fault_reason='navigation_release_failed',
                        person_search_goal_index=None, holding_stop=True):
                    return
                self._publish_safety_zero()
                self._publish_stop(True)
                self.get_logger().error(
                    f'[WP:{waypoint.number}] navigation stop lock could not be released; '
                    'navigation remains stopped')
                return
            self._advance()

    def _invalidate_person_search_session(self, *, publish_arrival: bool = True) -> bool:
        """Invalidate an active session before shutdown so late results are ignored."""
        had_active_arrival = self.person_search_waiting
        self._invalidate_person_area_stop()
        self.person_search_waiting = False
        self.person_search_faulted = False
        self.person_search_fault_reason = ''
        self.person_search_goal_index = None
        self.person_search_started_at = None
        self.person_search_arrival_acknowledged = False
        self.person_search_ack_received_at = None
        if had_active_arrival and publish_arrival:
            self._publish_person_area_arrival(False)
        return had_active_arrival

    def _advance(self) -> None:
        if self._is_shutting_down():
            return
        if not self._commit_state(index=self.index + 1, goal_handle=None,
                                  single_goal_recovery=False):
            return
        self._send_current_goal()

    def _complete_or_loop(self) -> None:
        if self._is_shutting_down():
            return
        if self.loop:
            if not self._commit_state(index=0, goal_handle=None, single_goal_recovery=False):
                return
            self.get_logger().info('Waypoint loop completed; returning to waypoint 1')
            self._send_current_goal()
            return
        if not self._commit_state(finished=True):
            return
        self.get_logger().info('All waypoints completed')
        self.request_shutdown('route completed')

    def _fail_person_area_navigation(self, reason: str) -> bool:
        """Latch an arrival-before-failure and hold the robot stopped."""
        if self._is_shutting_down():
            return True
        if (
                self.index >= len(self.waypoints)
                or not waypoint_uses_person_search(
                    self.waypoints[self.index], self.navigation_only)
        ):
            return False
        waypoint = self.waypoints[self.index]
        # A failed result request is not evidence that the remote goal ended.
        if not self._commit_state(
                goal_serial=self.goal_serial + 1, transition_in_progress=False,
                person_search_waiting=False, person_search_faulted=True,
                person_search_fault_reason=reason, person_search_goal_index=None,
                person_search_started_at=None,
                person_search_arrival_acknowledged=False,
                person_search_ack_received_at=None, holding_stop=True):
            return True
        self._begin_person_area_stop(starts_search=False)
        self.get_logger().error(
            f'[WP:{waypoint.number}] {reason}; person search was not started and '
            'navigation is stopping fail-closed')
        return True

    def _handle_failure(self, message: str, serial: int, *, unconfirmed=False) -> None:
        if self._is_shutting_down() or serial != self.goal_serial or self.finished:
            return
        waypoint = self.waypoints[self.index]
        self.get_logger().error(f'[WP:{waypoint.number}] {message}')
        if self._fail_person_area_navigation(message):
            return
        if unconfirmed or self.stop_on_failure:
            if not self._commit_state(finished=True):
                return
            self.request_shutdown(message)
        else:
            record = self.goal_records.get(serial)
            if (record is not None and record.end_index is not None
                    and record.end_index > self.index):
                # A path through many points can fail because a later point is
                # blocked. Retry the next unpassed point alone before skipping
                # anything; never skip an entire section containing tasks.
                if not self._commit_state(
                        single_goal_recovery=True, goal_handle=None, goal_result_future=None):
                    return
                self.get_logger().warning('Section failed; retrying the next waypoint alone')
                self._send_current_goal()
                return
            self.get_logger().warning(
                f'[WP:{waypoint.number}] Skipping failed waypoint and continuing '
                f'to next waypoint (stop_on_failure=false)')
            if not self._commit_state(
                    transition_in_progress=False, goal_handle=None, goal_result_future=None):
                return
            if self.index == len(self.waypoints) - 1:
                self._complete_or_loop()
            else:
                self._advance()

    def _cleanup_publish(self, callback, *args) -> bool:
        try:
            return callback(*args) is not False
        except Exception as error:  # noqa: BLE001 - attempt the remaining safety operations
            self.cleanup_io_failed = True
            self.get_logger().warning(f'Cleanup publish skipped: {error}')
            return False

    def release_stop(self) -> bool:
        """
        Drain owned goals once; release only after confirmed remote termination.

        Call outside executor callbacks. Spin only to receive action responses;
        all mission callbacks are gated by the irreversible shutdown state.
        A true return is a software confirmation, not proof of physical stop.
        """
        if self.shutdown_state != 'RUNNING':
            return self.cleanup_confirmed is True
        self.request_shutdown('cleanup')
        self.shutdown_state = 'STOPPING'
        self.finished = True
        self.goal_serial += 1
        self._invalidate_cancel_transition()
        had_active_arrival = self._invalidate_person_search_session(publish_arrival=False)
        self.waiting_until = None
        self.waiting_finishes_route = False
        self.holding_stop = True
        deadline = self.shutdown_request.started_at + SHUTDOWN_TIMEOUT_SEC
        confirm_deadline = deadline - SHUTDOWN_ZERO_SETTLE_SEC
        confirmed = False
        release_forbidden = False
        settle_until = None
        try:
            self.timer.cancel()
            self._cleanup_publish(self._publish_safety_zero)
            self._cleanup_publish(self._publish_stop, True)
            if had_active_arrival:
                self._cleanup_publish(self._publish_person_area_arrival, False)
            while rclpy.ok():
                now = monotonic()
                if settle_until is not None and now >= settle_until:
                    confirmed = not self.cleanup_io_failed and now <= deadline
                    break
                if now >= deadline:
                    break
                for record in self.goal_records.values():
                    self._ensure_cancel_requested(record)
                resolved = all(record.resolved for record in self.goal_records.values())
                now = monotonic()
                if (self.cleanup_io_failed or now > confirm_deadline
                        or (now >= confirm_deadline and not resolved)):
                    release_forbidden = True
                if settle_until is None and resolved and not release_forbidden:
                    # The publisher revalidates ownership, I/O and remaining
                    # budget after message preparation, immediately at dispatch.
                    self._cleanup_release_allowed = True
                    try:
                        released = self._cleanup_publish(self._publish_stop, False)
                    finally:
                        self._cleanup_release_allowed = False
                    if released:
                        settle_until = monotonic() + SHUTDOWN_ZERO_SETTLE_SEC
                    else:
                        release_forbidden = True
                # Missing the release cutoff is irreversible, but is NOT the
                # end of response draining. Keep handling late acceptance and
                # cancel/result callbacks until the absolute deadline. After
                # process destruction arbitrarily late acceptance is unknowable.
                self._cleanup_publish(self._publish_safety_zero)
                if settle_until is None or self.cleanup_io_failed:
                    self._cleanup_publish(self._publish_stop, True)
                spin_deadline = min(deadline, settle_until) if settle_until is not None else (
                    deadline if release_forbidden else confirm_deadline)
                if monotonic() >= deadline:
                    break
                rclpy.spin_once(
                    self, timeout_sec=max(
                        0.0, min(SHUTDOWN_SPIN_SEC, spin_deadline - monotonic())))
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
            confirmed = False
            self.get_logger().warning(f'Cleanup interrupted: {error}')
        finally:
            self._cleanup_release_allowed = False
            if not confirmed:
                self._cleanup_publish(self._publish_safety_zero)
                self._cleanup_publish(self._publish_stop, True)
                unresolved = [
                    str(record.goal_id) for record in self.goal_records.values()
                    if not record.resolved
                ]
                self.get_logger().error(
                    f'Shutdown unconfirmed; requesting stop; unresolved goals={unresolved}')
            self.cleanup_confirmed = confirmed
            self.shutdown_state = 'CLOSED'
        return confirmed


def main(args=None) -> int:
    """Handle signals while keeping the ROS context alive for common cleanup."""
    request = ShutdownRequest()
    previous_handlers = {}
    node: Optional[WaypointFollower] = None
    confirmed = False

    def handle_signal(signum, _frame):
        request.request(signal.Signals(signum).name, signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        node = WaypointFollower(shutdown_request=request)
        node.start()
        while rclpy.ok() and not node._is_shutting_down():
            rclpy.spin_once(node, timeout_sec=SHUTDOWN_SPIN_SEC)
    except (KeyboardInterrupt, SystemExit) as error:
        request.request(type(error).__name__)
    except Exception as error:  # noqa: BLE001
        request.request(f'exception: {error}')
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'waypoint_follow: {error}')
        raise
    finally:
        try:
            if node is not None:
                try:
                    confirmed = node.release_stop()
                finally:
                    node.destroy_node()
        finally:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            finally:
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)
    if not confirmed:
        return 2
    if request.signum is not None:
        return 128 + request.signum
    return 130 if request.reason == 'KeyboardInterrupt' else 0
