"""
Read-only compatibility bridge for the Launcher2 legacy follower.

The bridge publishes the existing typed state contract only after it has
verified the legacy follower parameters, an active Nav2 goal UUID, and a new
plan endpoint that uniquely matches the selected route.  It remains silent
when the modern follower's native ``reference_state_topic`` parameter exists.
"""

import math
from pathlib import Path
from time import monotonic

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from nav_msgs.msg import Path as PathMessage
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, ListParameters
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_action_status_default,
    QoSProfile,
    ReliabilityPolicy,
)
from robotbase_waypoint_interfaces.msg import WaypointFollowerState

from .legacy_compatibility import resolve_legacy_progress
from .route import load_route


_ACTIVE_STATUSES = {
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
}


def _goal_uuid(goal_info) -> str:
    values = bytes(int(value) & 0xff for value in goal_info.goal_id.uuid)
    return values.hex()


class LegacyFollowerBridge(Node):
    """Convert only independently verifiable legacy observations to state."""

    def __init__(self) -> None:
        super().__init__('legacy_follower_reference_bridge')
        defaults = {
            'follower_node_name': '/koko_waypoint_follower',
            'output_topic': '/waypoint_follower/reference_state',
            'plan_topic': '/plan',
            'parameter_poll_sec': 1.0,
            'publish_period_sec': 0.20,
            'action_status_timeout_sec': 2.5,
            'plan_timeout_sec': 6.5,
            'endpoint_tolerance_m': 0.40,
            'endpoint_separation_m': 0.20,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._validate_parameters()
        follower_name = str(self.get_parameter('follower_node_name').value).rstrip('/')
        # Probe ONLY services actually required by this read-only bridge.
        # AsyncParameterClient.services_are_ready() checks unrelated parameter
        # mutation services too, which can strand the bridge at startup.
        self._parameter_client = AsyncParameterClient(self, follower_name)
        self._parameter_list_service = self.create_client(
            ListParameters, follower_name + '/list_parameters'
        )
        self._parameter_get_service = self.create_client(
            GetParameters, follower_name + '/get_parameters'
        )
        self._parameter_future = None
        self._parameter_future_since = None
        self._next_parameter_query = monotonic()
        self._legacy_verified = False
        self._modern_follower = False
        self._follower_parameters_available = False
        self._route = None
        self._navigation_mode = None
        self._map_frame = 'map'
        self._publisher = None
        self._active_goals = {'to_pose': (), 'through_poses': ()}
        self._status_received_at = {'to_pose': None, 'through_poses': None}
        self._goal_uuid = None
        self._goal_generation = 0
        self._plan_sequence = 0
        self._minimum_plan_sequence = 1
        self._plan_endpoint = None
        self._plan_received_at = None
        self._to_pose_feedback = None
        self._through_feedback = None
        self._last_progress = None
        self._last_reason = None

        self.create_subscription(
            PathMessage,
            str(self.get_parameter('plan_topic').value),
            self._on_plan,
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            lambda message: self._on_status('to_pose', message),
            qos_profile_action_status_default,
        )
        self.create_subscription(
            GoalStatusArray,
            '/navigate_through_poses/_action/status',
            lambda message: self._on_status('through_poses', message),
            qos_profile_action_status_default,
        )
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self._on_to_pose_feedback,
            10,
        )
        self.create_subscription(
            NavigateThroughPoses.Impl.FeedbackMessage,
            '/navigate_through_poses/_action/feedback',
            self._on_through_feedback,
            10,
        )
        period = float(self.get_parameter('publish_period_sec').value)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            'legacy_follower_bridge_started mode=read_only fail_closed=true'
        )

    def _validate_parameters(self) -> None:
        for name in (
            'parameter_poll_sec',
            'publish_period_sec',
            'action_status_timeout_sec',
            'plan_timeout_sec',
            'endpoint_tolerance_m',
        ):
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        separation = float(self.get_parameter('endpoint_separation_m').value)
        if not math.isfinite(separation) or separation < 0.0:
            raise ValueError(
                'endpoint_separation_m must be finite and non-negative'
            )

    def _state_qos(self) -> QoSProfile:
        return QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

    def _ensure_publisher(self) -> None:
        if self._publisher is None:
            topic = str(self.get_parameter('output_topic').value)
            self._publisher = self.create_publisher(
                WaypointFollowerState,
                topic,
                self._state_qos(),
            )
            self._log_info(
                f'legacy_follower_state_publisher_created topic={topic}'
            )

    def _remove_publisher(self) -> None:
        if self._publisher is not None:
            self.destroy_publisher(self._publisher)
            self._publisher = None

    def _reset_goal_binding(self, reason: str) -> None:
        """Invalidate all observations that belonged to an earlier route."""
        self._goal_uuid = None
        self._goal_generation += 1
        self._plan_endpoint = None
        self._plan_received_at = None
        self._to_pose_feedback = None
        self._through_feedback = None
        self._last_progress = None
        self._minimum_plan_sequence = self._plan_sequence + 1
        self._log_reason(reason)

    def _query_parameters(self, now: float) -> None:
        if self._parameter_future is not None:
            started = getattr(self, '_parameter_future_since', None)
            if started is None or now - started < 2.5:
                return
            # An RPC can remain unresolved if the Follower restarts between
            # readiness and call_async. Discard the stale future and retry.
            self._parameter_future = None
            self._follower_parameters_available = False
            self._log_reason('follower_parameter_response_timeout')
        if now < self._next_parameter_query:
            return
        poll = float(self.get_parameter('parameter_poll_sec').value)
        self._next_parameter_query = now + poll
        list_service = getattr(self, '_parameter_list_service', None)
        ready = (
            list_service.service_is_ready()
            if list_service is not None
            else self._parameter_client.services_are_ready()
        )
        if not ready:
            self._follower_parameters_available = False
            self._log_reason(
                'follower_parameter_list_service_unavailable'
                if list_service is not None else 'follower_parameter_service_unavailable'
            )
            return
        # Do not include the optional modern-only reference_state_topic in a
        # GetParameters request.  Jazzy's rclpy parameter service returns an
        # empty value array when any requested name is undeclared, which made
        # all three valid legacy parameters appear missing.  Listing the exact
        # optional name first works for both declared and undeclared cases.
        if list_service is None:
            future = self._parameter_client.list_parameters(
                prefixes=['reference_state_topic'], depth=1
            )
        else:
            request = ListParameters.Request()
            request.prefixes = ['reference_state_topic']
            request.depth = 1
            future = list_service.call_async(request)
        self._parameter_future = future
        self._parameter_future_since = now
        future.add_done_callback(self._on_parameter_list)

    def _parameter_transport_failed(self, reason: str) -> None:
        """Keep a known route fail-closed while its parameter link is down."""
        self._follower_parameters_available = False
        self._log_reason(reason)

    def _on_parameter_list(self, future) -> None:
        if future is not self._parameter_future:
            return  # Ignore a timed-out RPC after the next retry started.
        self._parameter_future = None
        try:
            names = tuple(str(name) for name in future.result().result.names)
        except Exception as error:  # noqa: BLE001 - ROS service transport
            self._parameter_transport_failed(
                f'follower_parameter_list_failed {error}'
            )
            return
        if 'reference_state_topic' in names:
            self._follower_parameters_available = True
            self._modern_follower = True
            self._legacy_verified = False
            self._remove_publisher()
            self._log_reason('modern_follower_native_state')
            return

        self._modern_follower = False
        try:
            get_service = getattr(self, '_parameter_get_service', None)
            if get_service is None:
                parameter_future = self._parameter_client.get_parameters([
                    'waypoint_file', 'navigation_mode', 'map_frame',
                ])
            else:
                if not get_service.service_is_ready():
                    self._parameter_transport_failed(
                        'follower_parameter_get_service_unavailable'
                    )
                    return
                request = GetParameters.Request()
                request.names = ['waypoint_file', 'navigation_mode', 'map_frame']
                parameter_future = get_service.call_async(request)
        except Exception as error:  # noqa: BLE001 - ROS service transport
            self._parameter_transport_failed(
                f'follower_parameter_query_start_failed {error}'
            )
            return
        self._parameter_future = parameter_future
        self._parameter_future_since = monotonic()
        parameter_future.add_done_callback(self._on_parameters)

    def _on_parameters(self, future) -> None:
        if future is not self._parameter_future:
            return
        self._parameter_future = None
        try:
            values = list(future.result().values)
        except Exception as error:  # noqa: BLE001 - ROS service transport
            # Keep the last verified route identity so the timer can continue
            # publishing an explicit, disallowed HOLD snapshot.  Clearing it
            # here would let the last NAVIGATING heartbeat age out and could
            # briefly expose path-derived UI again after a subscriber restart.
            self._parameter_transport_failed(
                f'follower_parameter_query_failed {error}'
            )
            return
        if len(values) != 3:
            self._parameter_transport_failed(
                'follower_parameter_response_incomplete '
                f'expected=3 actual={len(values)}'
            )
            return
        self._follower_parameters_available = True
        if (
            values[0].type != ParameterType.PARAMETER_STRING
            or values[1].type != ParameterType.PARAMETER_STRING
        ):
            self._legacy_verified = False
            self._log_reason('legacy_follower_required_parameter_missing')
            return
        route_path = Path(values[0].string_value).expanduser()
        mode = values[1].string_value
        frame = (
            values[2].string_value
            if values[2].type == ParameterType.PARAMETER_STRING
            else 'map'
        )
        try:
            route = load_route(str(route_path))
            if mode not in ('to_pose', 'through_poses'):
                raise ValueError(f'invalid navigation_mode={mode!r}')
            if not route_path.is_absolute():
                raise ValueError('waypoint_file is not absolute')
        except (OSError, ValueError) as error:
            self._legacy_verified = False
            self._log_reason(f'legacy_follower_configuration_invalid {error}')
            return
        identity = (route.sha256, mode, frame)
        previous = None if self._route is None else (
            self._route.sha256,
            self._navigation_mode,
            self._map_frame,
        )
        self._route = route
        self._navigation_mode = mode
        self._map_frame = frame
        self._legacy_verified = True
        self._ensure_publisher()
        if identity != previous:
            self._reset_goal_binding('legacy_follower_configuration_changed')
        self._log_reason(
            'legacy_follower_verified '
            f'route={route.sha256[:12]} mode={mode}'
        )

    def _on_status(self, mode: str, message: GoalStatusArray) -> None:
        active = tuple(
            sorted(
                _goal_uuid(item.goal_info)
                for item in message.status_list
                if int(item.status) in _ACTIVE_STATUSES
            )
        )
        self._active_goals[mode] = active
        self._status_received_at[mode] = monotonic()
        if mode != self._navigation_mode:
            return
        if len(active) == 1 and active[0] != self._goal_uuid:
            previous_progress = self._last_progress
            self._goal_uuid = active[0]
            self._goal_generation += 1
            self._plan_endpoint = None
            self._plan_received_at = None
            self._to_pose_feedback = None
            self._through_feedback = None
            self._last_progress = None
            self._minimum_plan_sequence = self._plan_sequence + 1
            if previous_progress is not None:
                # Invalidate the old goal immediately.  Keeping its verified
                # index in a disallowed transition snapshot is safer than
                # leaving the preceding navigating heartbeat valid until its
                # timeout while the replacement plan is being generated.
                self._publish_state(
                    previous_progress,
                    WaypointFollowerState.ACTIVITY_TRANSITIONING,
                    False,
                )
            self._log_reason('waiting_for_plan_after_nav2_goal_change')

    def _on_plan(self, message: PathMessage) -> None:
        self._plan_sequence += 1
        if not message.poses or str(message.header.frame_id) != self._map_frame:
            self._plan_endpoint = None
            self._plan_received_at = None
            return
        position = message.poses[-1].pose.position
        x, y = float(position.x), float(position.y)
        if not math.isfinite(x) or not math.isfinite(y):
            self._plan_endpoint = None
            self._plan_received_at = None
            return
        self._plan_endpoint = (x, y, self._plan_sequence)
        self._plan_received_at = monotonic()

    def _on_through_feedback(self, message) -> None:
        try:
            goal_uuid = bytes(int(value) & 0xff for value in message.goal_id.uuid).hex()
            remaining = int(message.feedback.number_of_poses_remaining)
        except (AttributeError, TypeError, ValueError):
            return
        self._through_feedback = (goal_uuid, remaining, monotonic())

    def _on_to_pose_feedback(self, message) -> None:
        """Record only the UUID and receipt time as positive goal liveness."""
        try:
            goal_uuid = bytes(
                int(value) & 0xff for value in message.goal_id.uuid
            ).hex()
        except (AttributeError, TypeError, ValueError):
            return
        self._to_pose_feedback = (goal_uuid, monotonic())

    def _feedback_liveness(self, mode: str, goal_uuid: str) -> float | None:
        """Return a matching feedback receipt time without inventing a goal."""
        if mode == 'to_pose':
            feedback = self._to_pose_feedback
            if feedback is None or feedback[0] != goal_uuid:
                return None
            return feedback[1]
        feedback = self._through_feedback
        if feedback is None or feedback[0] != goal_uuid:
            return None
        return feedback[2]

    def _active_goal(self, now: float) -> tuple[str | None, str]:
        mode = self._navigation_mode
        if mode not in self._active_goals:
            return None, 'legacy_navigation_mode_unavailable'
        received = self._status_received_at[mode]
        timeout = float(self.get_parameter('action_status_timeout_sec').value)
        active = self._active_goals[mode]
        if len(active) == 0:
            return None, 'legacy_nav2_goal_inactive'
        if len(active) != 1:
            return None, 'legacy_nav2_goal_ambiguous'
        other_mode = (
            'through_poses' if mode == 'to_pose' else 'to_pose'
        )
        if self._active_goals[other_mode]:
            return None, 'legacy_nav2_action_mode_ambiguous'
        if received is None:
            return None, 'legacy_action_status_stale'
        feedback_received = self._feedback_liveness(mode, active[0])
        latest_liveness = max(
            received,
            feedback_received if feedback_received is not None else received,
        )
        if now - latest_liveness > timeout:
            return None, 'legacy_action_status_stale'
        return active[0], 'legacy_nav2_goal_active'

    def _resolve_progress(self, now: float):
        goal_uuid, reason = self._active_goal(now)
        if goal_uuid is None:
            return None, reason
        if goal_uuid != self._goal_uuid:
            return None, 'waiting_for_goal_binding'
        endpoint = self._plan_endpoint
        if endpoint is None or endpoint[2] < self._minimum_plan_sequence:
            return None, 'waiting_for_plan_after_nav2_goal_change'
        timeout = float(self.get_parameter('plan_timeout_sec').value)
        if self._plan_received_at is None or now - self._plan_received_at > timeout:
            return None, 'legacy_plan_stale'
        remaining = None
        if self._navigation_mode == 'through_poses':
            feedback = self._through_feedback
            if feedback is None or feedback[0] != goal_uuid:
                return None, 'through_poses_feedback_required'
            if now - feedback[2] > float(
                self.get_parameter('action_status_timeout_sec').value
            ):
                return None, 'through_poses_feedback_stale'
            remaining = feedback[1]
        try:
            progress = resolve_legacy_progress(
                self._route,
                self._navigation_mode,
                endpoint[0],
                endpoint[1],
                poses_remaining=remaining,
                tolerance_m=float(
                    self.get_parameter('endpoint_tolerance_m').value
                ),
                separation_m=float(
                    self.get_parameter('endpoint_separation_m').value
                ),
            )
        except ValueError as error:
            return None, str(error)
        return progress, 'legacy_progress_verified'

    def _publish_state(self, progress, activity: int, allowed: bool) -> None:
        if self._publisher is None or self._route is None or progress is None:
            return
        message = WaypointFollowerState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._map_frame
        message.schema_version = WaypointFollowerState.SCHEMA_VERSION
        message.route_sha256 = self._route.sha256
        message.next_waypoint_index = progress.next_index
        message.next_waypoint_number = self._route.waypoints[
            progress.next_index
        ].number
        message.navigation_mode = self._navigation_mode
        message.goal_generation = self._goal_generation
        message.activity = activity
        message.reference_allowed = bool(allowed)
        message.safe_section_end_index = progress.safe_end_index
        message.waypoint_count = len(self._route.waypoints)
        self._publisher.publish(message)

    def _tick(self) -> None:
        now = monotonic()
        self._query_parameters(now)
        if not self._legacy_verified or self._modern_follower:
            return
        if not self._follower_parameters_available:
            if self._last_progress is not None:
                self._publish_state(
                    self._last_progress,
                    WaypointFollowerState.ACTIVITY_HOLDING_STOP,
                    False,
                )
            return
        progress, reason = self._resolve_progress(now)
        if progress is not None:
            self._last_progress = progress
            self._publish_state(
                progress,
                WaypointFollowerState.ACTIVITY_NAVIGATING,
                True,
            )
            self._log_reason(reason)
            return
        # An inactive/cancelled goal is a confirmed stop.  Other failures are
        # transitions or uncertainty; both suppress all path-derived UI.
        if self._last_progress is not None:
            inactive = reason == 'legacy_nav2_goal_inactive'
            self._publish_state(
                self._last_progress,
                (
                    WaypointFollowerState.ACTIVITY_HOLDING_STOP
                    if inactive
                    else WaypointFollowerState.ACTIVITY_TRANSITIONING
                ),
                False,
            )
        self._log_reason(reason)

    def _log_reason(self, reason: str) -> None:
        if reason == self._last_reason:
            return
        self._last_reason = reason
        message = f'legacy_follower_compatibility reason={reason}'
        if not (
            reason.startswith('legacy_progress_verified')
            or reason.startswith('legacy_follower_verified')
            or reason == 'modern_follower_native_state'
        ):
            self._log_warning(message)
            return
        self._log_info(message)

    def _log_info(self, message: str) -> None:
        """Log INFO from a severity-stable rclpy caller location."""
        self.get_logger().info(message)

    def _log_warning(self, message: str) -> None:
        """Log WARNING from a severity-stable rclpy caller location."""
        self.get_logger().warning(message)


def main(args=None) -> None:
    """Run the compatibility bridge without any control-side interfaces."""
    rclpy.init(args=args)
    node = LegacyFollowerBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            # ros2 launch can forward another SIGINT while rclpy is already
            # tearing down parameter clients. The bridge owns no robot-side
            # resource or control operation that needs a second cleanup pass.
            pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass
