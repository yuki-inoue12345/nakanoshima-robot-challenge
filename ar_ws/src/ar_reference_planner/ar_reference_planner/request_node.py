"""Request a Nav2-computed path from the isolated reference planner."""

import math
from time import monotonic

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathThroughPoses
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robotbase_waypoint_interfaces.msg import ReferencePathState, WaypointFollowerState

from .follower_state import FollowerSnapshot, validate_snapshot
from .route import (
    load_route,
    planar_path_length,
    prefix_pose_count,
    select_horizon_by_index,
)


def _quaternion_from_yaw(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class ReferencePathRequestNode(Node):
    """Publish only successful results from a namespaced planning-only action."""

    def __init__(self) -> None:
        """Load the route and wait for typed follower state and isolated Nav2."""
        super().__init__('reference_path_request')
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('target_length_m', 11.0)
        self.declare_parameter('minimum_length_m', 8.0)
        self.declare_parameter('maximum_length_m', 15.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('action_name', 'compute_path_through_poses')
        self.declare_parameter('planner_state_service', 'planner_server/get_state')
        self.declare_parameter('reference_path_topic', '/ar/reference_path')
        self.declare_parameter(
            'reference_path_state_topic', '/ar/reference_path_state')
        self.declare_parameter(
            'follower_state_topic', '/waypoint_follower/reference_state')
        self.declare_parameter('refresh_period_sec', 3.0)
        self.declare_parameter('result_timeout_sec', 3.0)
        self.declare_parameter('validity_sec', 6.5)
        self.declare_parameter('max_result_stamp_age_sec', 3.0)
        self.declare_parameter('max_follower_state_age_sec', 2.5)

        waypoint_file = str(self.get_parameter('waypoint_file').value)
        if not waypoint_file:
            raise ValueError('waypoint_file is required')
        self._target_length = float(self.get_parameter('target_length_m').value)
        self._minimum_length = float(self.get_parameter('minimum_length_m').value)
        self._maximum_length = float(self.get_parameter('maximum_length_m').value)
        self._refresh_period = float(self.get_parameter('refresh_period_sec').value)
        self._result_timeout = float(self.get_parameter('result_timeout_sec').value)
        self._validity = float(self.get_parameter('validity_sec').value)
        self._max_stamp_age = float(
            self.get_parameter('max_result_stamp_age_sec').value
        )
        self._max_follower_state_age = float(
            self.get_parameter('max_follower_state_age_sec').value
        )
        for name, value in (
            ('target_length_m', self._target_length),
            ('minimum_length_m', self._minimum_length),
            ('maximum_length_m', self._maximum_length),
            ('refresh_period_sec', self._refresh_period),
            ('result_timeout_sec', self._result_timeout),
            ('validity_sec', self._validity),
            ('max_result_stamp_age_sec', self._max_stamp_age),
            ('max_follower_state_age_sec', self._max_follower_state_age),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        if not 0.0 < self._minimum_length <= self._target_length <= self._maximum_length:
            raise ValueError(
                'lengths must satisfy 0 < minimum <= target <= maximum'
            )
        if (self._result_timeout <= 0.0 or self._validity <= 0.0
                or self._max_follower_state_age <= 0.0):
            raise ValueError('timeout, validity and state age must be positive')

        self._map_frame = str(self.get_parameter('map_frame').value)
        self._planner_id = str(self.get_parameter('planner_id').value)
        self._route = load_route(waypoint_file)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            Path, str(self.get_parameter('reference_path_topic').value), qos
        )
        self._state_publisher = self.create_publisher(
            ReferencePathState,
            str(self.get_parameter('reference_path_state_topic').value),
            qos,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_subscription = self.create_subscription(
            WaypointFollowerState,
            str(self.get_parameter('follower_state_topic').value),
            self._follower_state_callback,
            state_qos,
        )
        self._action = ActionClient(
            self,
            ComputePathThroughPoses,
            str(self.get_parameter('action_name').value),
        )
        self._planner_state_client = self.create_client(
            GetState,
            str(self.get_parameter('planner_state_service').value),
        )
        self._planner_active = False
        self._planner_state_future = None
        self._last_planner_state_query = float('-inf')
        self._generation = 0
        self._active_generation = None
        self._request_started = None
        self._goal_handle = None
        self._cancel_requested_generation = None
        self._active_state_identity = None
        self._active_state_snapshot = None
        self._follower_state = None
        self._state_identity = None
        self._horizon = None
        self._next_request_at = monotonic()
        self._last_success_at = None
        self._available = False
        self._one_shot_finished = False
        self._stale_cleared = False
        self._unavailable_reason = None
        self._timer = self.create_timer(0.1, self._tick)
        # Clear any value retained by an already-running consumer before the
        # new planner has produced its first valid result.
        self._invalidate('startup_no_reference_path')
        self.get_logger().info(
            'reference_route_loaded '
            f'route_sha256={self._route.sha256} '
            f'waypoints={len(self._route.waypoints)} '
            'waiting_for_follower_state=true'
        )

    def _state_error(self, snapshot: FollowerSnapshot) -> str:
        return validate_snapshot(
            snapshot,
            self._route,
            self._map_frame,
            self.get_clock().now().nanoseconds,
            self._max_follower_state_age,
        )

    def _follower_state_callback(self, message: WaypointFollowerState) -> None:
        """Accept only fresh state matching the exact loaded route."""
        try:
            snapshot = FollowerSnapshot.from_message(message)
        except ValueError as error:
            self._make_state_unavailable(f'invalid_follower_state {error}')
            return
        error = self._state_error(snapshot)
        if error:
            self._make_state_unavailable(error)
            return

        previous = self._follower_state
        changed = previous is None or snapshot.identity != previous.identity
        self._follower_state = snapshot
        self._state_identity = snapshot.identity
        if not changed:
            return

        self._horizon = select_horizon_by_index(
            self._route,
            snapshot.next_index,
            snapshot.safe_end_index,
            self._target_length,
        )
        self._one_shot_finished = False
        self._next_request_at = monotonic()
        self._invalidate(
            'follower_state_changed '
            f'next_index={snapshot.next_index} '
            f'goal_generation={snapshot.goal_generation}'
        )
        self.get_logger().info(
            'reference_horizon_updated '
            f'route_sha256={snapshot.route_sha256} '
            f'next_index={snapshot.next_index} '
            f'next_waypoint={snapshot.next_number} '
            f'navigation_mode={snapshot.navigation_mode} '
            f'goal_generation={snapshot.goal_generation} '
            f'safe_end_index={snapshot.safe_end_index} '
            f'waypoints={list(self._horizon.waypoint_numbers)} '
            f'selection_length_m={self._horizon.approximate_length_m:.3f} '
            f'boundary={self._horizon.boundary_reason}'
        )
        self._cancel_outdated_request()

    def _make_state_unavailable(self, reason: str) -> None:
        self._follower_state = None
        self._state_identity = None
        self._horizon = None
        self._one_shot_finished = False
        self._invalidate(f'follower_state_unavailable reason={reason}')
        self._cancel_outdated_request()

    def _cancel_outdated_request(self) -> None:
        """Cancel but retain ownership until the sole outstanding request resolves."""
        if self._active_generation is None or self._goal_handle is None:
            return
        if self._active_state_identity == self._state_identity:
            return
        if self._cancel_requested_generation == self._active_generation:
            return
        try:
            self._goal_handle.cancel_goal_async()
            self._cancel_requested_generation = self._active_generation
        except Exception as error:  # noqa: BLE001 - timeout remains the fallback
            self.get_logger().warning(
                f'outdated_reference_cancel_failed error={error}'
            )

    def _tick(self) -> None:
        now = monotonic()
        self._query_planner_state(now)
        if self._follower_state is not None:
            state_error = self._state_error(self._follower_state)
            if state_error:
                self._make_state_unavailable(state_error)
        if self._active_generation is not None:
            if now - self._request_started > self._result_timeout:
                generation = self._active_generation
                outdated = self._active_state_identity != self._state_identity
                if (self._goal_handle is not None
                        and self._cancel_requested_generation != generation):
                    try:
                        self._goal_handle.cancel_goal_async()
                        self._cancel_requested_generation = generation
                    except Exception as error:  # noqa: BLE001
                        self.get_logger().warning(
                            f'timed_out_reference_cancel_failed error={error}'
                        )
                self._active_generation = None
                self._invalidate(f'planning_timeout generation={generation}')
                self._finish_cycle(now, immediate=outdated)
            return
        if (
            self._available and self._last_success_at is not None
            and now - self._last_success_at > self._validity
            and not self._stale_cleared
        ):
            self._invalidate('reference_path_expired')
            self._stale_cleared = True
        if self._follower_state is None or self._horizon is None:
            return
        if not self._planner_active:
            self._invalidate('reference_planner_not_active')
            self.get_logger().info(
                'waiting_for_reference_planner_lifecycle_active',
                throttle_duration_sec=2.0,
            )
            return
        if self._one_shot_finished or now < self._next_request_at:
            return
        if not self._action.server_is_ready():
            self.get_logger().info(
                'waiting_for_reference_planner', throttle_duration_sec=2.0
            )
            return
        self._send_request(now)

    def _query_planner_state(self, now: float) -> None:
        """Poll only this package's namespaced Planner lifecycle state."""
        if self._planner_state_future is not None:
            return
        if now - self._last_planner_state_query < 1.0:
            return
        self._last_planner_state_query = now
        if not self._planner_state_client.service_is_ready():
            self._planner_active = False
            return
        self._planner_state_future = self._planner_state_client.call_async(
            GetState.Request())
        self._planner_state_future.add_done_callback(
            self._planner_state_response)

    def _planner_state_response(self, future) -> None:
        self._planner_state_future = None
        try:
            self._planner_active = (
                future.result().current_state.id == State.PRIMARY_STATE_ACTIVE)
        except Exception as error:  # noqa: BLE001 - retry on the next poll
            self._planner_active = False
            self.get_logger().warning(
                f'reference_planner_state_query_failed error={error}'
            )
        if not self._planner_active:
            self._invalidate('reference_planner_not_active')

    def _send_request(self, now: float) -> None:
        if self._follower_state is None or self._horizon is None:
            return
        self._generation += 1
        generation = self._generation
        state = self._follower_state
        stamp = self.get_clock().now().to_msg()
        goals = []
        for item in self._horizon.goals:
            pose = PoseStamped()
            pose.header.frame_id = self._map_frame
            pose.header.stamp = stamp
            pose.pose.position.x = item.x
            pose.pose.position.y = item.y
            pose.pose.orientation.z, pose.pose.orientation.w = _quaternion_from_yaw(
                item.yaw
            )
            goals.append(pose)
        request = ComputePathThroughPoses.Goal()
        request.goals = goals
        request.planner_id = self._planner_id
        request.use_start = False
        self._active_generation = generation
        self._active_state_identity = state.identity
        self._active_state_snapshot = state
        self._request_started = now
        self._goal_handle = None
        self._cancel_requested_generation = None
        self.get_logger().info(
            'reference_plan_request '
            f'generation={generation} planner={self._planner_id} '
            f'route_sha256={state.route_sha256} '
            f'next_index={state.next_index} '
            f'goal_generation={state.goal_generation} '
            f'waypoints={list(self._horizon.waypoint_numbers)} '
            f'requested_at={self.get_clock().now().nanoseconds / 1e9:.9f}'
        )
        future = self._action.send_goal_async(request)
        future.add_done_callback(
            lambda completed, expected=generation: self._goal_response(completed, expected)
        )

    def _goal_response(self, future, generation: int) -> None:
        if generation != self._active_generation:
            return
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001 - ROS transport exception
            outdated = self._active_state_identity != self._state_identity
            self._active_generation = None
            self._invalidate(f'goal_transport_error generation={generation} error={exc}')
            self._finish_cycle(monotonic(), immediate=outdated)
            return
        if not handle.accepted:
            outdated = self._active_state_identity != self._state_identity
            self._active_generation = None
            self._invalidate(f'goal_rejected generation={generation}')
            self._finish_cycle(monotonic(), immediate=outdated)
            return
        self._goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, expected=generation: self._result(completed, expected)
        )
        if self._active_state_identity != self._state_identity:
            self._cancel_outdated_request()

    def _result(self, future, generation: int) -> None:
        if generation != self._active_generation:
            return
        request_identity = self._active_state_identity
        request_state = getattr(self, '_active_state_snapshot', None)
        self._active_generation = None
        elapsed = monotonic() - self._request_started
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as exc:  # noqa: BLE001 - ROS transport exception
            self._invalidate(f'result_transport_error generation={generation} error={exc}')
            self._finish_cycle(
                monotonic(), immediate=request_identity != self._state_identity
            )
            return
        if self._follower_state is not None:
            state_error = self._state_error(self._follower_state)
            if state_error:
                self._make_state_unavailable(state_error)
        if request_identity != self._state_identity:
            self._invalidate(
                f'outdated_result_rejected generation={generation}'
            )
            self._finish_cycle(monotonic(), immediate=True)
            return
        if request_state is None:
            self._invalidate(
                f'invalid_result generation={generation} reason=missing_request_state'
            )
            self._finish_cycle(monotonic())
            return
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or result.error_code != 0:
            self._invalidate(
                'planning_failed '
                f'generation={generation} status={wrapped.status} '
                f'error_code={result.error_code} error={result.error_msg!r}'
            )
            self._finish_cycle(monotonic())
            return
        error = self._validate_path(result.path)
        if error:
            self._invalidate(f'invalid_result generation={generation} reason={error}')
            self._finish_cycle(monotonic())
            return
        original_length = planar_path_length(result.path.poses)
        count = prefix_pose_count(result.path.poses, self._maximum_length)
        if count < 2:
            self._invalidate(f'invalid_result generation={generation} reason=short_prefix')
            self._finish_cycle(monotonic())
            return
        output = Path()
        output.header = result.path.header
        output.poses = list(result.path.poses[:count])
        output_length = planar_path_length(output.poses)
        self._publisher.publish(output)
        self._publish_state(
            valid=True,
            reason='',
            path=output,
            state=request_state,
            planner_generation=generation,
            path_length_m=output_length,
        )
        self._available = True
        self._unavailable_reason = None
        self._last_success_at = monotonic()
        self._stale_cleared = False
        planning_time = result.planning_time.sec + result.planning_time.nanosec / 1e9
        quality = 'short' if output_length < self._minimum_length else 'target_range'
        self.get_logger().info(
            'reference_plan_success '
            f'generation={generation} frame={output.header.frame_id!r} '
            f'stamp={output.header.stamp.sec + output.header.stamp.nanosec / 1e9:.9f} '
            f'waypoints={list(self._horizon.waypoint_numbers)} '
            f'follower_goal_generation={self._follower_state.goal_generation} '
            f'path_length_m={output_length:.3f} full_length_m={original_length:.3f} '
            f'poses={len(output.poses)} planning_time_sec={planning_time:.6f} '
            f'request_latency_sec={elapsed:.6f} quality={quality}'
        )
        self._finish_cycle(monotonic())

    def _validate_path(self, path: Path) -> str:
        if path.header.frame_id != self._map_frame:
            return f'frame_id={path.header.frame_id!r} expected={self._map_frame!r}'
        stamp = path.header.stamp.sec + path.header.stamp.nanosec / 1e9
        now = self.get_clock().now().nanoseconds / 1e9
        if stamp <= 0.0:
            return 'zero_timestamp'
        if stamp > now + 0.5:
            return 'future_timestamp'
        if now - stamp > self._max_stamp_age:
            return f'stale_timestamp age={now - stamp:.3f}'
        if len(path.poses) < 2:
            return 'fewer_than_two_poses'
        for index, pose in enumerate(path.poses):
            if pose.header.frame_id not in ('', self._map_frame):
                return f'pose_{index}_frame={pose.header.frame_id!r}'
            values = (
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
            if not all(math.isfinite(value) for value in values):
                return f'pose_{index}_nonfinite'
        return ''

    def _invalidate(self, reason: str) -> None:
        if not self._available and self._unavailable_reason == reason:
            return
        message = Path()
        message.header.frame_id = self._map_frame
        message.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(message)
        self._publish_state(
            valid=False,
            reason=reason,
            path=message,
            state=self._follower_state,
            planner_generation=self._generation,
            path_length_m=0.0,
        )
        self._available = False
        self._unavailable_reason = reason
        self.get_logger().warning(f'reference_plan_unavailable reason={reason}')

    def _publish_state(
        self,
        *,
        valid: bool,
        reason: str,
        path: Path,
        state: FollowerSnapshot | None,
        planner_generation: int,
        path_length_m: float,
    ) -> None:
        """Publish Path and its authorizing follower identity atomically."""
        message = ReferencePathState()
        message.header.frame_id = self._map_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.schema_version = ReferencePathState.SCHEMA_VERSION
        message.valid = bool(valid)
        message.invalid_reason = str(reason)
        message.planner_generation = int(planner_generation)
        message.path_length_m = float(path_length_m)
        message.path = path
        if state is not None:
            message.route_sha256 = state.route_sha256
            message.next_waypoint_index = state.next_index
            message.next_waypoint_number = state.next_number
            message.navigation_mode = state.navigation_mode
            message.goal_generation = state.goal_generation
            message.safe_section_end_index = state.safe_end_index
            message.waypoint_count = state.waypoint_count
            message.follower_state_stamp.sec = state.stamp_ns // 1_000_000_000
            message.follower_state_stamp.nanosec = state.stamp_ns % 1_000_000_000
        self._state_publisher.publish(message)

    def _finish_cycle(self, now: float, *, immediate: bool = False) -> None:
        self._goal_handle = None
        self._cancel_requested_generation = None
        self._active_state_identity = None
        self._active_state_snapshot = None
        if immediate:
            self._one_shot_finished = False
            self._next_request_at = now
        elif self._refresh_period == 0.0:
            self._one_shot_finished = True
        else:
            self._next_request_at = now + self._refresh_period


def main(args=None) -> None:
    """Run the reference path request node."""
    rclpy.init(args=args)
    node = None
    try:
        node = ReferencePathRequestNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
