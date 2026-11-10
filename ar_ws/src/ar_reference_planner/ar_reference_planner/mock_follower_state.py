"""Publish deterministic typed follower state for development-PC tests only."""

import hashlib
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robotbase_waypoint_interfaces.msg import WaypointFollowerState
import yaml


class MockFollowerState(Node):
    """Expose mutable ROS parameters as a mock read-only follower snapshot."""

    def __init__(self) -> None:
        super().__init__('mock_waypoint_follower_state')
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('next_waypoint_index', 1)
        self.declare_parameter('safe_section_end_index', -1)
        self.declare_parameter('goal_generation', 1)
        self.declare_parameter('navigation_mode', 'through_poses')
        self.declare_parameter('activity', 'navigating')
        self.declare_parameter(
            'state_topic', '/waypoint_follower/reference_state')

        path = Path(str(self.get_parameter('waypoint_file').value)).resolve()
        payload = path.read_bytes()
        document = yaml.safe_load(payload)
        if not isinstance(document, dict) or not isinstance(
                document.get('waypoints'), list) or not document['waypoints']:
            raise ValueError('waypoint_file must contain a non-empty waypoints list')
        self._waypoints = document['waypoints']
        self._route_sha256 = hashlib.sha256(payload).hexdigest()
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            WaypointFollowerState,
            str(self.get_parameter('state_topic').value),
            qos,
        )
        self._timer = self.create_timer(0.5, self._publish)
        self._publish()

    def _publish(self) -> None:
        index = int(self.get_parameter('next_waypoint_index').value)
        if not 0 <= index < len(self._waypoints):
            raise ValueError('next_waypoint_index is outside the mock route')
        safe_end = int(self.get_parameter('safe_section_end_index').value)
        if safe_end < 0:
            safe_end = len(self._waypoints) - 1
        activities = {
            'navigating': WaypointFollowerState.ACTIVITY_NAVIGATING,
            'paused': WaypointFollowerState.ACTIVITY_MANUAL_PAUSED,
            'person_search': WaypointFollowerState.ACTIVITY_PERSON_SEARCH_WAITING,
            'waiting': WaypointFollowerState.ACTIVITY_TASK_WAITING,
        }
        activity_name = str(self.get_parameter('activity').value)
        if activity_name not in activities:
            raise ValueError(f'unsupported mock activity: {activity_name}')

        message = WaypointFollowerState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        message.schema_version = WaypointFollowerState.SCHEMA_VERSION
        message.route_sha256 = self._route_sha256
        message.next_waypoint_index = index
        message.next_waypoint_number = int(
            self._waypoints[index].get('number', index + 1))
        message.navigation_mode = str(
            self.get_parameter('navigation_mode').value)
        message.goal_generation = int(
            self.get_parameter('goal_generation').value)
        message.activity = activities[activity_name]
        message.reference_allowed = (
            message.activity == WaypointFollowerState.ACTIVITY_NAVIGATING)
        message.safe_section_end_index = safe_end
        message.waypoint_count = len(self._waypoints)
        self._publisher.publish(message)


def main(args=None) -> None:
    """Run the development-only state publisher."""
    rclpy.init(args=args)
    node = MockFollowerState()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
