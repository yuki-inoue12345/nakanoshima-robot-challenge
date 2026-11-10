"""Publish a deterministic development-only map for planner integration tests."""

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class SyntheticMapPublisher(Node):
    """Publish free space with one obstacle crossing a test route leg."""

    def __init__(self) -> None:
        """Create a transient-local synthetic map publisher."""
        super().__init__('synthetic_reference_map')
        self.declare_parameter('map_topic', '/ar_reference_test/map')
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            OccupancyGrid, str(self.get_parameter('map_topic').value), qos
        )
        self._message = self._make_map()
        self._timer = self.create_timer(1.0, self._publish)
        self._publish()

    def _make_map(self) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.frame_id = 'map'
        message.info.resolution = 0.05
        message.info.width = 400
        message.info.height = 400
        message.info.origin.position.x = -5.0
        message.info.origin.position.y = -5.0
        message.info.origin.orientation.w = 1.0
        data = [0] * (message.info.width * message.info.height)
        # Blocks the direct WP4 -> WP5 segment in 0919_nakanosima_way.yaml.
        for row in range(message.info.height):
            y = message.info.origin.position.y + (row + 0.5) * message.info.resolution
            if not 5.35 <= y <= 6.20:
                continue
            for column in range(message.info.width):
                x = message.info.origin.position.x + (column + 0.5) * message.info.resolution
                if 2.25 <= x <= 3.35:
                    data[row * message.info.width + column] = 100
        message.data = data
        return message

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        self._message.header.stamp = now
        self._message.info.map_load_time = now
        self._publisher.publish(self._message)


def main(args=None) -> None:
    """Run the deterministic development map publisher."""
    rclpy.init(args=args)
    node = SyntheticMapPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
