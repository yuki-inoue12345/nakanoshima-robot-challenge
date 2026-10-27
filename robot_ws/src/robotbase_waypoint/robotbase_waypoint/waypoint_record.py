"""Record the robot map pose into a Sirius-compatible waypoint YAML file."""

import math
import os
import select
import sys
import termios
import tty
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from .waypoint_io import append_waypoint


PAUSE_WAYPOINT_KEY = 'p'
PAUSE_WAYPOINT_SECONDS = 10.0


def quaternion_to_yaw(rotation) -> float:
    """Convert a geometry quaternion to planar yaw."""
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


class WaypointRecorder(Node):
    """Record once, after travel distance, or on each Enter press."""

    def __init__(self) -> None:
        super().__init__('koko_waypoint_recorder')
        default_prefix = os.environ.get('ROBOTBASE_TF_PREFIX', 'robot')
        self.declare_parameter('output_file', '')
        self.declare_parameter('continuous', False)
        self.declare_parameter('enter_to_record', False)
        self.declare_parameter('distance_threshold', 2.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', f'{default_prefix}/base_footprint')
        self.declare_parameter('pause_waypoint_topic', '/waypoint_record/pause')

        self.output_file = str(self.get_parameter('output_file').value)
        self.continuous = bool(self.get_parameter('continuous').value)
        self.enter_to_record = bool(self.get_parameter('enter_to_record').value)
        self.distance_threshold = float(self.get_parameter('distance_threshold').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.pause_waypoint_topic = str(
            self.get_parameter('pause_waypoint_topic').value)
        if not self.output_file:
            raise ValueError('output_file parameter is required')
        if self.distance_threshold <= 0.0:
            raise ValueError('distance_threshold must be positive')
        if self.enter_to_record and self.continuous:
            raise ValueError('enter_to_record and continuous cannot both be enabled')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_position = None
        self.latest_pose = None
        self.pause_waypoint_requested = False
        self.pause_waypoint_subscription = self.create_subscription(
            Empty, self.pause_waypoint_topic,
            self._pause_waypoint_callback, 10)
        self._stdin_fd = None
        self._stdin_attributes = None
        if (self.continuous or self.enter_to_record) and sys.stdin.isatty():
            try:
                self._stdin_fd = sys.stdin.fileno()
                self._stdin_attributes = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
            except (AttributeError, OSError, termios.error):
                self._stdin_fd = None
                self._stdin_attributes = None
        self.timer = self.create_timer(0.2, self._timer_callback)
        mode = ('enter' if self.enter_to_record else
                'continuous' if self.continuous else 'one-shot')
        self.get_logger().info(
            f'Recording mode={mode}, output={self.output_file}, '
            f'TF={self.map_frame}->{self.base_frame}')
        if self.continuous:
            self.get_logger().info(
                f"Press '{PAUSE_WAYPOINT_KEY}' to save a waypoint with a "
                f'{PAUSE_WAYPOINT_SECONDS:.0f}-second pause')

    def _pause_waypoint_callback(self, _message: Empty) -> None:
        """Queue a pause waypoint requested from the keyop terminal."""
        self.pause_waypoint_requested = True
        self.get_logger().info(
            f'Received pause waypoint request on {self.pause_waypoint_topic}')

    def _read_record_command(self):
        """Read an optional non-blocking keyboard command."""
        if not (self.continuous or self.enter_to_record):
            return None
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        if self._stdin_fd is not None:
            raw = os.read(self._stdin_fd, 1)
            if not raw:
                self.get_logger().error('Standard input closed; stopping recorder')
                rclpy.shutdown()
                return None
            key = raw.decode(errors='ignore').lower()
        else:
            line = sys.stdin.readline()
            if not line:
                self.get_logger().error('Standard input closed; stopping recorder')
                rclpy.shutdown()
                return None
            key = line.strip().lower()
        if key == PAUSE_WAYPOINT_KEY:
            return 'pause'
        if self.enter_to_record and key == '':
            return 'record'
        return None

    def _timer_callback(self) -> None:
        command = self._read_record_command()
        if command == 'pause':
            self.pause_waypoint_requested = True
        if self.pause_waypoint_requested:
            command = 'pause'
        if self.enter_to_record and command is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as error:
            self.get_logger().warning(
                f'TF unavailable: {error}', throttle_duration_sec=2.0)
            return

        translation = transform.transform.translation
        yaw = quaternion_to_yaw(transform.transform.rotation)
        self.latest_pose = (translation.x, translation.y, yaw)
        if command != 'pause' and self.continuous and self.last_position is not None:
            distance = math.hypot(
                translation.x - self.last_position[0],
                translation.y - self.last_position[1])
            if distance < self.distance_threshold:
                return

        append_options = {}
        if command == 'pause':
            append_options = {
                'stop': True,
                'wait_time': PAUSE_WAYPOINT_SECONDS,
            }
        waypoint = append_waypoint(
            self.output_file, translation.x, translation.y, yaw,
            **append_options)
        self.last_position = (translation.x, translation.y)
        if command == 'pause':
            self.pause_waypoint_requested = False
        pause_note = (
            f' pause={PAUSE_WAYPOINT_SECONDS:.0f}s'
            if command == 'pause' else '')
        self.get_logger().info(
            f'[WP:{waypoint.number}] saved '
            f'({waypoint.x:.3f}, {waypoint.y:.3f}, '
            f'{waypoint.angle_radians:.3f} rad){pause_note}')
        if not self.continuous and not self.enter_to_record:
            rclpy.shutdown()

    def save_final_waypoint(self) -> None:
        """Append the latest pose on shutdown unless it was already saved."""
        if not self.continuous or self.latest_pose is None:
            return
        x, y, yaw = self.latest_pose
        if self.last_position is not None:
            distance = math.hypot(x - self.last_position[0], y - self.last_position[1])
            if distance <= 1e-6:
                return
        waypoint = append_waypoint(self.output_file, x, y, yaw)
        self.last_position = (x, y)
        self.get_logger().info(
            f'[WP:{waypoint.number}] saved final pose '
            f'({waypoint.x:.3f}, {waypoint.y:.3f}, '
            f'{waypoint.angle_radians:.3f} rad)')

    def destroy_node(self):
        """Restore terminal input settings before destroying the ROS node."""
        if self._stdin_fd is not None and self._stdin_attributes is not None:
            try:
                termios.tcsetattr(
                    self._stdin_fd, termios.TCSADRAIN, self._stdin_attributes)
            except (OSError, termios.error):
                pass
            self._stdin_fd = None
            self._stdin_attributes = None
        return super().destroy_node()


def main(args=None) -> None:
    """Run the waypoint recorder."""
    rclpy.init(args=args)
    node: Optional[WaypointRecorder] = None
    try:
        node = WaypointRecorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as error:  # noqa: BLE001
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'waypoint_record: {error}')
        if rclpy.ok():
            rclpy.shutdown()
        raise
    finally:
        if node is not None:
            try:
                node.save_final_waypoint()
            except Exception as error:  # noqa: BLE001
                node.get_logger().error(f'Failed to save final waypoint: {error}')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
