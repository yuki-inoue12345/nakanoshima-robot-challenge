"""ROS 2 viewer node. This node creates subscriptions only, never publishers."""

from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMessage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .image_source import SimulationSource, make_image_source
from .path_adapter import PathStore, path_message_to_array
from .projection import CameraIntrinsics, CameraMount, project_clipped_segments
from .rendering import render_frame
from .tf_adapter import transform_message_points
from .viewer_logic import project_path_state


class ArPathViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("ar_path_viewer")
        defaults = {
            "path_topic": "/plan",
            "base_frame": "robot/base_footprint",
            "path_timeout_sec": 6.5,
            "width": 640,
            "height": 480,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
            "camera_x": 0.20,
            "camera_y": 0.0,
            "camera_z": 0.6967,
            "camera_yaw_deg": 0.0,
            "camera_pitch_up_deg": 7.5,
            "camera_roll_deg": 0.0,
            "near_clip_m": 0.10,
            "line_b": 0,
            "line_g": 255,
            "line_r": 0,
            "line_thickness": 5,
            "image_input_type": "simulation",
            "image_input_uri": "",
            "display": False,
            "output_video": "",
            "fps": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._config_error = ""
        self._camera_error = ""
        try:
            self._intrinsics = CameraIntrinsics(
                width=self._int("width"), height=self._int("height"),
                fx=self._float("fx"), fy=self._float("fy"),
                cx=self._float("cx"), cy=self._float("cy"),
            )
            self._intrinsics.validate()
            self._mount = CameraMount(
                x=self._float("camera_x"), y=self._float("camera_y"),
                z=self._float("camera_z"), yaw_deg=self._float("camera_yaw_deg"),
                pitch_up_deg=self._float("camera_pitch_up_deg"),
                roll_deg=self._float("camera_roll_deg"),
            )
            self._mount.validate()
            self._near = self._float("near_clip_m")
            self._path_store = PathStore(self._float("path_timeout_sec"))
        except ValueError as exc:
            self._config_error = str(exc)
            self.get_logger().error(f"Invalid configuration: {exc}")
            self._intrinsics = CameraIntrinsics()
            self._mount = CameraMount()
            self._near = 0.10
            self._path_store = PathStore()

        try:
            self._image_source = make_image_source(
                self._str("image_input_type"), self._str("image_input_uri"),
                self._intrinsics.width, self._intrinsics.height,
            )
        except OSError as exc:
            self._camera_error = str(exc)
            self.get_logger().error(f"Camera unavailable: {exc}")
            self._image_source = SimulationSource(
                self._intrinsics.width, self._intrinsics.height
            )
        except ValueError as exc:
            self._config_error = str(exc)
            self.get_logger().error(f"Invalid image input configuration: {exc}")
            self._image_source = SimulationSource(
                self._intrinsics.width, self._intrinsics.height
            )

        self._base_frame = self._str("base_frame")
        self._color = (self._int("line_b"), self._int("line_g"), self._int("line_r"))
        self._thickness = self._int("line_thickness")
        self._display = self._bool("display")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            PathMessage, self._str("path_topic"), self._on_path, qos
        )
        self._writer = self._make_writer(self._str("output_video"), self._float("fps"))
        self._timer = self.create_timer(1.0 / max(1.0, self._float("fps")), self._tick)
        self.get_logger().info("AR Path Viewer started in read-only/subscriber-only mode")

    def _str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _make_writer(self, path: str, fps: float):
        if not path:
            return None
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (self._intrinsics.width, self._intrinsics.height),
        )
        if not writer.isOpened():
            self.get_logger().error(f"Cannot open output video: {destination}")
            writer.release()
            return None
        return writer

    def _on_path(self, message: PathMessage) -> None:
        try:
            points, frame = path_message_to_array(message)
            self._path_store.replace(points, frame)
        except ValueError as exc:
            # Do not keep showing a formerly valid route after an invalid update.
            self._path_store.clear()
            self.get_logger().warning(f"Rejected invalid Path: {exc}")

    def _points_in_base(self, points: np.ndarray, frame_id: str) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._base_frame, frame_id, Time(), timeout=Duration(seconds=0.05)
        )
        return transform_message_points(points, transform)

    def _tick(self) -> None:
        image = self._image_source.read()
        if image is None:
            image = np.zeros((self._intrinsics.height, self._intrinsics.width, 3), np.uint8)
            frame = render_frame(image, [], "CAMERA UNAVAILABLE")
        elif self._camera_error:
            frame = render_frame(image, [], "CAMERA UNAVAILABLE", detail=self._camera_error)
        elif self._config_error:
            frame = render_frame(image, [], "INVALID CONFIG", detail=self._config_error)
        else:
            if image.shape[:2] != (self._intrinsics.height, self._intrinsics.width):
                image = cv2.resize(image, (self._intrinsics.width, self._intrinsics.height))
            state, snapshot = self._path_store.state()
            segments = []
            if state == "ACTIVE" and snapshot is not None:
                try:
                    state, segments = project_path_state(
                        snapshot.points, snapshot.frame_id, self._base_frame,
                        self._points_in_base, self._mount, self._intrinsics, self._near,
                    )
                except ValueError as exc:
                    state = "INVALID CONFIG"
                    self.get_logger().warning(str(exc))
            frame = render_frame(image, segments, state, self._color, self._thickness)
        if self._writer is not None:
            self._writer.write(frame)
        if self._display:
            cv2.imshow("AR Path Viewer", frame)
            cv2.waitKey(1)

    def destroy_node(self):
        self._image_source.close()
        if self._writer is not None:
            self._writer.release()
        if self._display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArPathViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
