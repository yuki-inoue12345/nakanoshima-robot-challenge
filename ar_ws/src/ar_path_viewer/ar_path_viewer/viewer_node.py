"""ROS 2 viewer node. This node creates subscriptions only, never publishers."""

from pathlib import Path
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .diagnostics import DiagnosticReporter
from .image_source import SimulationSource, make_image_source
from .localization_monitor import (
    LocalizationMonitor,
    LocalizationThresholds,
    amcl_message_to_observation,
)
from .path_adapter import PathStore, path_message_to_array
from .projection import CameraIntrinsics, CameraMount
from .rendering import render_frame
from .tf_adapter import transform_message_points
from .viewer_logic import project_path_state


class ArPathViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("ar_path_viewer")
        self._shutdown_requested = False
        self._resources_released = False
        defaults = {
            "path_topic": "/plan",
            "amcl_pose_topic": "/amcl_pose",
            "base_frame": "robot/base_footprint",
            "localization_global_frame": "map",
            "path_timeout_sec": 6.5,
            "localization_pose_timeout_sec": 3.0,
            "localization_stamp_max_age_sec": 3.0,
            "localization_future_tolerance_sec": 1.5,
            "localization_tf_max_age_sec": 3.0,
            "localization_max_position_stddev_m": 0.75,
            "localization_max_yaw_stddev_rad": 0.75,
            "localization_jump_distance_m": 1.5,
            "localization_jump_yaw_rad": 1.2,
            "localization_max_linear_speed_mps": 1.5,
            "localization_max_angular_speed_radps": 2.0,
            "localization_stationary_hold_distance_m": 0.5,
            "localization_stationary_hold_yaw_rad": 0.5,
            "localization_recovery_samples": 1,
            "localization_recovery_sec": 2.0,
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
            "diagnostic_logging": True,
            "diagnostic_log_interval_sec": 5.0,
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

        try:
            self._diagnostics = DiagnosticReporter(
                self._bool("diagnostic_logging"),
                self.get_logger(),
                self._float("diagnostic_log_interval_sec"),
            )
        except ValueError as exc:
            self._config_error = str(exc)
            self.get_logger().error(f"Invalid diagnostic configuration: {exc}")
            self._diagnostics = DiagnosticReporter(False, self.get_logger())

        self._base_frame = self._str("base_frame")
        try:
            localization_thresholds = LocalizationThresholds(
                expected_frame=self._str("localization_global_frame"),
                pose_timeout_sec=self._float("localization_pose_timeout_sec"),
                stamp_max_age_sec=self._float("localization_stamp_max_age_sec"),
                future_tolerance_sec=self._float("localization_future_tolerance_sec"),
                tf_max_age_sec=self._float("localization_tf_max_age_sec"),
                max_position_stddev_m=self._float(
                    "localization_max_position_stddev_m"
                ),
                max_yaw_stddev_rad=self._float("localization_max_yaw_stddev_rad"),
                jump_distance_m=self._float("localization_jump_distance_m"),
                jump_yaw_rad=self._float("localization_jump_yaw_rad"),
                max_linear_speed_mps=self._float(
                    "localization_max_linear_speed_mps"
                ),
                max_angular_speed_radps=self._float(
                    "localization_max_angular_speed_radps"
                ),
                stationary_hold_distance_m=self._float(
                    "localization_stationary_hold_distance_m"
                ),
                stationary_hold_yaw_rad=self._float(
                    "localization_stationary_hold_yaw_rad"
                ),
                recovery_samples=self._int("localization_recovery_samples"),
                recovery_sec=self._float("localization_recovery_sec"),
            )
            self._localization = LocalizationMonitor(localization_thresholds)
        except ValueError as exc:
            self._config_error = self._config_error or str(exc)
            self.get_logger().error(f"Invalid localization configuration: {exc}")
            self._localization = LocalizationMonitor()
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
        self._amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self._str("amcl_pose_topic"),
            self._on_amcl_pose,
            qos,
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
        raw_frame = str(getattr(getattr(message, "header", None), "frame_id", ""))
        raw_poses = getattr(message, "poses", ())
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        header_stamp_sec = None
        if stamp is not None:
            header_stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self._diagnostics.path_received(raw_frame, len(raw_poses), header_stamp_sec)
        previous = self._path_store.snapshot()
        try:
            points, frame = path_message_to_array(message)
            snapshot = self._path_store.replace(points, frame)
            if len(points) == 0:
                self._diagnostics.path_discarded("empty Path; current route cleared")
            else:
                updated = previous is not None and len(previous.points) > 0
                self._diagnostics.path_accepted(
                    frame, len(points), snapshot.generation, updated
                )
        except ValueError as exc:
            # Do not keep showing a formerly valid route after an invalid update.
            self._path_store.clear()
            self._diagnostics.path_discarded(str(exc))

    def _on_amcl_pose(self, message: PoseWithCovarianceStamped) -> None:
        try:
            observation = amcl_message_to_observation(message)
            status = self._localization.observe(observation, self._ros_now_sec())
        except (AttributeError, TypeError, ValueError) as exc:
            status = self._localization.reject(f"invalid AMCL message: {exc}")
        self._diagnostics.localization_update(
            status.state, status.detail, status.receive_count
        )

    def _ros_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _points_in_base(self, points: np.ndarray, frame_id: str) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._base_frame, frame_id, Time(), timeout=Duration(seconds=0.05)
        )
        return transform_message_points(points, transform)

    def _check_localization_tf(
        self,
    ) -> tuple[bool, str, tuple[float, float, float] | None]:
        global_frame = self._localization.thresholds.expected_frame
        if global_frame == self._base_frame:
            return True, "", None
        try:
            transform = self._tf_buffer.lookup_transform(
                global_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception as exc:
            self._diagnostics.tf_failure(self._base_frame, global_frame, exc)
            return (
                False,
                f"source_frame={self._base_frame!r}, target_frame={global_frame!r}; "
                f"{type(exc).__name__}: {exc}",
                None,
            )
        stamp = transform.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        age = self._ros_now_sec() - stamp_sec
        if stamp_sec <= 0.0 or age > self._localization.thresholds.tf_max_age_sec:
            return (
                False,
                f"localization TF stale: age={age:.3f}s "
                f"limit={self._localization.thresholds.tf_max_age_sec:.3f}s",
                None,
            )
        if age < -self._localization.thresholds.future_tolerance_sec:
            return (
                False,
                f"localization TF is in the future: offset={-age:.3f}s",
                None,
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quaternion = (rotation.x, rotation.y, rotation.z, rotation.w)
        norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
        if not math.isfinite(norm) or norm < 1e-9:
            return False, "localization TF has an invalid quaternion", None
        xq, yq, zq, wq = (float(value) / norm for value in quaternion)
        yaw = math.atan2(
            2.0 * (wq * zq + xq * yq),
            1.0 - 2.0 * (yq * yq + zq * zq),
        )
        tf_pose = (float(translation.x), float(translation.y), float(yaw))
        if not all(math.isfinite(value) for value in tf_pose):
            return False, "localization TF pose contains NaN or Inf", None
        return True, "", tf_pose

    def _tick(self) -> None:
        if self._shutdown_requested:
            return
        image = self._image_source.read()
        state = "NO PATH"
        detail = ""
        if image is None:
            image = np.zeros((self._intrinsics.height, self._intrinsics.width, 3), np.uint8)
            state = "CAMERA UNAVAILABLE"
            detail = "image source returned no frame"
            frame = render_frame(image, [], state)
        elif self._camera_error:
            state = "CAMERA UNAVAILABLE"
            detail = self._camera_error
            frame = render_frame(image, [], state, detail=detail)
        elif self._config_error:
            state = "INVALID CONFIG"
            detail = self._config_error
            frame = render_frame(image, [], state, detail=detail)
        else:
            if image.shape[:2] != (self._intrinsics.height, self._intrinsics.width):
                image = cv2.resize(image, (self._intrinsics.width, self._intrinsics.height))
            state, snapshot = self._path_store.state()
            segments = []
            if state == "ACTIVE" and snapshot is not None:
                tf_available, tf_detail, tf_pose = self._check_localization_tf()
                localization = self._localization.status(
                    freshness_confirmed=tf_available,
                    tf_pose=tf_pose,
                )
                self._diagnostics.localization_update(
                    localization.state,
                    localization.detail,
                    localization.receive_count,
                )
                if not localization.stable:
                    state = localization.state
                    detail = localization.detail
                elif not tf_available:
                    state = "TF UNAVAILABLE"
                    detail = tf_detail
                else:
                    try:
                        state, segments = project_path_state(
                            snapshot.points, snapshot.frame_id, self._base_frame,
                            self._points_in_base, self._mount, self._intrinsics, self._near,
                            on_tf_error=lambda exc: self._diagnostics.tf_failure(
                                snapshot.frame_id, self._base_frame, exc
                            ),
                        )
                        if state == "PATH OUT OF VIEW":
                            self._diagnostics.path_out_of_view(
                                snapshot.frame_id, len(snapshot.points)
                            )
                            detail = (
                                f"frame_id={snapshot.frame_id!r}, "
                                f"points={len(snapshot.points)}"
                            )
                        elif state == "TF UNAVAILABLE":
                            detail = (
                                f"source_frame={snapshot.frame_id!r}, "
                                f"target_frame={self._base_frame!r}"
                            )
                        else:
                            detail = (
                                f"generation={snapshot.generation}, "
                                f"frame_id={snapshot.frame_id!r}, "
                                f"points={len(snapshot.points)}, segments={len(segments)}"
                            )
                    except ValueError as exc:
                        state = "INVALID CONFIG"
                        detail = str(exc)
            elif state == "NO PATH":
                detail = self._diagnostics.no_path_detail()
            elif state == "PATH STALE" and snapshot is not None:
                age = max(0.0, self._path_store.age())
                detail = (
                    f"age={age:.3f}s timeout={self._path_store.timeout_sec:.3f}s "
                    f"generation={snapshot.generation}"
                )
            frame = render_frame(
                image, segments, state, self._color, self._thickness, detail=detail
            )
        self._diagnostics.state_changed(state, detail)
        if self._writer is not None:
            self._writer.write(frame)
        if self._display:
            cv2.imshow("AR Path Viewer", frame)
            wait_key = getattr(cv2, "waitKeyEx", cv2.waitKey)
            if is_exit_key(wait_key(1)):
                self._request_shutdown("keyboard q/Q/F10")

    def _request_shutdown(self, reason: str) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.get_logger().info(f"AR Path Viewer shutdown requested: {reason}")
        self._release_resources()
        shutdown_ros_context()

    def _release_resources(self) -> None:
        if self._resources_released:
            return
        self._resources_released = True
        self._image_source.close()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._display:
            cv2.destroyAllWindows()

    def destroy_node(self):
        self._release_resources()
        return super().destroy_node()


def is_exit_key(key_code: int) -> bool:
    """Recognize q/Q and common Win32, GTK, and Qt F10 codes."""
    if key_code is None or key_code < 0:
        return False
    code = int(key_code) & 0xFFFFFFFF
    return code in {ord("q"), ord("Q"), 0x00790000, 0x0000FFC7, 0x01000039}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArPathViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        shutdown_ros_context()


def shutdown_ros_context() -> None:
    """Idempotently shut down rclpy after node destruction."""
    rclpy.try_shutdown()
