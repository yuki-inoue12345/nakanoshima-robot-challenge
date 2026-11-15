"""ROS 2 viewer node. This node creates subscriptions only, never publishers."""

from pathlib import Path
import math
import traceback

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from robotbase_waypoint_interfaces.msg import (
    ReferencePathState,
    WaypointFollowerState,
)
from tf2_ros import Buffer, TransformListener

from .camera_model import CameraUndistorter
from .costmap_overlay import (
    CostmapOverlaySettings,
    CostmapStore,
    project_costmap_points,
)
from .costmap_explanation import (
    CostmapExplanationSettings,
    explain_costmap_and_path,
)
from .diagnostics import DiagnosticReporter
from .display_window import ACTION_COPY_LOG, ACTION_VIEW_LOG, DisplayWindow
from .image_source import SimulationSource, make_image_source
from .guidance_stability import GuidanceStabilityGate
from .log_gui import LogGui
from .localization_monitor import (
    LocalizationMonitor,
    LocalizationThresholds,
    amcl_message_to_observation,
)
from .offscreen_guidance import GuidanceSettings
from .path_adapter import PathStore, path_message_to_array
from .path_transition import PathTransitionGate
from .projection import (
    DEFAULT_CAMERA_Y_M,
    CameraIntrinsics,
    CameraMount,
    ground_plane_view,
)
from .reference_path import ReferencePathGate
from .rendering import render_frame
from .session_log import SessionLog, resolve_session_paths
from .tf_adapter import transform_message_points
from .viewer_logic import PathDisplayGeometry, evaluate_path_state
from .waypoint_progress import WaypointProgressGate


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
            # Kept off until Phase 4 verifies production keepout/costmap parity.
            "reference_path_enabled": False,
            "reference_path_state_topic": "/ar/reference_path_state",
            "reference_follower_state_topic": "/waypoint_follower/reference_state",
            "reference_path_timeout_sec": 6.5,
            "reference_follower_timeout_sec": 2.5,
            "waypoint_distance_enabled": True,
            # Read-only perception overlay. Production remains opt-in until
            # the NUC topic contract and real C920 alignment are verified.
            "perception_costmap_enabled": False,
            "perception_costmap_topic": "/local_costmap/costmap",
            "perception_costmap_timeout_sec": 1.5,
            "perception_costmap_stamp_max_age_sec": 1.5,
            "perception_costmap_future_tolerance_sec": 0.5,
            "perception_costmap_tf_max_age_sec": 1.5,
            "perception_costmap_high_cost_min": 60,
            "perception_costmap_lethal_cost_min": 100,
            "perception_costmap_max_points_per_class": 450,
            "perception_costmap_plane_height_m": 0.02,
            "perception_costmap_alpha": 0.42,
            "perception_costmap_radius_px": 2,
            "perception_explanation_enabled": False,
            "perception_explanation_forward_min_m": 0.40,
            "perception_explanation_forward_max_m": 2.50,
            "perception_explanation_center_half_width_m": 0.45,
            "perception_explanation_side_outer_width_m": 1.20,
            "perception_explanation_maximum_unknown_fraction": 0.25,
            "perception_explanation_low_cost_max_fraction": 0.08,
            "perception_explanation_blocked_lethal_fraction": 0.03,
            "perception_explanation_side_cost_min_fraction": 0.08,
            "perception_explanation_side_cost_difference": 0.04,
            "perception_explanation_minimum_region_cells": 80,
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
            "camera_undistort": False,
            "camera_undistort_alpha": 0.0,
            "camera_distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "camera_x": 0.20,
            "camera_y": DEFAULT_CAMERA_Y_M,
            "camera_z": 0.6967,
            "camera_yaw_deg": 0.0,
            "camera_pitch_up_deg": 7.5,
            "camera_roll_deg": 0.0,
            "near_clip_m": 0.10,
            "path_display_mode": "elevated",
            "path_display_height_m": 0.50,
            "calibration_profile_label": "",
            "offscreen_guidance_enabled": True,
            "offscreen_guidance_lookahead_m": 1.5,
            "offscreen_guidance_min_remaining_m": 0.6,
            "offscreen_guidance_near_goal_m": 0.6,
            "offscreen_guidance_max_path_start_distance_m": 1.0,
            "offscreen_guidance_straight_angle_deg": 15.0,
            "offscreen_guidance_max_bearing_deg": 80.0,
            "offscreen_guidance_max_bearing_change_deg": 65.0,
            "guidance_require_follower_state": True,
            "guidance_direction_hold_sec": 0.35,
            "path_change_hold_sec": 0.35,
            "path_change_max_displacement_m": 0.60,
            "path_change_max_heading_deg": 35.0,
            "path_overlay_alpha": 0.82,
            "debug_overlay_default": False,
            "guidance_line_b": 0,
            "guidance_line_g": 165,
            "guidance_line_r": 255,
            "line_b": 0,
            "line_g": 255,
            "line_r": 0,
            "line_thickness": 5,
            "reference_line_b": 255,
            "reference_line_g": 255,
            "reference_line_r": 0,
            "reference_line_thickness": 3,
            "reference_dash_px": 14.0,
            "reference_gap_px": 9.0,
            "image_input_type": "simulation",
            "image_input_uri": "",
            "display": False,
            "display_width": 960,
            "display_height": 720,
            "output_video": "",
            "fps": 20.0,
            "diagnostic_logging": True,
            "diagnostic_log_interval_sec": 5.0,
            "log_output": "",
            "log_output_dir": "~/ar_navigation_project/outputs",
            "log_max_entries": 4000,
            "log_copy_max_chars": 100000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._config_error = ""
        self._camera_error = ""
        self._log_copy_max_chars = max(2000, self._int("log_copy_max_chars"))
        log_max_entries = max(100, self._int("log_max_entries"))
        self._session_paths = resolve_session_paths(
            self._str("output_video"),
            self._str("log_output"),
            self._str("log_output_dir"),
        )
        self._session_log = SessionLog(self._session_paths.log, log_max_entries)
        self._log_gui = LogGui()
        try:
            self._source_intrinsics = CameraIntrinsics(
                width=self._int("width"), height=self._int("height"),
                fx=self._float("fx"), fy=self._float("fy"),
                cx=self._float("cx"), cy=self._float("cy"),
            )
            self._source_intrinsics.validate()
            self._undistorter = None
            self._intrinsics = self._source_intrinsics
            if self._bool("camera_undistort"):
                self._undistorter = CameraUndistorter(
                    self._source_intrinsics,
                    self._float_list("camera_distortion_coefficients"),
                    self._float("camera_undistort_alpha"),
                )
                self._intrinsics = self._undistorter.description.output_intrinsics
            self._mount = CameraMount(
                x=self._float("camera_x"), y=self._float("camera_y"),
                z=self._float("camera_z"), yaw_deg=self._float("camera_yaw_deg"),
                pitch_up_deg=self._float("camera_pitch_up_deg"),
                roll_deg=self._float("camera_roll_deg"),
            )
            self._mount.validate()
            self._near = self._float("near_clip_m")
            self._path_display_geometry = PathDisplayGeometry(
                mode=self._str("path_display_mode"),
                height_m=self._float("path_display_height_m"),
            )
            self._path_display_geometry.validate()
            self._calibration_profile_label = self._str(
                "calibration_profile_label"
            ).strip()
            if (
                len(self._calibration_profile_label) > 24
                or any(ord(character) < 32 for character in self._calibration_profile_label)
            ):
                raise ValueError(
                    "calibration_profile_label must be at most 24 printable characters"
                )
            self._path_store = PathStore(self._float("path_timeout_sec"))
            self._reference_gate = ReferencePathGate(
                timeout_sec=self._float("reference_path_timeout_sec"),
                follower_timeout_sec=self._float(
                    "reference_follower_timeout_sec"
                ),
                expected_frame=self._str("localization_global_frame"),
            )
            self._waypoint_progress_gate = WaypointProgressGate(
                timeout_sec=self._float("reference_follower_timeout_sec"),
                expected_frame=self._str("localization_global_frame"),
            )
            self._guidance_settings = GuidanceSettings(
                enabled=self._bool("offscreen_guidance_enabled"),
                lookahead_m=self._float("offscreen_guidance_lookahead_m"),
                min_remaining_m=self._float(
                    "offscreen_guidance_min_remaining_m"
                ),
                near_goal_m=self._float("offscreen_guidance_near_goal_m"),
                max_path_start_distance_m=self._float(
                    "offscreen_guidance_max_path_start_distance_m"
                ),
                straight_angle_deg=self._float(
                    "offscreen_guidance_straight_angle_deg"
                ),
                max_bearing_deg=self._float(
                    "offscreen_guidance_max_bearing_deg"
                ),
                max_bearing_change_deg=self._float(
                    "offscreen_guidance_max_bearing_change_deg"
                ),
            )
            self._guidance_settings.validate()
            self._guidance_stability = GuidanceStabilityGate(
                self._float("guidance_direction_hold_sec")
            )
            self._path_transition_gate = PathTransitionGate(
                hold_sec=self._float("path_change_hold_sec"),
                max_displacement_m=self._float(
                    "path_change_max_displacement_m"
                ),
                max_heading_change_deg=self._float(
                    "path_change_max_heading_deg"
                ),
            )
            self._path_alpha = self._float("path_overlay_alpha")
            if not 0.0 < self._path_alpha <= 1.0:
                raise ValueError("path_overlay_alpha must be in (0, 1]")
            self._perception_costmap_enabled = self._bool(
                "perception_costmap_enabled"
            )
            self._costmap_settings = CostmapOverlaySettings(
                timeout_sec=self._float("perception_costmap_timeout_sec"),
                stamp_max_age_sec=self._float(
                    "perception_costmap_stamp_max_age_sec"
                ),
                future_tolerance_sec=self._float(
                    "perception_costmap_future_tolerance_sec"
                ),
                high_cost_min=self._int("perception_costmap_high_cost_min"),
                lethal_cost_min=self._int(
                    "perception_costmap_lethal_cost_min"
                ),
                max_points_per_class=self._int(
                    "perception_costmap_max_points_per_class"
                ),
                plane_height_m=self._float(
                    "perception_costmap_plane_height_m"
                ),
            )
            self._costmap_settings.validate()
            self._costmap_tf_max_age = self._float(
                "perception_costmap_tf_max_age_sec"
            )
            self._costmap_alpha = self._float("perception_costmap_alpha")
            self._costmap_radius = self._int("perception_costmap_radius_px")
            if (
                not math.isfinite(self._costmap_tf_max_age)
                or self._costmap_tf_max_age <= 0.0
            ):
                raise ValueError(
                    "perception_costmap_tf_max_age_sec must be positive"
                )
            if (
                not math.isfinite(self._costmap_alpha)
                or not 0.0 < self._costmap_alpha <= 1.0
                or self._costmap_radius < 1
            ):
                raise ValueError("invalid costmap overlay alpha or radius")
            self._perception_explanation_enabled = self._bool(
                "perception_explanation_enabled"
            )
            self._costmap_explanation_settings = CostmapExplanationSettings(
                forward_min_m=self._float(
                    "perception_explanation_forward_min_m"
                ),
                forward_max_m=self._float(
                    "perception_explanation_forward_max_m"
                ),
                center_half_width_m=self._float(
                    "perception_explanation_center_half_width_m"
                ),
                side_outer_width_m=self._float(
                    "perception_explanation_side_outer_width_m"
                ),
                maximum_unknown_fraction=self._float(
                    "perception_explanation_maximum_unknown_fraction"
                ),
                low_cost_max_fraction=self._float(
                    "perception_explanation_low_cost_max_fraction"
                ),
                blocked_lethal_fraction=self._float(
                    "perception_explanation_blocked_lethal_fraction"
                ),
                side_cost_min_fraction=self._float(
                    "perception_explanation_side_cost_min_fraction"
                ),
                side_cost_difference=self._float(
                    "perception_explanation_side_cost_difference"
                ),
                minimum_region_cells=self._int(
                    "perception_explanation_minimum_region_cells"
                ),
            )
            self._costmap_explanation_settings.validate()
        except ValueError as exc:
            self._config_error = str(exc)
            self._log_error(f"Invalid configuration: {exc}")
            self._source_intrinsics = CameraIntrinsics()
            self._intrinsics = self._source_intrinsics
            self._undistorter = None
            self._mount = CameraMount()
            self._near = 0.10
            self._path_display_geometry = PathDisplayGeometry()
            self._calibration_profile_label = ""
            self._path_store = PathStore()
            self._reference_gate = ReferencePathGate()
            self._waypoint_progress_gate = WaypointProgressGate()
            self._guidance_settings = GuidanceSettings(enabled=False)
            self._guidance_stability = GuidanceStabilityGate(0.0)
            self._path_transition_gate = PathTransitionGate(0.0)
            self._path_alpha = 1.0
            self._perception_costmap_enabled = False
            self._costmap_settings = CostmapOverlaySettings()
            self._costmap_tf_max_age = 1.5
            self._costmap_alpha = 0.42
            self._costmap_radius = 2
            self._perception_explanation_enabled = False
            self._costmap_explanation_settings = CostmapExplanationSettings()

        self._costmap_store = CostmapStore(self._costmap_settings)

        try:
            self._image_source = make_image_source(
                self._str("image_input_type"), self._str("image_input_uri"),
                self._intrinsics.width, self._intrinsics.height,
            )
        except OSError as exc:
            self._camera_error = str(exc)
            self._log_error(f"Camera unavailable: {exc}")
            self._image_source = SimulationSource(
                self._intrinsics.width, self._intrinsics.height
            )
        except ValueError as exc:
            self._config_error = str(exc)
            self._log_error(f"Invalid image input configuration: {exc}")
            self._image_source = SimulationSource(
                self._intrinsics.width, self._intrinsics.height
            )

        try:
            self._diagnostics = DiagnosticReporter(
                self._bool("diagnostic_logging"),
                self.get_logger(),
                self._float("diagnostic_log_interval_sec"),
                record_sink=self._session_log.record,
            )
        except ValueError as exc:
            self._config_error = str(exc)
            self._log_error(f"Invalid diagnostic configuration: {exc}")
            self._diagnostics = DiagnosticReporter(False, self.get_logger())
        ground_view = ground_plane_view(self._mount, self._intrinsics)

        def geometry_value(value: float | None) -> str:
            return "none" if value is None else f"{value:.3f}"

        self._log_info(
            "camera_configuration "
            f"width={self._intrinsics.width} height={self._intrinsics.height} "
            f"fx={self._intrinsics.fx:.3f} fy={self._intrinsics.fy:.3f} "
            f"cx={self._intrinsics.cx:.3f} cy={self._intrinsics.cy:.3f} "
            f"source_fx={self._source_intrinsics.fx:.3f} "
            f"source_fy={self._source_intrinsics.fy:.3f} "
            f"source_cx={self._source_intrinsics.cx:.3f} "
            f"source_cy={self._source_intrinsics.cy:.3f} "
            f"undistort_enabled={self._undistorter is not None} "
            "distortion_coefficients="
            f"{self._float_list('camera_distortion_coefficients')} "
            f"camera_x={self._mount.x:.4f} camera_y={self._mount.y:.4f} "
            f"camera_z={self._mount.z:.4f} "
            f"camera_yaw_deg={self._mount.yaw_deg:.3f} "
            f"camera_pitch_up_deg={self._mount.pitch_up_deg:.3f} "
            f"camera_roll_deg={self._mount.roll_deg:.3f} "
            f"near_clip_m={self._near:.3f} "
            f"ground_horizon_v_px={geometry_value(ground_view.horizon_v_px)} "
            "ground_bottom_center_camera_distance_m="
            f"{geometry_value(ground_view.bottom_center_camera_distance_m)} "
            "ground_bottom_center_base_x_m="
            f"{geometry_value(ground_view.bottom_center_base_x_m)} "
            "ground_bottom_center_base_y_m="
            f"{geometry_value(ground_view.bottom_center_base_y_m)}"
        )
        self._log_info(
            "output_files "
            f"video={str(self._session_paths.video)!r} "
            f"log={str(self._session_paths.log)!r}"
        )
        self._log_info(
            "offscreen_guidance_configuration "
            f"enabled={self._guidance_settings.enabled} "
            f"lookahead_m={self._guidance_settings.lookahead_m:.3f} "
            f"min_remaining_m={self._guidance_settings.min_remaining_m:.3f} "
            f"near_goal_m={self._guidance_settings.near_goal_m:.3f} "
            "max_path_start_distance_m="
            f"{self._guidance_settings.max_path_start_distance_m:.3f} "
            f"straight_angle_deg={self._guidance_settings.straight_angle_deg:.3f} "
            f"max_bearing_deg={self._guidance_settings.max_bearing_deg:.3f} "
            "max_bearing_change_deg="
            f"{self._guidance_settings.max_bearing_change_deg:.3f}"
        )
        self._log_info(
            "path_display_configuration "
            f"mode={self._path_display_geometry.mode} "
            f"height_m={self._path_display_geometry.height_m:.3f} "
            f"calibration_profile={self._calibration_profile_label!r} "
            "preserves_nav2_xy=true guidance_uses_original_path=true"
        )
        if self._session_log.file_error:
            self._log_warning(
                "Automatic log file unavailable; in-memory log remains active: "
                f"{self._session_log.file_error}"
            )

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
            self._log_error(f"Invalid localization configuration: {exc}")
            self._localization = LocalizationMonitor()
        self._color = (self._int("line_b"), self._int("line_g"), self._int("line_r"))
        self._reference_enabled = self._bool("reference_path_enabled")
        self._waypoint_distance_enabled = self._bool(
            "waypoint_distance_enabled"
        )
        self._reference_color = (
            self._int("reference_line_b"),
            self._int("reference_line_g"),
            self._int("reference_line_r"),
        )
        self._reference_thickness = self._int("reference_line_thickness")
        self._reference_dash_px = self._float("reference_dash_px")
        self._reference_gap_px = self._float("reference_gap_px")
        self._guidance_color = (
            self._int("guidance_line_b"),
            self._int("guidance_line_g"),
            self._int("guidance_line_r"),
        )
        self._thickness = self._int("line_thickness")
        self._display = self._bool("display")
        self._debug_overlay = self._bool("debug_overlay_default")
        self._guidance_require_follower = self._bool(
            "guidance_require_follower_state"
        )
        self._display_window = None
        if self._display:
            try:
                self._display_window = DisplayWindow(
                    self._int("display_width"), self._int("display_height")
                )
            except (AttributeError, RuntimeError, ValueError, cv2.error) as exc:
                self._display = False
                self._log_warning(
                    f"OpenCV display unavailable; recording continues: {exc}"
                )
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
        self._costmap_subscription = None
        if self._perception_costmap_enabled:
            costmap_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._costmap_subscription = self.create_subscription(
                OccupancyGrid,
                self._str("perception_costmap_topic"),
                self._on_local_costmap,
                costmap_qos,
            )
        self._reference_subscription = None
        if self._reference_enabled:
            self._reference_subscription = self.create_subscription(
                ReferencePathState,
                self._str("reference_path_state_topic"),
                self._on_reference_path_state,
                qos,
            )
        reference_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._reference_follower_subscription = None
        if self._reference_enabled or self._waypoint_distance_enabled:
            self._reference_follower_subscription = self.create_subscription(
                WaypointFollowerState,
                self._str("reference_follower_state_topic"),
                self._on_reference_follower_state,
                reference_state_qos,
            )
        self._amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self._str("amcl_pose_topic"),
            self._on_amcl_pose,
            qos,
        )
        output_video = "" if self._session_paths.video is None else str(
            self._session_paths.video
        )
        self._writer = self._make_writer(output_video, self._float("fps"))
        self._timer = self.create_timer(1.0 / max(1.0, self._float("fps")), self._tick)
        self._log_info("AR Path Viewer started in read-only/subscriber-only mode")
        self._log_info(
            "reference_path_configuration "
            f"enabled={self._reference_enabled} "
            f"state_topic={self._str('reference_path_state_topic')!r} "
            f"follower_topic={self._str('reference_follower_state_topic')!r} "
            f"timeout_sec={self._reference_gate.path_store.timeout_sec:.3f} "
            f"follower_timeout_sec={self._reference_gate.follower_timeout_sec:.3f} "
            f"color_bgr={self._reference_color} style=dashed"
        )
        self._log_info(
            "waypoint_distance_configuration "
            f"enabled={self._waypoint_distance_enabled} "
            "label='PLAN LEFT' meaning=remaining_real_plan_polyline "
            "not_waypoint_distance=true"
        )
        self._log_info(
            "navigation_ui_configuration "
            f"debug_overlay={self._debug_overlay} toggle_key='D' "
            f"guidance_require_follower={self._guidance_require_follower} "
            f"direction_hold_sec={self._guidance_stability.hold_sec:.3f}"
        )
        self._log_info(
            "path_transition_configuration "
            f"hold_sec={self._path_transition_gate.hold_sec:.3f} "
            "max_displacement_m="
            f"{self._path_transition_gate.max_displacement_m:.3f} "
            "max_heading_deg="
            f"{self._path_transition_gate.max_heading_change_deg:.1f} "
            f"path_overlay_alpha={self._path_alpha:.2f} "
            "old_path_held=false"
        )
        self._log_info(
            "perception_costmap_configuration "
            f"enabled={self._perception_costmap_enabled} "
            f"topic={self._str('perception_costmap_topic')!r} "
            "message_type='nav_msgs/msg/OccupancyGrid' "
            f"timeout_sec={self._costmap_settings.timeout_sec:.3f} "
            f"high_cost_min={self._costmap_settings.high_cost_min} "
            f"lethal_cost_min={self._costmap_settings.lethal_cost_min} "
            f"max_points_per_class="
            f"{self._costmap_settings.max_points_per_class} "
            "semantic_labels='unknown,high_cost,lethal_cost' "
            "does_not_claim_camera_object_or_danger=true"
        )
        self._log_info(
            "perception_explanation_configuration "
            f"enabled={self._perception_explanation_enabled} "
            "inputs='local_costmap,real_plan' "
            "causal_planner_claim=false danger_label=false"
        )

    def _str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float_list(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    def _log_info(self, message: str) -> None:
        self._session_log.record("INFO", message)
        self.get_logger().info(message)

    def _log_warning(self, message: str) -> None:
        self._session_log.record("WARNING", message)
        self.get_logger().warning(message)

    def _log_error(self, message: str) -> None:
        self._session_log.record("ERROR", message)
        self.get_logger().error(message)

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
            self._log_error(f"Cannot open output video: {destination}")
            writer.release()
            return None
        return writer

    def _handle_display_action(self, action: str) -> None:
        if action == ACTION_VIEW_LOG:
            success, message = self._log_gui.show(
                self._session_log.text(), refresh_source=self._session_log.text
            )
            toast = "Log window opened" if success else "Log view failed"
        elif action == ACTION_COPY_LOG:
            try:
                content = self._session_log.copy_text(self._log_copy_max_chars)
                success, message = self._log_gui.copy(content)
            except Exception as exc:
                success = False
                message = f"ログコピー失敗: {type(exc).__name__}: {exc}"
            toast = "Log copied" if success else "Log copy failed"
        else:
            return
        if success:
            self._log_info(f"log_ui action={action} result=success detail={message}")
        else:
            self._log_warning(f"log_ui action={action} result=failure detail={message}")
        if self._display_window is not None:
            self._display_window.set_status(toast, success)

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

    def _on_local_costmap(self, message: OccupancyGrid) -> None:
        accepted = self._costmap_store.update(message, self._ros_now_sec())
        snapshot = self._costmap_store.snapshot
        detail = self._costmap_store.reason
        if accepted and snapshot is not None:
            detail = (
                f"frame={snapshot.frame_id!r} "
                f"unknown_cells={snapshot.unknown_cells} "
                f"high_cost_cells={snapshot.high_cost_cells} "
                f"lethal_cost_cells={snapshot.lethal_cost_cells} "
                f"sampled_cells={snapshot.sampled_cells}"
            )
        self._diagnostics.costmap_update(accepted, detail)

    def _on_reference_follower_state(
        self, message: WaypointFollowerState
    ) -> None:
        snapshot = self._path_store.snapshot()
        path_generation = 0 if snapshot is None else snapshot.generation
        progress_accepted = self._waypoint_progress_gate.update(
            message, self._ros_now_sec(), path_generation
        )
        if self._reference_enabled:
            accepted = self._reference_gate.update_follower(
                message, self._ros_now_sec()
            )
            self._diagnostics.reference_update(
                "follower_state",
                accepted,
                self._reference_gate.reason,
            )
        elif not progress_accepted:
            self._diagnostics.reference_update(
                "waypoint_progress",
                False,
                self._waypoint_progress_gate.reason,
            )

    def _on_reference_path_state(self, message: ReferencePathState) -> None:
        accepted = self._reference_gate.update_reference(
            message, self._ros_now_sec()
        )
        point_count = len(getattr(getattr(message, "path", None), "poses", ()))
        metadata = self._reference_gate.metadata
        detail = self._reference_gate.reason
        if accepted and metadata is not None:
            detail = (
                f"route={metadata.identity.route_sha256[:12]} "
                f"next_index={metadata.identity.next_index} "
                f"goal_generation={metadata.identity.goal_generation} "
                f"planner_generation={metadata.planner_generation} "
                f"points={point_count} "
                f"path_length_m={metadata.declared_length_m:.3f}"
            )
        self._diagnostics.reference_update("path_state", accepted, detail)

    def _ros_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _points_in_base(self, points: np.ndarray, frame_id: str) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._base_frame, frame_id, Time(), timeout=Duration(seconds=0.05)
        )
        return transform_message_points(points, transform)

    def _costmap_arrays_in_base(
        self, snapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform drawing and evidence cells with one timestamped TF."""
        if snapshot.frame_id == self._base_frame:
            return (
                snapshot.points.copy(),
                snapshot.analysis_points.copy(),
            )
        query_time = Time(nanoseconds=int(round(snapshot.stamp_sec * 1e9)))
        transform = self._tf_buffer.lookup_transform(
            self._base_frame,
            snapshot.frame_id,
            query_time,
            timeout=Duration(seconds=0.05),
        )
        stamp = transform.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        age = self._ros_now_sec() - stamp_sec
        if stamp_sec <= 0.0 or age > self._costmap_tf_max_age:
            raise ValueError(
                f"costmap TF stale age={age:.3f}s "
                f"limit={self._costmap_tf_max_age:.3f}s"
            )
        if age < -self._costmap_settings.future_tolerance_sec:
            raise ValueError(
                f"costmap TF future offset={-age:.3f}s"
            )
        combined = np.concatenate(
            (snapshot.points, snapshot.analysis_points), axis=0
        )
        transformed = transform_message_points(combined, transform)
        split = snapshot.points.shape[0]
        return transformed[:split], transformed[split:]

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
        guidance = None
        waypoint_status = ""
        navigation_status = ""
        if image is None:
            image = np.zeros((self._intrinsics.height, self._intrinsics.width, 3), np.uint8)
            state = "CAMERA UNAVAILABLE"
            detail = "image source returned no frame"
            frame = render_frame(
                image, [], state, debug_overlay=self._debug_overlay
            )
        elif self._camera_error:
            state = "CAMERA UNAVAILABLE"
            detail = self._camera_error
            frame = render_frame(
                image, [], state, detail=detail,
                debug_overlay=self._debug_overlay,
            )
        elif self._config_error:
            state = "INVALID CONFIG"
            detail = self._config_error
            frame = render_frame(
                image, [], state, detail=detail,
                debug_overlay=self._debug_overlay,
            )
        else:
            source_shape = (
                self._source_intrinsics.height,
                self._source_intrinsics.width,
            )
            if image.shape[:2] != source_shape:
                image = cv2.resize(
                    image,
                    (
                        self._source_intrinsics.width,
                        self._source_intrinsics.height,
                    ),
                )
            if self._undistorter is not None:
                image = self._undistorter.apply(image)
            state, snapshot = self._path_store.state()
            segments = []
            reference_segments = []
            reference_enabled = bool(
                getattr(self, "_reference_enabled", False)
            )
            reference_state = "DISABLED"
            reference_snapshot = None
            reference_reason = "disabled_by_configuration"
            reference_gate = getattr(self, "_reference_gate", None)
            if reference_enabled and reference_gate is not None:
                reference_state, reference_snapshot, reference_reason = (
                    reference_gate.state(self._ros_now_sec())
                )
            costmap_enabled = bool(
                getattr(self, "_perception_costmap_enabled", False)
            )
            costmap_state = "DISABLED"
            costmap_snapshot = None
            costmap_reason = "disabled_by_configuration"
            costmap_projection = None
            costmap_analysis_points_base = None
            costmap_explanation = None
            costmap_explanation_reason = "disabled_by_configuration"
            path_points_base = None
            if costmap_enabled:
                costmap_state, costmap_snapshot, costmap_reason = (
                    self._costmap_store.state(self._ros_now_sec())
                )

            localization = None
            tf_available = False
            tf_detail = ""
            if (
                state == "ACTIVE"
                or reference_state == "ACTIVE"
                or costmap_state == "ACTIVE"
            ):
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

            if state == "ACTIVE" and snapshot is not None:
                if not localization.stable:
                    state = localization.state
                    detail = localization.detail
                elif not tf_available:
                    state = "TF UNAVAILABLE"
                    detail = tf_detail
                else:
                    try:
                        evaluation = evaluate_path_state(
                            snapshot.points, snapshot.frame_id, self._base_frame,
                            self._points_in_base, self._mount, self._intrinsics, self._near,
                            display_geometry=self._path_display_geometry,
                            guidance_settings=self._guidance_settings,
                            on_tf_error=lambda exc: self._diagnostics.tf_failure(
                                snapshot.frame_id, self._base_frame, exc
                            ),
                        )
                        state = evaluation.state
                        segments = evaluation.segments
                        projection_stats = evaluation.stats
                        guidance = evaluation.guidance
                        progress = evaluation.progress
                        path_points_base = evaluation.points_base
                        transition = None
                        transition_gate = getattr(
                            self, "_path_transition_gate", None
                        )
                        if (
                            transition_gate is not None
                            and state in ("ACTIVE", "PATH OUT OF VIEW")
                        ):
                            transition = transition_gate.update(
                                snapshot.generation,
                                snapshot.points,
                                self._ros_now_sec(),
                            )
                        hold_status = self._waypoint_progress_gate.navigation_hold_status(
                            self._ros_now_sec()
                        )
                        if hold_status:
                            if transition_gate is not None:
                                transition_gate.clear()
                            self._guidance_stability.clear()
                            state = "NAVIGATION HOLD"
                            segments = []
                            guidance = None
                            progress = None
                            navigation_status = hold_status
                            detail = (
                                f"follower_hold={hold_status!r}; "
                                "real_path_hidden=true"
                            )
                        elif transition is not None and not transition.visible:
                            self._guidance_stability.clear()
                            state = "REPLANNING"
                            segments = []
                            guidance = None
                            progress = None
                            navigation_status = transition.status
                            detail = (
                                f"{transition.detail}; old_path_held=false; "
                                "real_path_hidden=true"
                            )
                        if (
                            state in ("ACTIVE", "PATH OUT OF VIEW")
                            and self._waypoint_distance_enabled
                            and progress is not None
                        ):
                            waypoint_status = (
                                self._waypoint_progress_gate.label(
                                    self._ros_now_sec(),
                                    snapshot.generation,
                                    progress.remaining_path_m,
                                    progress.anchor_distance_m,
                                    self._guidance_settings
                                    .max_path_start_distance_m,
                                )
                                or ""
                            )
                        if (
                            state in ("ACTIVE", "PATH OUT OF VIEW")
                            and guidance is not None
                        ):
                            authorized = (
                                not self._guidance_require_follower
                                or self._waypoint_progress_gate.authorizes_plan(
                                    self._ros_now_sec(), snapshot.generation
                                )
                            )
                            if not authorized:
                                self._guidance_stability.clear()
                                guidance = None
                            else:
                                stable = self._guidance_stability.update(
                                    guidance, self._ros_now_sec()
                                )
                                guidance = stable.guidance
                                navigation_status = stable.status
                        else:
                            self._guidance_stability.clear()
                        if projection_stats is not None:
                            self._diagnostics.projection_update(
                                snapshot.frame_id,
                                projection_stats.log_detail(),
                                projection_stats.out_of_view_reason,
                            )
                        if state == "PATH OUT OF VIEW" and projection_stats is not None:
                            self._diagnostics.offscreen_guidance(
                                snapshot.frame_id, evaluation.guidance_detail
                            )
                            guidance_suffix = (
                                f"; guide={guidance.direction}"
                                if guidance is not None
                                else "; guide=none"
                            )
                            detail = (
                                f"frame={snapshot.frame_id} "
                                f"{projection_stats.out_of_view_reason}; "
                                f"{projection_stats.compact_detail()}"
                                f"{guidance_suffix}"
                            )
                        elif state == "TF UNAVAILABLE":
                            detail = (
                                f"source_frame={snapshot.frame_id!r}, "
                                f"target_frame={self._base_frame!r}"
                            )
                        elif (
                            state not in ("REPLANNING", "NAVIGATION HOLD")
                            and projection_stats is not None
                        ):
                            detail = (
                                f"gen={snapshot.generation} frame={snapshot.frame_id} "
                                f"{projection_stats.compact_detail()}"
                            )
                    except ValueError as exc:
                        state = "INVALID CONFIG"
                        detail = str(exc)
            elif state == "NO PATH":
                transition_gate = getattr(self, "_path_transition_gate", None)
                if transition_gate is not None:
                    transition_gate.clear()
                detail = self._diagnostics.no_path_detail()
            elif state == "PATH STALE" and snapshot is not None:
                transition_gate = getattr(self, "_path_transition_gate", None)
                if transition_gate is not None:
                    transition_gate.clear()
                age = max(0.0, self._path_store.age())
                detail = (
                    f"age={age:.3f}s timeout={self._path_store.timeout_sec:.3f}s "
                    f"generation={snapshot.generation}"
                )

            if reference_state == "ACTIVE" and reference_snapshot is not None:
                if localization is None or not localization.stable:
                    reference_reason = (
                        "localization_not_stable"
                        if localization is None
                        else localization.state
                    )
                elif not tf_available:
                    reference_reason = f"tf_unavailable {tf_detail}"
                else:
                    try:
                        reference_evaluation = evaluate_path_state(
                            reference_snapshot.points,
                            reference_snapshot.frame_id,
                            self._base_frame,
                            self._points_in_base,
                            self._mount,
                            self._intrinsics,
                            self._near,
                            display_geometry=self._path_display_geometry,
                            on_tf_error=lambda exc: self._diagnostics.tf_failure(
                                reference_snapshot.frame_id,
                                self._base_frame,
                                exc,
                            ),
                        )
                    except ValueError as exc:
                        reference_reason = f"invalid_projection {exc}"
                    else:
                        reference_segments = reference_evaluation.segments
                        reference_stats = reference_evaluation.stats
                        if reference_evaluation.state == "ACTIVE":
                            reference_reason = "visible"
                        elif reference_evaluation.state == "PATH OUT OF VIEW":
                            out_reason = (
                                "no_projectable_segment"
                                if reference_stats is None
                                else reference_stats.out_of_view_reason
                            )
                            reference_reason = f"out_of_view {out_reason}"
                        else:
                            reference_reason = reference_evaluation.state.lower()
                        stats_detail = (
                            "stats=unavailable"
                            if reference_stats is None
                            else reference_stats.log_detail()
                        )
                        self._diagnostics.reference_projection(
                            reference_snapshot.frame_id,
                            bool(reference_segments),
                            f"{stats_detail} projected_segments="
                            f"{len(reference_segments)}",
                            reference_reason,
                        )

            if costmap_state == "ACTIVE" and costmap_snapshot is not None:
                if localization is None or not localization.stable:
                    costmap_reason = (
                        "localization_not_stable"
                        if localization is None
                        else localization.state
                    )
                elif not tf_available:
                    costmap_reason = f"localization_tf_unavailable {tf_detail}"
                else:
                    try:
                        (
                            points_in_base,
                            costmap_analysis_points_base,
                        ) = self._costmap_arrays_in_base(
                            costmap_snapshot
                        )
                        costmap_projection = project_costmap_points(
                            points_in_base,
                            costmap_snapshot.classes,
                            self._mount,
                            self._intrinsics,
                            self._near,
                        )
                    except Exception as exc:  # tf2 exceptions vary by ROS distro
                        costmap_reason = (
                            f"costmap_tf_or_projection_unavailable "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        costmap_reason = (
                            "visible"
                            if costmap_projection.visible_points > 0
                            else "out_of_view"
                        )
                visible = (
                    0
                    if costmap_projection is None
                    else costmap_projection.visible_points
                )
                self._diagnostics.costmap_projection(
                    costmap_projection is not None,
                    costmap_snapshot.frame_id,
                    visible,
                    costmap_snapshot.sampled_cells,
                    costmap_reason,
                )

            explanation_enabled = bool(
                getattr(self, "_perception_explanation_enabled", False)
            )
            if explanation_enabled:
                if not costmap_enabled:
                    costmap_explanation_reason = "costmap_overlay_disabled"
                elif costmap_snapshot is None:
                    costmap_explanation_reason = costmap_reason
                elif costmap_projection is None:
                    costmap_explanation_reason = costmap_reason
                elif costmap_analysis_points_base is None:
                    costmap_explanation_reason = "costmap_tf_unavailable"
                elif path_points_base is None:
                    costmap_explanation_reason = "real_path_unavailable"
                elif localization is None or not localization.stable:
                    costmap_explanation_reason = "localization_not_stable"
                else:
                    decision = explain_costmap_and_path(
                        costmap_analysis_points_base,
                        costmap_snapshot.analysis_costs,
                        path_direction=(
                            None if guidance is None else guidance.direction
                        ),
                        viewer_state=state,
                        high_cost_min=self._costmap_settings.high_cost_min,
                        lethal_cost_min=self._costmap_settings.lethal_cost_min,
                        settings=self._costmap_explanation_settings,
                    )
                    costmap_explanation = decision.explanation
                    costmap_explanation_reason = decision.reason
                explanation_detail = (
                    ""
                    if costmap_explanation is None
                    else costmap_explanation.detail
                )
                self._diagnostics.costmap_explanation(
                    costmap_explanation is not None,
                    costmap_explanation_reason,
                    explanation_detail,
                )

            if reference_enabled:
                metadata = (
                    None if reference_gate is None else reference_gate.metadata
                )
                age = (
                    0.0 if reference_gate is None else max(0.0, reference_gate.age())
                )
                reference_summary = (
                    f"ref={reference_reason} points="
                    f"{0 if reference_snapshot is None else len(reference_snapshot.points)} "
                    f"length_m={0.0 if metadata is None else metadata.declared_length_m:.2f} "
                    f"age={age:.2f}s segments={len(reference_segments)}"
                )
                if not reference_segments:
                    reference_frame = (
                        self._localization.thresholds.expected_frame
                        if reference_snapshot is None
                        else reference_snapshot.frame_id
                    )
                    self._diagnostics.reference_projection(
                        reference_frame,
                        False,
                        reference_summary,
                        reference_reason,
                    )
                detail = f"{detail}; {reference_summary}" if detail else reference_summary
            if costmap_enabled:
                costmap_summary = (
                    f"costmap={costmap_reason} sampled="
                    f"{0 if costmap_snapshot is None else costmap_snapshot.sampled_cells} "
                    f"visible={0 if costmap_projection is None else costmap_projection.visible_points}"
                )
                detail = (
                    f"{detail}; {costmap_summary}"
                    if detail else costmap_summary
                )
            if explanation_enabled and self._debug_overlay:
                explanation_summary = (
                    "explanation="
                    f"{costmap_explanation_reason}"
                )
                detail = (
                    f"{detail}; {explanation_summary}"
                    if detail else explanation_summary
                )
            frame = render_frame(
                image,
                segments,
                state,
                self._color,
                self._thickness,
                detail=detail,
                guidance=guidance,
                guidance_color_bgr=self._guidance_color,
                reference_segments=reference_segments,
                reference_color_bgr=getattr(
                    self, "_reference_color", (255, 255, 0)
                ),
                reference_thickness=getattr(
                    self, "_reference_thickness", 3
                ),
                reference_dash_px=getattr(self, "_reference_dash_px", 14.0),
                reference_gap_px=getattr(self, "_reference_gap_px", 9.0),
                display_mode_label=" | ".join(filter(None, (
                    self._path_display_geometry.label,
                    self._calibration_profile_label,
                ))),
                waypoint_status=waypoint_status,
                navigation_status=navigation_status,
                debug_overlay=self._debug_overlay,
                path_alpha=getattr(self, "_path_alpha", 1.0),
                costmap_projection=costmap_projection,
                costmap_alpha=getattr(self, "_costmap_alpha", 0.42),
                costmap_radius_px=getattr(self, "_costmap_radius", 2),
                costmap_explanation=costmap_explanation,
            )
        self._diagnostics.state_changed(state, detail)
        if self._writer is not None:
            self._writer.write(frame)
        if self._display:
            try:
                key_code = self._display_window.show_and_poll(frame)
            except (AttributeError, RuntimeError, cv2.error) as exc:
                self._log_warning(
                    f"OpenCV display failed; recording continues: {exc}"
                )
                self._display = False
                if self._display_window is not None:
                    self._display_window.close()
            else:
                consume_actions = getattr(
                    self._display_window, "consume_actions", lambda: []
                )
                for action in consume_actions():
                    self._handle_display_action(action)
                if is_debug_toggle_key(key_code):
                    self._debug_overlay = not self._debug_overlay
                    mode = "DEBUG" if self._debug_overlay else "NORMAL"
                    self._log_info(f"navigation UI mode changed to {mode}")
                    if self._display_window is not None:
                        self._display_window.set_status(f"UI {mode}", True)
                if is_exit_key(key_code):
                    self._request_shutdown(
                        f"keyboard q/Q/F10 key_code={int(key_code)}"
                    )
        log_gui = getattr(self, "_log_gui", None)
        if log_gui is not None:
            log_gui.poll()

    def _request_shutdown(self, reason: str) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._log_info(f"AR Path Viewer shutdown requested: {reason}")
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
        if self._display_window is not None:
            self._display_window.close()
        log_gui = getattr(self, "_log_gui", None)
        if log_gui is not None:
            log_gui.close()
        session_log = getattr(self, "_session_log", None)
        if session_log is not None:
            session_log.close()

    def destroy_node(self):
        self._release_resources()
        return super().destroy_node()


def is_exit_key(key_code: int) -> bool:
    """Recognize q/Q and common Win32, GTK, and Qt F10 codes."""
    if key_code is None or key_code < 0:
        return False
    code = int(key_code) & 0xFFFFFFFF
    return code in {ord("q"), ord("Q"), 0x00790000, 0x0000FFC7, 0x01000039}


def is_debug_toggle_key(key_code: int) -> bool:
    """Use D/d for a backend-independent normal/debug overlay toggle."""
    if key_code is None or key_code < 0:
        return False
    return (int(key_code) & 0xFFFFFFFF) in {ord('d'), ord('D')}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArPathViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._log_info("AR Path Viewer shutdown requested: Ctrl+C")
    except ExternalShutdownException:
        # A sibling launch shutdown is a normal termination path.  Resources
        # are released by finally without reporting a false viewer crash.
        pass
    except Exception as exc:
        node._log_error(
            f"Unhandled exception: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        raise
    finally:
        node.destroy_node()
        shutdown_ros_context()


def shutdown_ros_context() -> None:
    """Idempotently shut down rclpy after node destruction."""
    rclpy.try_shutdown()
