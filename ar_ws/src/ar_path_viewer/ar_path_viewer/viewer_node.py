"""ROS 2 viewer node. This node creates subscriptions only, never publishers."""

import math
from dataclasses import replace
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMessage
from rcl_interfaces.msg import Log
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
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
    CostmapCameraSyncSettings,
    CostmapOverlaySettings,
    CostmapStore,
    CostmapTransformQueue,
    CostmapVisualDisplay,
    CostmapVisualHold,
    SynchronizedCostmap,
    align_costmap_to_image_time,
    estimate_image_frame_timing,
    project_costmap_points,
)
from .costmap_ready_history import CostmapReadyHistory
from .costmap_path_sync import (
    capture_costmap_path,
    check_costmap_path_pair,
    follower_goal_identity,
)
from .async_video_writer import AsyncPacedVideoWriter
from .costmap_alerts import (
    CostmapAlert,
    CostmapAlertGate,
    CostmapAlertHistoryGate,
    CostmapAlertSettings,
    evaluate_path_cost_alert,
    evaluate_forward_cost_alert,
    evaluate_robot_proximity_alert,
)
from .costmap_explanation import (
    CostmapExplanationSettings,
    explain_costmap_and_path,
)
from .costmap_minimap import CostmapMinimap
from .diagnostics import DiagnosticReporter
from .display_dispatch import LatestDisplayFrame
from .display_window import (
    ACTION_COPY_LOG,
    ACTION_TOGGLE_ROBOT_VIEW,
    ACTION_VIEW_LOG,
    DisplayWindow,
)
from .image_source import SimulationSource, make_image_source
from .guidance_stability import GuidanceStabilityGate
from .latest_image_source import LatestImageSource
from .ros_image_source import RosLatestImageSource
from .log_gui import LogGui
from .localization_monitor import (
    LocalizationMonitor,
    LocalizationThresholds,
    LocalizationWarningGate,
    amcl_message_to_observation,
)
from .offscreen_guidance import GuidanceSettings, OffscreenGuidance
from .rotation_motion import OdomRotationMotionGate, RotationMotionGate
from .odometry_distance import OdomTravelDistanceMeter
from .path_adapter import PathStore, path_message_to_array
from .path_transition import PathTransitionGate
from .perception_frame_delay import PerceptionFrameDelay
from .projection import (
    DEFAULT_CAMERA_Y_M,
    CameraIntrinsics,
    CameraMount,
    ground_plane_view,
)
from .reference_path import ReferencePathGate
from .route_destination import destination_from_reference_state
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
        self._main_thread_id = threading.get_ident()
        self._ui_state_lock = threading.Lock()
        self._robot_view_expanded = False
        self._display_frames = LatestDisplayFrame()
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
            # Review-only; keep exact-TF Costmap samples older than C920.
            "perception_costmap_camera_history_enabled": False,
            "perception_minimap_visual_hold_sec": 1.0,
            # Hide rarely visible floor dots only; keep Costmap sensing,
            # explanatory notices, and ROBOT VIEW independent.
            "perception_costmap_camera_overlay_enabled": False,
            "perception_costmap_topic": "/local_costmap/costmap",
            "perception_costmap_timeout_sec": 1.5,
            "perception_costmap_stamp_max_age_sec": 1.5,
            "perception_costmap_future_tolerance_sec": 0.5,
            "perception_costmap_tf_max_age_sec": 1.5,
            "perception_costmap_high_cost_min": 60,
            "perception_costmap_lethal_cost_min": 100,
            "perception_costmap_max_points_per_class": 450,
            "perception_costmap_max_grid_width_cells": 512,
            "perception_costmap_max_grid_height_cells": 512,
            "perception_costmap_max_grid_cells": 65536,
            "perception_costmap_plane_height_m": 0.02,
            "perception_costmap_tf_wait_timeout_sec": 1.5,
            "perception_costmap_max_pending_messages": 8,
            # OpenCV supplies no hardware exposure timestamp for the C920.
            # These conservative host-time and robot-motion limits therefore
            # gate all Costmap-derived camera overlays.
            "perception_camera_sync_max_age_sec": 0.40,
            "perception_camera_sync_max_capture_duration_sec": 0.12,
            "perception_camera_sync_max_frame_transport_age_sec": 0.12,
            "perception_camera_sync_max_translation_m": 0.20,
            "perception_camera_sync_max_rotation_deg": 15.0,
            "perception_camera_sync_clock_tolerance_sec": 0.02,
            "perception_camera_sync_future_tolerance_sec": 0.02,
            "perception_camera_sync_exact_tf_tolerance_sec": 0.005,
            # Default preserves the established exact-time delayed mode.
            # Review profiles may opt out to keep newest C920 video direct.
            "perception_camera_frame_delay_enabled": True,
            # Review-only: let the image-time TF arrive without blocking ROS.
            "perception_camera_frame_delay_sec": 0.12,
            "perception_camera_frame_max_age_sec": 0.30,
            "perception_costmap_alpha": 0.42,
            "perception_costmap_radius_px": 2,
            # Display-only continuation of a previously exact-time projection.
            # It never feeds alerts, explanations, or top-down analysis.
            "perception_costmap_visual_hold_sec": 0.30,
            "perception_minimap_enabled": False,
            "perception_minimap_size_px": 142,
            "perception_minimap_range_m": 3.0,
            "perception_alert_enabled": False,
            "perception_alert_path_corridor_width_m": 0.35,
            "perception_alert_nearby_lethal_distance_m": 0.30,
            "perception_alert_robot_half_length_m": 0.30,
            "perception_alert_robot_half_width_m": 0.29,
            # Longer verification: a frame of inflation is not a warning.
            "perception_alert_confirm_sec": 0.85,
            "perception_alert_forward_confirm_sec": 1.25,
            "perception_alert_forward_max_m": 1.50,
            "perception_alert_forward_half_width_m": 0.30,
            "perception_alert_forward_min_cost": 85,
            "perception_alert_forward_min_cells": 3,
            "perception_alert_release_sec": 0.40,
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
            "localization_warning_history_sec": 10.0,
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
            "rotation_guidance_enabled": False,
            "rotation_odom_topic": "/odom/filtered",
            "rotation_odom_frame": "robot/odom",
            "final_waypoint_follower_node_name": "koko_waypoint_follower",
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
            self._rotation_guidance_enabled = self._bool(
                "rotation_guidance_enabled"
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
            self._perception_costmap_visible = self._perception_costmap_enabled
            self._perception_costmap_camera_overlay_enabled = self._bool(
                "perception_costmap_camera_overlay_enabled"
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
                max_grid_width_cells=self._int(
                    "perception_costmap_max_grid_width_cells"
                ),
                max_grid_height_cells=self._int(
                    "perception_costmap_max_grid_height_cells"
                ),
                max_grid_cells=self._int("perception_costmap_max_grid_cells"),
                plane_height_m=self._float(
                    "perception_costmap_plane_height_m"
                ),
                tf_wait_timeout_sec=self._float(
                    "perception_costmap_tf_wait_timeout_sec"
                ),
                max_pending_messages=self._int(
                    "perception_costmap_max_pending_messages"
                ),
            )
            self._costmap_settings.validate()
            self._costmap_camera_sync_settings = CostmapCameraSyncSettings(
                max_costmap_age_sec=self._float(
                    "perception_camera_sync_max_age_sec"
                ),
                max_capture_duration_sec=self._float(
                    "perception_camera_sync_max_capture_duration_sec"
                ),
                max_frame_transport_age_sec=self._float(
                    "perception_camera_sync_max_frame_transport_age_sec"
                ),
                max_translation_m=self._float(
                    "perception_camera_sync_max_translation_m"
                ),
                max_rotation_deg=self._float(
                    "perception_camera_sync_max_rotation_deg"
                ),
                clock_tolerance_sec=self._float(
                    "perception_camera_sync_clock_tolerance_sec"
                ),
                future_tolerance_sec=self._float(
                    "perception_camera_sync_future_tolerance_sec"
                ),
                exact_tf_tolerance_sec=self._float(
                    "perception_camera_sync_exact_tf_tolerance_sec"
                ),
            )
            self._costmap_camera_sync_settings.validate()
            self._costmap_tf_max_age = self._float(
                "perception_costmap_tf_max_age_sec"
            )
            self._costmap_alpha = self._float("perception_costmap_alpha")
            self._costmap_radius = self._int("perception_costmap_radius_px")
            self._costmap_visual_hold_sec = self._float(
                "perception_costmap_visual_hold_sec"
            )
            self._perception_minimap_enabled = self._bool(
                "perception_minimap_enabled"
            )
            self._perception_minimap_visible = self._perception_minimap_enabled
            self._costmap_minimap_size = self._int(
                "perception_minimap_size_px"
            )
            self._costmap_minimap_range = self._float(
                "perception_minimap_range_m"
            )
            self._perception_alert_enabled = self._bool(
                "perception_alert_enabled"
            )
            self._perception_alert_visible = self._perception_alert_enabled
            self._costmap_alert_settings = CostmapAlertSettings(
                path_corridor_width_m=self._float(
                    "perception_alert_path_corridor_width_m"
                ),
                nearby_lethal_distance_m=self._float(
                    "perception_alert_nearby_lethal_distance_m"
                ),
                robot_half_length_m=self._float(
                    "perception_alert_robot_half_length_m"
                ),
                robot_half_width_m=self._float(
                    "perception_alert_robot_half_width_m"
                ),
                confirm_sec=self._float("perception_alert_confirm_sec"),
                release_sec=self._float("perception_alert_release_sec"),
                forward_max_m=self._float("perception_alert_forward_max_m"),
                forward_half_width_m=self._float(
                    "perception_alert_forward_half_width_m"
                ),
                forward_min_cost=self._int("perception_alert_forward_min_cost"),
                forward_min_cells=self._int("perception_alert_forward_min_cells"),
            )
            self._costmap_alert_settings.validate()
            self._costmap_alert_gate = CostmapAlertGate(
                self._costmap_alert_settings.confirm_sec,
                self._costmap_alert_settings.release_sec,
                clock=self._ros_now_sec,
                confirm_by_kind={
                    "forward": self._float("perception_alert_forward_confirm_sec"),
                    "high_path": self._float("perception_alert_forward_confirm_sec"),
                },
            )
            self._costmap_alert_history_gate = CostmapAlertHistoryGate(
                5.0, clock=self._ros_now_sec
            )
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
                or not math.isfinite(self._costmap_visual_hold_sec)
                or self._costmap_visual_hold_sec < 0.0
            ):
                raise ValueError(
                    "invalid costmap overlay alpha, radius, or visual hold"
                )
            if (
                self._costmap_minimap_size < 80
                or not math.isfinite(self._costmap_minimap_range)
                or self._costmap_minimap_range <= 0.0
            ):
                raise ValueError("invalid costmap minimap settings")
            self._perception_explanation_enabled = self._bool(
                "perception_explanation_enabled"
            )
            self._perception_explanation_visible = (
                self._perception_explanation_enabled
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
            self._rotation_guidance_enabled = False
            self._path_transition_gate = PathTransitionGate(0.0)
            self._path_alpha = 1.0
            self._perception_costmap_enabled = False
            self._perception_costmap_visible = False
            self._perception_costmap_camera_overlay_enabled = False
            self._costmap_settings = CostmapOverlaySettings()
            self._costmap_camera_sync_settings = CostmapCameraSyncSettings()
            self._costmap_tf_max_age = 1.5
            self._costmap_alpha = 0.42
            self._costmap_radius = 2
            self._costmap_visual_hold_sec = 0.0
            self._perception_minimap_enabled = False
            self._perception_minimap_visible = False
            self._costmap_minimap_size = 142
            self._costmap_minimap_range = 3.0
            self._perception_alert_enabled = False
            self._perception_alert_visible = False
            self._costmap_alert_settings = CostmapAlertSettings()
            self._costmap_alert_gate = CostmapAlertGate(0.0)
            self._costmap_alert_history_gate = CostmapAlertHistoryGate(2.5)
            self._perception_explanation_enabled = False
            self._perception_explanation_visible = False
            self._costmap_explanation_settings = CostmapExplanationSettings()

        self._rotation_motion_gate = RotationMotionGate()
        self._odom_rotation_gate = OdomRotationMotionGate()
        self._odom_distance_meter = OdomTravelDistanceMeter()
        self._last_reported_rotation = None
        self._last_verified_final_goal = None
        self._final_waypoint_success = None
        self._costmap_ready_history = (
            CostmapReadyHistory()
            if self._perception_costmap_enabled and
            self._bool("perception_costmap_camera_history_enabled")
            else None
        )
        self._costmap_store = CostmapStore(self._costmap_settings)
        self._costmap_transform_queue = CostmapTransformQueue(
            self._costmap_settings
        )
        self._costmap_visual_hold = CostmapVisualHold(
            self._costmap_visual_hold_sec
        )
        self._last_costmap_visual_context = ""
        self._last_image_frame_timing = None
        self._minimap_visual_hold_sec = (
            self._float("perception_minimap_visual_hold_sec")
        )
        if (not math.isfinite(self._minimap_visual_hold_sec)
                or not 0.0 <= self._minimap_visual_hold_sec <= 2.0):
            self._config_error = (
                "invalid perception_minimap_visual_hold_sec"
            )
        self._perception_frame_delay = None
        # Camera-first review: never postpone the live C920 preview.
        # With delay disabled, Costmap must still pass the existing
        # exact-time, motion and freshness checks before use.
        if (self._perception_costmap_enabled and
                self._bool("perception_camera_frame_delay_enabled") and
                self._str("image_input_type").strip().lower() ==
                "camera_timestamped"):
            self._perception_frame_delay = PerceptionFrameDelay(
                delay_sec=self._float("perception_camera_frame_delay_sec"),
                max_age_sec=self._float("perception_camera_frame_max_age_sec"),
            )
        self._path_updated_until = 0.0

        try:
            self._image_input_kind = self._str("image_input_type").strip().lower()
            if self._image_input_kind == "ros_image":
                self._image_source = RosLatestImageSource(
                    self, self._str("image_input_uri"),
                    self._intrinsics.width, self._intrinsics.height,
                )
            else:
                self._image_source = make_image_source(
                    self._image_input_kind, self._str("image_input_uri"),
                    self._intrinsics.width, self._intrinsics.height,
                    fps=self._float("fps"),
                )
            if self._image_input_kind == "camera_timestamped":
                self._image_source = LatestImageSource(self._image_source)
        except OSError as exc:
            self._camera_error = str(exc)
            self._log_error(f"Camera unavailable: {exc}")
            self._image_input_kind = "unavailable"
            self._image_source = SimulationSource(
                self._intrinsics.width, self._intrinsics.height
            )
        except ValueError as exc:
            self._config_error = str(exc)
            self._log_error(f"Invalid image input configuration: {exc}")
            self._image_input_kind = "unavailable"
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
        capture_rate_reason = str(getattr(
            self._image_source, 'capture_rate_reason', 'unverified'
        ))
        rate_message = (
            "camera_rate_configuration "
            f"viewer_target_fps={self._float('fps'):.3f} "
            f"input_type={self._image_input_kind!r} "
            "requested_capture_fps="
            f"{getattr(self._image_source, 'requested_capture_fps', None)} "
            "effective_capture_fps="
            f"{getattr(self._image_source, 'effective_capture_fps', None)} "
            "capture_rate_reason="
            f"{capture_rate_reason} "
            "display_api_timing_not_monitor_scanout=true"
        )
        if (
            self._image_input_kind == 'camera_timestamped'
            and capture_rate_reason != 'matched'
        ):
            self._log_warning(rate_message)
        else:
            self._log_info(rate_message)
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
            self._localization_warning_gate = LocalizationWarningGate(
                self._float("localization_warning_history_sec")
            )
        except ValueError as exc:
            self._config_error = self._config_error or str(exc)
            self._log_error(f"Invalid localization configuration: {exc}")
            self._localization = LocalizationMonitor()
            self._localization_warning_gate = LocalizationWarningGate()
        self._last_localization_warning_key = None
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
        self._target_fps = self._float("fps")
        self._display_period_sec = 1.0 / max(1.0, self._target_fps)
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
        # The 30 fps rendering timer is intentionally kept in the default
        # callback group.  Real/reference Path updates and Follower heartbeats
        # use one separate serial group: their ordering is preserved, while a
        # long render cannot delay receipt past the 2.5 s safety limit.  The
        # stores/gates read concurrently by the render callback are locked.
        self._route_callback_group = MutuallyExclusiveCallbackGroup()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            PathMessage, self._str("path_topic"), self._on_path, qos,
            callback_group=self._route_callback_group,
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
                callback_group=self._route_callback_group,
            )
        reference_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Both subscriptions are read-only. Follower /rosout is evidence
        # of the Follower's own arrival threshold, NOT route completion.
        self._odom_rotation_subscription = self.create_subscription(
            Odometry, self._str('rotation_odom_topic'),
            self._on_rotation_odom, 10,
        )
        self._follower_log_subscription = self.create_subscription(
            Log, '/rosout', self._on_follower_log, 100,
        )
        self._reference_follower_subscription = None
        self._route_destination = None
        if self._reference_enabled or self._waypoint_distance_enabled:
            self._reference_follower_subscription = self.create_subscription(
                WaypointFollowerState,
                self._str("reference_follower_state_topic"),
                self._on_reference_follower_state,
                reference_state_qos,
                callback_group=self._route_callback_group,
            )
        # The robot AMCL publisher is RELIABLE + TRANSIENT_LOCAL.
        # Request its latest retained sample on Viewer startup; the monitor
        # still rejects latched samples with stale header timestamps.
        self._amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self._str("amcl_pose_topic"),
            self._on_amcl_pose,
            reference_state_qos,
        )
        output_video = "" if self._session_paths.video is None else str(
            self._session_paths.video
        )
        self._writer = self._make_writer(output_video, self._target_fps)
        self._timer = self.create_timer(self._display_period_sec, self._tick)
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
            "camera_preview_policy "
            f"delay_enabled={self._perception_frame_delay is not None} "
            f"costmap_enabled={self._perception_costmap_enabled} "
            "camera_source_timestamps_unchanged=true"
        )
        if self._perception_frame_delay is not None:
            self._log_info(
                "perception_frame_delay_configuration "
                f"enabled=true delay_sec="
                f"{self._perception_frame_delay.delay_sec:.3f} "
                f"max_age_sec={self._perception_frame_delay.max_age_sec:.3f} "
                "read_only=true"
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
            f"tf_wait_timeout_sec="
            f"{self._costmap_settings.tf_wait_timeout_sec:.3f} "
            f"max_pending={self._costmap_settings.max_pending_messages} "
            "semantic_labels='unknown,high_cost,lethal_cost' "
            "does_not_claim_camera_object_or_danger=true"
        )
        self._log_info(
            "perception_explanation_configuration "
            f"enabled={self._perception_explanation_enabled} "
            "inputs='local_costmap,real_plan' "
            "causal_planner_claim=false danger_label=false"
        )
        self._log_info(
            "perception_review_ui_configuration "
            f"minimap_enabled={self._perception_minimap_enabled} "
            f"alert_enabled={self._perception_alert_enabled} "
            "all_layers_opt_in=true exact_timestamp_tf=true"
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
        return AsyncPacedVideoWriter(writer, fps)

    def _handle_display_action(self, action: str) -> None:
        ArPathViewerNode._require_main_thread(self)
        if action == ACTION_TOGGLE_ROBOT_VIEW:
            configured = bool(getattr(self, "_perception_minimap_enabled", False))
            if not configured:
                if self._display_window is not None:
                    self._display_window.set_status("ROBOT VIEW not configured", False)
                return
            lock = getattr(self, "_ui_state_lock", None)
            if lock is None:
                self._robot_view_expanded = not bool(getattr(
                    self, "_robot_view_expanded", False
                ))
                expanded = self._robot_view_expanded
            else:
                with lock:
                    self._robot_view_expanded = not bool(getattr(
                        self, "_robot_view_expanded", False
                    ))
                    expanded = self._robot_view_expanded
            self._log_info(f"robot_view expanded={expanded}")
            if self._display_window is not None:
                self._display_window.set_status(
                    f"ROBOT VIEW {'OPEN' if expanded else 'CLOSED'}", True
                )
            return
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

    def _require_main_thread(self) -> None:
        owner = getattr(self, "_main_thread_id", threading.get_ident())
        if threading.get_ident() != owner:
            raise RuntimeError("Viewer GUI operation must run on the main thread")

    def _ui_state_snapshot(self) -> dict[str, bool]:
        defaults = {
            "_display": False,
            "_debug_overlay": False,
            "_perception_costmap_visible": True,
            "_perception_explanation_visible": True,
            "_perception_minimap_visible": True,
            "_perception_alert_visible": True,
            "_robot_view_expanded": False,
        }
        lock = getattr(self, "_ui_state_lock", None)
        if lock is None:
            return {
                name: bool(getattr(self, name, default))
                for name, default in defaults.items()
            }
        with lock:
            return {
                name: bool(getattr(self, name, default))
                for name, default in defaults.items()
            }

    def _publish_display_frame(
        self, frame: np.ndarray,
        source_timestamp_monotonic_sec: float | None = None,
    ) -> None:
        if not ArPathViewerNode._ui_state_snapshot(self)["_display"]:
            return
        exchange = getattr(self, "_display_frames", None)
        if exchange is not None:
            exchange.publish(frame, source_timestamp_monotonic_sec)

    def _poll_log_gui(self) -> None:
        ArPathViewerNode._require_main_thread(self)
        log_gui = getattr(self, "_log_gui", None)
        if log_gui is not None:
            log_gui.poll()

    def _process_display_frame(self, frame: np.ndarray) -> bool:
        """Run every OpenCV/Tk GUI operation on the creating main thread."""
        ArPathViewerNode._require_main_thread(self)
        with getattr(self, "_ui_state_lock", threading.Lock()):
            display_enabled = bool(getattr(self, "_display", False))
            display_window = getattr(self, "_display_window", None)
        if not display_enabled or display_window is None:
            ArPathViewerNode._poll_log_gui(self)
            return False
        try:
            key_code = display_window.show_and_poll(frame)
        except (AttributeError, RuntimeError, cv2.error) as exc:
            self._log_warning(
                f"OpenCV display failed; recording continues: {exc}"
            )
            try:
                display_window.close()
            finally:
                lock = getattr(self, "_ui_state_lock", None)
                if lock is None:
                    self._display = False
                    self._display_window = None
                else:
                    with lock:
                        self._display = False
                        self._display_window = None
            ArPathViewerNode._poll_log_gui(self)
            return False

        consume_actions = getattr(
            display_window, "consume_actions", lambda: []
        )
        for action in consume_actions():
            handler = getattr(self, "_handle_display_action", None)
            if handler is None:
                ArPathViewerNode._handle_display_action(self, action)
            else:
                handler(action)
        if is_debug_toggle_key(key_code):
            lock = getattr(self, "_ui_state_lock", None)
            if lock is None:
                self._debug_overlay = not self._debug_overlay
                debug_overlay = self._debug_overlay
            else:
                with lock:
                    self._debug_overlay = not self._debug_overlay
                    debug_overlay = self._debug_overlay
            mode = "DEBUG" if debug_overlay else "NORMAL"
            self._log_info(f"navigation UI mode changed to {mode}")
            display_window.set_status(f"UI {mode}", True)
        perception_toggle = perception_toggle_key(key_code)
        if perception_toggle is not None:
            attribute, label = perception_toggle
            configured = bool(getattr(
                self, attribute.replace("_visible", "_enabled"), False
            ))
            if configured:
                lock = getattr(self, "_ui_state_lock", None)
                if lock is None:
                    visible = not bool(getattr(self, attribute, True))
                    setattr(self, attribute, visible)
                else:
                    with lock:
                        visible = not bool(getattr(self, attribute, True))
                        setattr(self, attribute, visible)
                self._log_info(
                    f"perception UI layer={label} visible={visible}"
                )
                display_window.set_status(
                    f"{label} {'ON' if visible else 'OFF'}", True
                )
            else:
                display_window.set_status(f"{label} not configured", False)
        if is_exit_key(key_code):
            self._request_shutdown(
                f"keyboard q/Q/F10 key_code={int(key_code)}"
            )
        ArPathViewerNode._poll_log_gui(self)
        return True

    def run_main_thread_ui(self, executor_done: threading.Event) -> None:
        """Pump HighGUI on the main thread while ROS callbacks run elsewhere."""
        ArPathViewerNode._require_main_thread(self)
        sequence = 0
        last_frame = np.zeros(
            (self._intrinsics.height, self._intrinsics.width, 3), np.uint8
        )
        last_metadata = None
        period = float(getattr(self, "_display_period_sec", 0.05))
        exchange = getattr(self, "_display_frames", None)
        while (
            not self._shutdown_requested
            and not executor_done.is_set()
            and rclpy.ok(context=self.context)
        ):
            if exchange is not None:
                next_sequence, frame, metadata = (
                    exchange.wait_for_new_with_metadata(sequence, period)
                )
                new_frame = frame is not None
                sequence = next_sequence
                if frame is not None:
                    last_frame = frame
                    last_metadata = metadata
            else:
                executor_done.wait(period)
                new_frame = False
            if ArPathViewerNode._ui_state_snapshot(self)["_display"]:
                shown = ArPathViewerNode._process_display_frame(
                    self, last_frame
                )
                if shown:
                    self._diagnostics.camera_frame_presented(
                        new_frame=new_frame,
                        source_timestamp_monotonic_sec=(
                            None if last_metadata is None else
                            last_metadata.source_timestamp_monotonic_sec
                        ),
                        target_fps=getattr(self, "_target_fps", 20.0),
                    )
            else:
                ArPathViewerNode._poll_log_gui(self)

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
            # An absent/zero header stamp is acceptable for route rendering,
            # but Costmap-versus-Path alerts will then fail closed.
            valid_stamp = (
                header_stamp_sec if header_stamp_sec is not None
                and math.isfinite(header_stamp_sec)
                and header_stamp_sec > 0.0 else None
            )
            snapshot = self._path_store.replace(
                points, frame, header_stamp_sec=valid_stamp,
                received_ros_sec=self._ros_now_sec(),
            )
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
        transform_queue = getattr(self, "_costmap_transform_queue", None)
        if transform_queue is None:
            # Test harnesses and older composed construction paths may create
            # the store directly.  Preserve the same store clock and safety
            # settings rather than bypassing exact-time synchronization.
            transform_queue = CostmapTransformQueue(
                self._costmap_settings,
                clock=self._costmap_store.clock,
            )
            self._costmap_transform_queue = transform_queue
        previous_snapshot = self._costmap_store.snapshot
        accepted = self._costmap_store.update(message, self._ros_now_sec())
        queued = False
        snapshot = self._costmap_store.snapshot
        detail = self._costmap_store.reason
        if accepted and snapshot is not None:
            if (
                previous_snapshot is not None
                and previous_snapshot.frame_id != snapshot.frame_id
            ):
                visual_hold = getattr(self, "_costmap_visual_hold", None)
                if visual_hold is not None:
                    visual_hold.clear()
                self._last_costmap_visual_context = ""
            queued = transform_queue.submit(snapshot)
            if queued:
                ready_history = getattr(
                    self, "_costmap_ready_history", None
                )
                if ready_history is not None:
                    ready_history.invalidate_if_frame_changed(
                        snapshot.frame_id
                    )
                # Freeze the real /plan epoch at Costmap *receipt*, not at
                # late exact-time TF resolution. Keep separate bindings for
                # old ready and new pending Costmaps.
                bindings = getattr(self, "_costmap_path_bindings", None)
                if bindings is None:
                    bindings = {}
                    self._costmap_path_bindings = bindings
                key = (snapshot.frame_id, snapshot.stamp_sec)
                bindings[key] = capture_costmap_path(
                    snapshot.stamp_sec, snapshot.frame_id,
                    self._path_store.snapshot(),
                    follower_goal_identity(
                        getattr(self, "_waypoint_progress_gate", None),
                        self._ros_now_sec(),
                    ),
                )
                # Costmap windows are <2s. Limit long-running bookkeeping.
                while len(bindings) > 32:
                    bindings.pop(next(iter(bindings)))
                detail = (
                    f"frame={snapshot.frame_id!r} "
                    f"unknown_cells={snapshot.unknown_cells} "
                    f"high_cost_cells={snapshot.high_cost_cells} "
                    f"lethal_cost_cells={snapshot.lethal_cost_cells} "
                    f"sampled_cells={snapshot.sampled_cells}"
                )
            else:
                detail = transform_queue.reason
        elif detail.startswith("out_of_order_local_costmap_ignored"):
            # Preserve the newer store/ready pair. This old callback cannot
            # revive old geometry or invalidate fresh synchronized data.
            pass
        else:
            transform_queue.clear(detail)
            ready_history = getattr(self, "_costmap_ready_history", None)
            if ready_history is not None:
                ready_history.clear()
            getattr(self, "_costmap_path_bindings", {}).clear()
            visual_hold = getattr(self, "_costmap_visual_hold", None)
            if visual_hold is not None:
                visual_hold.clear()
            self._last_costmap_visual_context = ""
            alert_gate = getattr(self, "_costmap_alert_gate", None)
            if alert_gate is not None:
                alert_gate.clear()
        self._diagnostics.costmap_update(accepted and queued, detail)

    def _on_rotation_odom(self, message: Odometry) -> None:
        # Filtered odometry reports measured motion, never motor commands.
        now = self._ros_now_sec()
        stamp = (float(message.header.stamp.sec)
                 + float(message.header.stamp.nanosec) * 1e-9)
        if (stamp <= 0.0 or abs(now - stamp) > 0.75
                or str(message.header.frame_id)
                != self._str('rotation_odom_frame')):
            self._odom_rotation_gate.clear()
            return
        meter = getattr(self, '_odom_distance_meter', None)
        if meter is not None:
            # Only valid /odom/filtered timestamp and frame enter this meter.
            meter.observe(message.pose.pose.position.x,
                          message.pose.pose.position.y, stamp)
        direction = self._odom_rotation_gate.update(
            message.twist.twist.linear.x,
            message.twist.twist.angular.z,
            now,
        )
        if direction != self._last_reported_rotation:
            self._last_reported_rotation = direction
            if direction is not None:
                self._log_info(
                    f'measured_in_place_rotation direction={direction!r}'
                )

    def _on_follower_log(self, message: Log) -> None:
        """Only the real Follower threshold event can confirm final arrival."""
        verified = getattr(self, '_last_verified_final_goal', None)
        if verified is None:
            return
        now = self._ros_now_sec()
        _route, _generation, wp_number, wp_count, verified_at = verified
        if not 0.0 <= now - verified_at <= 7.0:
            return
        stamp = (float(message.stamp.sec)
                 + float(message.stamp.nanosec) * 1e-9)
        from .final_waypoint_log import verified_final_arrival
        distance = verified_final_arrival(
            message.name, message.msg, now - stamp,
            wp_number, wp_count,
            expected_node_name=self._str('final_waypoint_follower_node_name'),
        )
        if distance is None:
            return
        if getattr(self, '_final_waypoint_success', None) is not None:
            return
        self._final_waypoint_success = (now, wp_number)
        self._log_info(
            f'final_waypoint_threshold_verified waypoint={wp_number} '
            f'distance_m={distance:.2f} route_complete_claim=false'
        )
        meter = getattr(self, '_odom_distance_meter', None)
        if meter is not None and hasattr(meter, "diagnostics"):
            evidence = meter.diagnostics()
            self._log_info(
                'odom_distance_audit '
                f"observed_m={evidence['observed_m']:.3f} "
                f"accepted_steps={evidence['accepted_steps']} "
                f"rejected_gap={evidence['rejected_gap']} "
                f"rejected_jump={evidence['rejected_jump']} "
                f"rejected_bad_time={evidence['rejected_bad_time']} "
                f"stale_follower_events={evidence['stale_follower_events']}"
            )

    def _on_reference_follower_state(
        self, message: WaypointFollowerState
    ) -> None:
        meter = getattr(self, '_odom_distance_meter', None)
        if meter is not None:
            try:
                stamp = (float(message.header.stamp.sec)
                         + 1e-9 * float(message.header.stamp.nanosec))
                age = self._ros_now_sec() - stamp
                if (int(message.schema_version) == 1
                        and str(message.header.frame_id)
                        == self._str('localization_global_frame')
                        and 0.0 <= age <= 2.5):
                    meter.set_follower(str(message.route_sha256),
                                       int(message.waypoint_count),
                                       int(message.activity))
            except (AttributeError, TypeError, ValueError):
                pass  # Unverified state must never advance the meter.
        snapshot = self._path_store.snapshot()
        path_generation = 0 if snapshot is None else snapshot.generation
        progress_accepted = self._waypoint_progress_gate.update(
            message, self._ros_now_sec(), path_generation
        )
        if progress_accepted:
            destination = getattr(self, '_route_destination', None)
            if destination is not None and (
                destination.route_sha256 != str(message.route_sha256)
                or destination.waypoint_count != int(message.waypoint_count)
            ):
                self._route_destination = None
            if (str(message.navigation_mode) == 'to_pose'
                    and int(message.next_waypoint_index)
                    == int(message.waypoint_count) - 1):
                self._last_verified_final_goal = (
                    str(message.route_sha256),
                    int(message.goal_generation),
                    int(message.next_waypoint_number),
                    int(message.waypoint_count),
                    self._ros_now_sec(),
                )
            else:
                self._last_verified_final_goal = None
                self._final_waypoint_success = None
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
        try:
            destination = destination_from_reference_state(message)
        except (AttributeError, TypeError, ValueError):
            destination = None
        current_view = self._waypoint_progress_gate.current_view
        if (
            destination is not None
            and current_view is not None
            and destination.route_sha256 == current_view.route_sha256
            and destination.waypoint_count == current_view.waypoint_count
        ):
            previous = getattr(self, '_route_destination', None)
            self._route_destination = destination
            if previous != destination:
                self._log_info(
                    'final_waypoint_direct verified=true '
                    f'route={destination.route_sha256[:12]} '
                    f'waypoint={destination.waypoint_number} '
                    'meaning=map_plane_straight_line_not_nav2_path'
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
        # Do not block the 30 fps rendering timer for an optional TF.
        # If TF is unavailable, existing fail-closed route gating hides it.
        transform = self._tf_buffer.lookup_transform(
            self._base_frame, frame_id, Time(), timeout=Duration(seconds=0.0)
        )
        return transform_message_points(points, transform)

    def _points_in_base_at(
        self, points: np.ndarray, frame_id: str, stamp_sec: float
    ) -> np.ndarray:
        """Transform geometry at an explicit sensor timestamp, never latest."""
        if frame_id == self._base_frame:
            return np.asarray(points, dtype=np.float64).copy()
        query_time = Time(nanoseconds=int(round(float(stamp_sec) * 1e9)))
        transform = self._tf_buffer.lookup_transform(
            self._base_frame,
            frame_id,
            query_time,
            timeout=Duration(seconds=0.0),
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
            timeout=Duration(seconds=0.0),
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

    def _costmap_arrays_at_image_time(
        self, snapshot, image_timing, *, max_costmap_age_sec=None
    ):
        """Re-express Costmap cells in base at this image's host timestamp.

        The exact Costmap-time transform remains the admission prerequisite.
        A second exact lookup at the image timestamp prevents those old
        base-frame coordinates from being drawn on a later camera pose.
        """
        if snapshot.frame_id == self._base_frame:
            # Base-frame cells have no fixed-frame anchor with which to undo
            # intervening robot motion.  Refuse rather than assume stationary.
            raise ValueError("costmap_base_frame_cannot_be_time_reprojected")
        costmap_query = Time(
            nanoseconds=int(round(float(snapshot.stamp_sec) * 1e9))
        )
        image_query = Time(
            nanoseconds=int(round(float(image_timing.stamp_sec) * 1e9))
        )
        transform_at_costmap = self._tf_buffer.lookup_transform(
            self._base_frame,
            snapshot.frame_id,
            costmap_query,
            timeout=Duration(seconds=0.0),
        )
        transform_at_image = self._tf_buffer.lookup_transform(
            self._base_frame,
            snapshot.frame_id,
            image_query,
            timeout=Duration(seconds=0.0),
        )
        settings = self._costmap_camera_sync_settings
        if max_costmap_age_sec is not None:
            settings = replace(
                settings, max_costmap_age_sec=float(max_costmap_age_sec)
            )
        return align_costmap_to_image_time(
            snapshot,
            image_timing=image_timing,
            transform_at_costmap=transform_at_costmap,
            transform_at_image=transform_at_image,
            settings=settings,
        )

    def _check_localization_tf(
        self,
    ) -> tuple[bool, str, tuple[float, float, float] | None]:
        global_frame = self._localization.thresholds.expected_frame
        if global_frame == self._base_frame:
            return True, "", None
        try:
            # Never block the ROS render timer for an optional TF sample.
            # A missing transform fails closed and will be checked next tick.
            transform = self._tf_buffer.lookup_transform(
                global_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.0),
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

    def _localization_warning_for_frame(
        self,
        status,
        *,
        tf_checked: bool,
        tf_available: bool,
        tf_detail: str,
    ):
        gate = getattr(self, "_localization_warning_gate", None)
        if gate is None:
            gate = LocalizationWarningGate()
            self._localization_warning_gate = gate
        warning = gate.update(
            status,
            tf_checked=tf_checked,
            tf_available=tf_available,
            tf_detail=tf_detail,
        )
        key = None if warning is None else (warning.mode, warning.reason)
        previous = getattr(self, "_last_localization_warning_key", None)
        if key != previous:
            if warning is None:
                self._log_info("localization_warning cleared")
            elif warning.active:
                self._log_warning(
                    "localization_warning active=true "
                    f"reason={warning.reason!r} detail={warning.detail!r}"
                )
            else:
                self._log_info(
                    "localization_warning active=false history=true "
                    f"reason={warning.reason!r}"
                )
            self._last_localization_warning_key = key
        # Display only severe, sustained warnings. Keep ALL actual
        # localization/TF/Costmap validity gates and diagnostics unchanged.
        # Routine recovery history and short AMCL timing gaps are not banners.
        display_warning = None
        if warning is not None and warning.active:
            now_mono = time.monotonic()
            since = getattr(self, '_localization_ui_bad_since', None)
            if since is None:
                since = now_mono
                self._localization_ui_bad_since = since
            if now_mono - since >= 5.0:
                display_warning = warning
        else:
            self._localization_ui_bad_since = None
        self._last_localization_warning_for_display = display_warning
        self._last_localization_warning_checked_monotonic = time.monotonic()
        return display_warning

    def _display_minimap_snapshot(
        self, current, *, now: float, allow_recent: bool
    ):
        """Keep a short, explicitly historical top-down snapshot for UI only.

        No history is ever passed to semantic alerts, motion, TF, or paths.
        """
        if current is not None:
            self._last_minimap_snapshot_ui = (float(now), current)
            return current, ""
        if not allow_recent:
            self._last_minimap_snapshot_ui = None
            return None, ""
        cached = getattr(self, "_last_minimap_snapshot_ui", None)
        if cached is None:
            return None, ""
        age = float(now) - float(cached[0])
        hold_sec = getattr(self, "_minimap_visual_hold_sec", 1.0)
        if not math.isfinite(age) or not 0.0 <= age <= hold_sec:
            self._last_minimap_snapshot_ui = None
            return None, ""
        return cached[1], f"LAST SNAPSHOT {age:.1f}s"

    def _recent_alert_history_for_ui(self):
        """Render history in preview without calling it a live warning."""
        history_gate = getattr(self, "_costmap_alert_history_gate", None)
        if history_gate is None:
            return None
        return history_gate.update(
            None,
            evaluation_status="unconfirmed",
            goal_identity=follower_goal_identity(
                getattr(self, "_waypoint_progress_gate", None),
                self._ros_now_sec(),
            ),
        )

    def _tick(self) -> None:
        if self._shutdown_requested:
            return
        # GUI keyboard actions are written by the main thread.  Use one
        # coherent snapshot for the entire rendered frame instead of reading
        # those booleans repeatedly while the main thread can change them.
        ui_state = ArPathViewerNode._ui_state_snapshot(self)
        debug_overlay = ui_state["_debug_overlay"]
        robot_view_enabled = bool(
            getattr(self, "_perception_minimap_enabled", False)
            and ui_state["_perception_minimap_visible"]
        )
        robot_view_expanded = ui_state["_robot_view_expanded"]
        capture_started_monotonic = time.monotonic()
        previous_tick_started = getattr(self, "_last_tick_started_monotonic", None)
        tick_interval_sec = (
            None if previous_tick_started is None
            else capture_started_monotonic - float(previous_tick_started)
        )
        self._last_tick_started_monotonic = capture_started_monotonic
        capture_started_ros = self._ros_now_sec()
        captured_image = self._image_source.read()
        source_read_metrics = getattr(
            self._image_source, "last_read_metrics", None
        )
        # The timestamped camera is captured continuously by a bounded worker.
        # A timer firing between two physical frames is not a disconnect and
        # must not replay pixels with a fabricated new timestamp.  Keep the
        # already-presented GUI frame while TF and other ROS callbacks run.
        if (
            captured_image is None
            and bool(getattr(self._image_source, "temporary_empty", False))
        ):
            return
        capture_finished_ros = self._ros_now_sec()
        capture_finished_monotonic = time.monotonic()
        previous_timing = getattr(self, "_last_image_frame_timing", None)
        image_timing = estimate_image_frame_timing(
            ros_started_sec=capture_started_ros,
            ros_finished_sec=capture_finished_ros,
            monotonic_started_sec=capture_started_monotonic,
            monotonic_finished_sec=capture_finished_monotonic,
            source_kind=(
                "camera_timestamped"
                if getattr(self, "_image_input_kind", "") == "ros_image"
                and getattr(self._image_source, "frame_timestamp_kind", "")
                == "v4l2_monotonic_soe"
                else getattr(self, "_image_input_kind", "simulation")
            ),
            settings=getattr(
                self,
                "_costmap_camera_sync_settings",
                CostmapCameraSyncSettings(),
            ),
            previous_stamp_sec=(
                None if previous_timing is None else previous_timing.stamp_sec
            ),
            previous_timing=previous_timing,
            source_timestamp_monotonic_sec=getattr(
                self._image_source, "frame_timestamp_monotonic", None
            ),
            timestamp_source=getattr(
                self._image_source, "frame_timestamp_kind", ""
            ),
        )
        captured_image_timing = image_timing
        self._last_image_frame_timing = image_timing
        # The newest successfully acquired image has a separate role from the
        # delayed image used for exact-time AR.  A missing/untrusted exposure
        # timestamp must suppress overlays, never replace valid C920 pixels
        # with a synthetic black frame.
        preview_image = captured_image
        if preview_image is not None:
            source_shape = (
                self._source_intrinsics.height,
                self._source_intrinsics.width,
            )
            if preview_image.shape[:2] != source_shape:
                preview_image = cv2.resize(
                    preview_image,
                    (
                        self._source_intrinsics.width,
                        self._source_intrinsics.height,
                    ),
                )
            if self._undistorter is not None:
                preview_image = self._undistorter.apply(preview_image)
        image = preview_image
        display_image_timing = image_timing
        waiting_for_delayed_frame = False
        frame_sync_wait_reason = ""
        delayed_frame = None
        retry_delayed_frame = False
        camera_pipeline_mode = "direct"
        frame_delay = getattr(self, "_perception_frame_delay", None)
        if frame_delay is not None:
            camera_pipeline_mode = "preview_sync_wait"
            if preview_image is None:
                # Do not replay a buffered old frame when the current camera
                # acquisition failed.  CAMERA UNAVAILABLE is explicit.
                frame_delay.clear()
            elif not image_timing.usable:
                # An invalid acquired timestamp starts a new unsafe epoch.
                frame_delay.clear()
                waiting_for_delayed_frame = True
                frame_sync_wait_reason = image_timing.reason
            else:
                frame_delay.add(preview_image, image_timing)
            if preview_image is not None and image_timing.usable:
                delayed_frame = frame_delay.peek_latest_eligible(time.monotonic())
                if delayed_frame is not None:
                    image = delayed_frame.image
                    image_timing = delayed_frame.timing
                    display_image_timing = image_timing
                    camera_pipeline_mode = "synchronized"
                else:
                    waiting_for_delayed_frame = True
                    frame_sync_wait_reason = "image_time_tf_buffering"
        state = "NO PATH"
        detail = ""
        guidance = None
        waypoint_status = ""
        final_waypoint_status = ""
        navigation_status = ""
        base_localization_status = self._localization.status()
        localization_warning = None
        render_started_monotonic = time.monotonic()
        render_duration_sec = 0.0
        if image is None:
            image = np.zeros((self._intrinsics.height, self._intrinsics.width, 3), np.uint8)
            state = "CAMERA UNAVAILABLE"
            detail = "image source returned no frame"
            localization_warning = ArPathViewerNode._localization_warning_for_frame(
                self,
                base_localization_status,
                tf_checked=False,
                tf_available=False,
                tf_detail="",
            )
            frame = render_frame(
                image, [], state, debug_overlay=debug_overlay,
                localization_warning=localization_warning,
                robot_view_enabled=robot_view_enabled,
                robot_view_expanded=robot_view_expanded,
            )
            render_duration_sec = time.monotonic() - render_started_monotonic
        elif self._camera_error:
            state = "CAMERA UNAVAILABLE"
            detail = self._camera_error
            localization_warning = ArPathViewerNode._localization_warning_for_frame(
                self,
                base_localization_status,
                tf_checked=False,
                tf_available=False,
                tf_detail="",
            )
            render_started_monotonic = time.monotonic()
            frame = render_frame(
                image, [], state, detail=detail,
                debug_overlay=debug_overlay,
                localization_warning=localization_warning,
                robot_view_enabled=robot_view_enabled,
                robot_view_expanded=robot_view_expanded,
            )
            render_duration_sec = time.monotonic() - render_started_monotonic
        elif self._config_error:
            state = "INVALID CONFIG"
            detail = self._config_error
            localization_warning = ArPathViewerNode._localization_warning_for_frame(
                self,
                base_localization_status,
                tf_checked=False,
                tf_available=False,
                tf_detail="",
            )
            render_started_monotonic = time.monotonic()
            frame = render_frame(
                image, [], state, detail=detail,
                debug_overlay=debug_overlay,
                localization_warning=localization_warning,
                robot_view_enabled=robot_view_enabled,
                robot_view_expanded=robot_view_expanded,
            )
            render_duration_sec = time.monotonic() - render_started_monotonic
        elif waiting_for_delayed_frame:
            # Show the current camera image while every AR layer remains
            # fail-closed.  This is a live preview, not a time-aligned frame.
            state = "FRAME SYNC WAIT"
            detail = (
                f"camera_live overlays_hidden reason={frame_sync_wait_reason}; "
                f"queue={frame_delay.pending_count} "
                f"discarded={frame_delay.discarded} "
                f"image_stamp={captured_image_timing.stamp_sec:.6f}"
            )
            # A short camera/TF synchronization wait is NOT a new AMCL
            # verification.  Retain a recent *verified* UI observation for
            # at most 0.5s; do not flip the warning gate without a TF check.
            localization_warning = getattr(
                self, "_last_localization_warning_for_display", None
            )
            last_checked = getattr(
                self, "_last_localization_warning_checked_monotonic", 0.0
            )
            if time.monotonic() - last_checked > 0.5:
                localization_warning = None
            recent_minimap, minimap_status = (
                ArPathViewerNode._display_minimap_snapshot(
                    self, None, now=time.monotonic(),
                    allow_recent=bool(base_localization_status.stable),
                ) if robot_view_enabled and robot_view_expanded else (None, "")
            )
            render_started_monotonic = time.monotonic()
            frame = render_frame(
                image, [], state, detail=detail,
                debug_overlay=debug_overlay,
                localization_warning=localization_warning,
                costmap_minimap=recent_minimap,
                costmap_minimap_status=minimap_status,
                costmap_alert_history=(
                    ArPathViewerNode._recent_alert_history_for_ui(self)
                ),
                robot_view_enabled=robot_view_enabled,
                robot_view_expanded=robot_view_expanded,
            )
            render_duration_sec = time.monotonic() - render_started_monotonic
        else:
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
                and ui_state["_perception_costmap_visible"]
            )
            costmap_state = "DISABLED"
            costmap_snapshot = None
            costmap_reason = "disabled_by_configuration"
            costmap_projection = None
            costmap_visual_projection = None
            costmap_visual_age_sec = None
            costmap_display_projection = None
            costmap_display_hold_age = None
            costmap_display_mode = "hidden"
            costmap_analysis_points_base = None
            costmap_path_points_base = None
            costmap_reference_points_base = None
            costmap_minimap = None
            costmap_alert = None
            costmap_alert_history = None
            costmap_alert_evaluation = "unavailable"
            costmap_alert_reason = "disabled_by_configuration"
            alert_gate = getattr(self, "_costmap_alert_gate", None)
            if alert_gate is None:
                alert_gate = CostmapAlertGate(0.0, clock=self._ros_now_sec)
                self._costmap_alert_gate = alert_gate
            costmap_explanation = None
            costmap_explanation_reason = "disabled_by_configuration"
            path_points_base = None
            real_path_image_history = False
            reference_path_image_history = False
            if costmap_enabled:
                if delayed_frame is not None:
                    costmap_state, costmap_snapshot, costmap_reason = (
                        self._costmap_store.state_for_image(
                            image_timing.stamp_sec, self._ros_now_sec()
                        )
                    )
                else:
                    costmap_state, costmap_snapshot, costmap_reason = (
                        self._costmap_store.state(self._ros_now_sec())
                    )

            localization = base_localization_status
            tf_available = False
            tf_detail = ""
            tf_checked = bool(
                state == "ACTIVE"
                or reference_state == "ACTIVE"
                or costmap_state == "ACTIVE"
            )
            if tf_checked:
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
            localization_warning = ArPathViewerNode._localization_warning_for_frame(
                self,
                localization,
                tf_checked=tf_checked,
                tf_available=tf_available,
                tf_detail=tf_detail,
            )
            # Actual robot yaw, never a proposed/imagined rotate.
            rotation_gate = getattr(self, "_rotation_motion_gate", None)
            rotation_direction = (
                rotation_gate.update(
                    tf_pose if (
                        getattr(self, "_rotation_guidance_enabled", False)
                        and tf_available and localization is not None
                        and localization.stable
                    ) else None,
                    self._ros_now_sec(),
                )
                if rotation_gate is not None else None
            )
            # Independent real-odometry evidence remains available when TF
            # heading is too sparsely sampled to pass the TF-only gate.
            # All existing follower, stable-localization and Path authorization
            # requirements still apply before an orange cue is rendered.
            if (getattr(self, "_rotation_guidance_enabled", False)
                    and tf_available and localization is not None
                    and localization.stable):
                odom_gate = getattr(self, "_odom_rotation_gate", None)
                if odom_gate is not None:
                    rotation_direction = (
                        odom_gate.direction(self._ros_now_sec())
                        or rotation_direction
                    )

            # A *measured* /odom/filtered turn does not depend on Follower
            # goal/index or AMCL map localization. Never infer planned turns.
            # OdomRotationMotionGate already enforces source frame, velocity,
            # direction consistency and a fresh (<0.22 s) measurement.
            odom_gate = getattr(self, '_odom_rotation_gate', None)
            observed_rotation_status = (
                '' if odom_gate is None
                else (odom_gate.direction(self._ros_now_sec()) or '')
            )

            if state == "ACTIVE" and snapshot is not None:
                if (
                    delayed_frame is not None
                    and snapshot.received_ros_sec is not None
                    and snapshot.received_ros_sec > image_timing.stamp_sec
                ):
                    historical_state, historical = (
                        self._path_store.state_for_ros_time(
                            image_timing.stamp_sec
                        )
                    )
                    historical_authorized = bool(
                        historical_state == "ACTIVE"
                        and historical is not None
                        and self._waypoint_progress_gate.authorizes_plan(
                            self._ros_now_sec(), historical.generation
                        )
                    )
                    if historical_authorized:
                        snapshot = historical
                        real_path_image_history = True
                    else:
                        state = "FRAME SYNC WAIT"
                        detail = (
                            "real_path_newer_than_image offset="
                            f"{snapshot.received_ros_sec - image_timing.stamp_sec:.3f}s "
                            "image_time_history_unavailable"
                        )
                        snapshot = None
                        segments = []
                        guidance = None
                        progress = None
                if (
                    state == "ACTIVE" and snapshot is not None
                    and not localization.stable
                ):
                    state = localization.state
                    detail = localization.detail
                elif state == "ACTIVE" and snapshot is not None and not tf_available:
                    state = "TF UNAVAILABLE"
                    detail = tf_detail
                elif state == "ACTIVE" and snapshot is not None:
                    try:
                        evaluation = evaluate_path_state(
                            snapshot.points, snapshot.frame_id, self._base_frame,
                            (
                                self._points_in_base
                                if delayed_frame is None
                                else lambda points, frame_id: (
                                    ArPathViewerNode._points_in_base_at(
                                        self, points, frame_id,
                                        image_timing.stamp_sec,
                                    )
                                )
                            ),
                            self._mount, self._intrinsics, self._near,
                            display_geometry=self._path_display_geometry,
                            guidance_settings=self._guidance_settings,
                            on_tf_error=lambda exc: self._diagnostics.tf_failure(
                                snapshot.frame_id, self._base_frame, exc
                            ),
                        )
                        state = evaluation.state
                        if (
                            delayed_frame is not None
                            and evaluation.state == "TF UNAVAILABLE"
                        ):
                            retry_delayed_frame = True
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
                        elif (
                            transition is not None
                            and transition.detail == "replan_confirmed"
                        ):
                            self._path_updated_until = self._ros_now_sec() + 1.0
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
                            if waypoint_status:
                                # Show only verified NEXT WP /plan length.
                                # No map-plane FINAL WP DIRECT calculation/UI.
                                destination_label = (
                                    self._waypoint_progress_gate
                                    .destination_status(
                                        self._ros_now_sec(),
                                        snapshot.generation,
                                        progress.remaining_path_m,
                                        progress.anchor_distance_m,
                                        self._guidance_settings
                                        .max_path_start_distance_m,
                                    ) or ''
                                )
                                final_waypoint_status = (
                                    destination_label.split(' | ', 1)[0]
                                )
                        if (
                            state in ("ACTIVE", "PATH OUT OF VIEW")
                            and rotation_direction is not None
                            and self._waypoint_progress_gate
                            .authorizes_plan(
                                self._ros_now_sec(), snapshot.generation
                            )
                        ):
                            guidance = OffscreenGuidance(
                                rotation_direction,
                                70.0 if rotation_direction ==
                                "ROTATE LEFT" else -70.0,
                                (progress.remaining_path_m
                                 if progress is not None else 0.0),
                                (progress.endpoint_distance_m
                                 if progress is not None else 0.0),
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
                            elif guidance.direction.startswith('ROTATE '):
                                # RotationMotionGate already requires a fresh,
                                # sustained measured yaw run. Applying the
                                # generic 0.35 s path-direction debounce again
                                # caused a measured ~0.9 s total lag.
                                self._guidance_stability.clear()
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
                            if real_path_image_history:
                                detail += " path_time=image_history"
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
                if (
                    delayed_frame is not None
                    and reference_snapshot.received_ros_sec is not None
                    and reference_snapshot.received_ros_sec
                    > image_timing.stamp_sec
                ):
                    historical_state, historical = (
                        reference_gate.state_for_ros_time(
                            image_timing.stamp_sec
                        )
                    )
                    if historical_state == "ACTIVE" and historical is not None:
                        reference_snapshot = historical
                        reference_path_image_history = True
                    else:
                        reference_reason = (
                            "reference_path_newer_than_image offset="
                            f"{reference_snapshot.received_ros_sec - image_timing.stamp_sec:.3f}s "
                            "image_time_history_unavailable"
                        )
                        reference_snapshot = None
                if reference_snapshot is None:
                    reference_segments = []
                elif localization is None or not localization.stable:
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
                            (
                                self._points_in_base
                                if delayed_frame is None
                                else lambda points, frame_id: (
                                    ArPathViewerNode._points_in_base_at(
                                        self, points, frame_id,
                                        image_timing.stamp_sec,
                                    )
                                )
                            ),
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
                        if (
                            delayed_frame is not None
                            and reference_evaluation.state == "TF UNAVAILABLE"
                        ):
                            retry_delayed_frame = True
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
                        if reference_path_image_history:
                            reference_reason += " image_time_history"
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
                    alert_gate.clear()
                elif not tf_available:
                    costmap_reason = f"localization_tf_unavailable {tf_detail}"
                    alert_gate.clear()
                else:
                    if delayed_frame is not None:
                        try:
                            drawing, analysis = self._costmap_arrays_in_base(
                                costmap_snapshot
                            )
                        except Exception as exc:
                            sync_state = "WAITING_TF"
                            synchronized = None
                            sync_reason = (
                                "waiting_for_costmap_exact_tf "
                                f"{type(exc).__name__}: {exc}"
                            )
                            retry_delayed_frame = True
                        else:
                            sync_state = "ACTIVE"
                            synchronized = SynchronizedCostmap(
                                snapshot=costmap_snapshot,
                                points_in_base=np.asarray(
                                    drawing, dtype=np.float64
                                ),
                                analysis_points_in_base=np.asarray(
                                    analysis, dtype=np.float64
                                ),
                                synchronized_at=time.monotonic(),
                            )
                            sync_reason = "history_exact_tf_ready"
                    else:
                        sync_state, synchronized, sync_reason = (
                            self._costmap_transform_queue.resolve(
                                self._ros_now_sec(), self._costmap_arrays_in_base
                            )
                        )
                    self._diagnostics.costmap_sync(
                        sync_state,
                        self._costmap_transform_queue.pending_count,
                        self._costmap_transform_queue.resolved_count,
                        self._costmap_transform_queue.expired_count,
                        sync_reason,
                    )
                    # Camera-first review: a newly resolved Costmap may
                    # be younger than this unchanged camera image.
                    # Pick an actual, previously exact-TF-ready grid.
                    ready_history = getattr(
                        self, "_costmap_ready_history", None
                    )
                    if ready_history is not None and delayed_frame is None:
                        if synchronized is not None:
                            ready_history.add(synchronized)
                        if image_timing.usable:
                            matched = ready_history.for_image(
                                image_timing.stamp_sec,
                                self._ros_now_sec(),
                                maximum_image_age_sec=(
                                    self._costmap_camera_sync_settings
                                    .max_costmap_age_sec
                                    + self._costmap_visual_hold_sec
                                ),
                                maximum_ros_age_sec=(
                                    self._costmap_settings.stamp_max_age_sec
                                ),
                            )
                            synchronized = matched
                        else:
                            synchronized = None
                        if synchronized is None:
                            sync_reason = (
                                "no_exact_ready_costmap_before_image"
                            )
                    if synchronized is None:
                        costmap_reason = sync_reason
                        if (image_timing.usable
                                and localization is not None
                                and localization.stable
                                and tf_available):
                            alert_gate.suspend()
                        else:
                            alert_gate.clear()
                    else:
                        costmap_snapshot = synchronized.snapshot
                        costmap_age = (
                            image_timing.stamp_sec - costmap_snapshot.stamp_sec
                        )
                        semantic_age_limit = (
                            self._costmap_camera_sync_settings.max_costmap_age_sec
                        )
                        configured_visual_hold = getattr(
                            self, "_costmap_visual_hold_sec",
                            getattr(
                                getattr(self, "_costmap_visual_hold", None),
                                "max_age_sec", 0.0,
                            ),
                        )
                        visual_age_limit = (
                            semantic_age_limit
                            + max(0.0, float(configured_visual_hold))
                        )
                        visual_only = bool(
                            costmap_age > semantic_age_limit
                            and costmap_age <= visual_age_limit
                        )
                        try:
                            camera_sync = (
                                ArPathViewerNode._costmap_arrays_at_image_time(
                                    self,
                                    costmap_snapshot,
                                    image_timing,
                                    max_costmap_age_sec=(
                                        visual_age_limit if visual_only else None
                                    ),
                                )
                            )
                        except Exception as exc:
                            # Suppress every Costmap-derived visual.  Keeping
                            # alerts or the mini-map while the camera alignment
                            # is unknown would still misrepresent "now".
                            costmap_reason = (
                                "camera_time_alignment_unavailable "
                                f"{type(exc).__name__}: {exc}"
                            )
                            costmap_analysis_points_base = None
                            if (image_timing.usable
                                    and isinstance(exc, ValueError)
                                    and str(exc).startswith((
                                        "costmap_camera_age_exceeded",
                                        "costmap_newer_than_image",
                                    ))
                                    and localization is not None
                                    and localization.stable):
                                alert_gate.suspend()
                            else:
                                alert_gate.clear()
                            if not isinstance(exc, ValueError):
                                retry_delayed_frame = (
                                    delayed_frame is not None
                                )
                        else:
                            points_in_base = camera_sync.points_in_base
                            if visual_only:
                                # This is an exact image-time reprojection of
                                # the last Costmap, but its cell contents exceed
                                # the semantic freshness limit. It is display
                                # evidence only: no alert, explanation, or
                                # minimap input is populated below.
                                if getattr(self, '_perception_costmap_camera_overlay_enabled', True):
                                    costmap_visual_projection = project_costmap_points(
                                        points_in_base,
                                        costmap_snapshot.classes,
                                        self._mount,
                                        self._intrinsics,
                                        self._near,
                                    )
                                costmap_visual_age_sec = costmap_age
                                costmap_reason = (
                                    "visual_only_exact_time_reprojection "
                                    f"age={costmap_age:.3f}s "
                                    f"semantic_limit={semantic_age_limit:.3f}s "
                                    f"translation={camera_sync.translation_m:.3f}m "
                                    f"rotation={camera_sync.rotation_deg:.2f}deg"
                                )
                            if not visual_only:
                                # Only the original semantic age limit can
                                # populate evidence consumed by warnings and
                                # route explanations.
                                costmap_analysis_points_base = (
                                    camera_sync.analysis_points_in_base
                                )
                                if getattr(self, '_perception_costmap_camera_overlay_enabled', True):
                                    costmap_projection = project_costmap_points(
                                        points_in_base,
                                        costmap_snapshot.classes,
                                        self._mount,
                                        self._intrinsics,
                                        self._near,
                                    )
                                if costmap_projection is None:
                                    costmap_reason = "camera_costmap_overlay_disabled"
                                else:
                                    costmap_reason = (
                                        "visible"
                                        if costmap_projection.visible_points > 0
                                        else "out_of_view"
                                    )
                                costmap_reason += (
                                    " camera_time_aligned "
                                    f"age={camera_sync.costmap_age_sec:.3f}s "
                                    f"translation={camera_sync.translation_m:.3f}m "
                                    f"rotation={camera_sync.rotation_deg:.2f}deg"
                                )
                                bound_path, pairing_reason = check_costmap_path_pair(
                                    getattr(
                                        self, "_costmap_path_bindings", {}
                                    ).get(
                                        (
                                            costmap_snapshot.frame_id,
                                            costmap_snapshot.stamp_sec,
                                        )
                                    ),
                                    costmap_stamp_sec=costmap_snapshot.stamp_sec,
                                    costmap_frame_id=costmap_snapshot.frame_id,
                                    current_path=snapshot,
                                    current_follower_identity=(
                                        follower_goal_identity(
                                            getattr(
                                                self,
                                                "_waypoint_progress_gate",
                                                None,
                                            ),
                                            self._ros_now_sec(),
                                        )
                                    ),
                                )
                                if bound_path is None:
                                    costmap_reason += (
                                        f" path_pair_rejected={pairing_reason}"
                                    )
                                else:
                                    try:
                                        costmap_path_points_base = (
                                            ArPathViewerNode._points_in_base_at(
                                                self,
                                                bound_path.points,
                                                bound_path.frame_id,
                                                image_timing.stamp_sec,
                                            )
                                        )
                                    except Exception as exc:
                                        costmap_path_points_base = None
                                        costmap_reason += (
                                            " path_image_tf_unavailable "
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                if reference_snapshot is not None:
                                    try:
                                        costmap_reference_points_base = (
                                            ArPathViewerNode._points_in_base_at(
                                                self,
                                                reference_snapshot.points,
                                                reference_snapshot.frame_id,
                                                image_timing.stamp_sec,
                                            )
                                        )
                                    except Exception:
                                        # Reference geometry is optional on
                                        # the mini-map and never weakens the
                                        # real-path evidence gate.
                                        costmap_reference_points_base = None
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
            elif costmap_enabled:
                if delayed_frame is None:
                    self._costmap_transform_queue.clear(costmap_reason)
                alert_gate.clear()

            # Rendering may briefly retain only the image pixels from the
            # last exact-time-confirmed projection. Semantic consumers below
            # continue to use the live variables above and therefore remain
            # fail-closed whenever current synchronization is unavailable.
            visual_hold = getattr(self, "_costmap_visual_hold", None)
            if visual_hold is None:
                # Preserve composed/test construction paths which predate the
                # display-only cache. A zero-second cache still passes a live
                # projection through and can never retain it as stale output.
                visual_hold = CostmapVisualHold(
                    getattr(self, "_costmap_visual_hold_sec", 0.0)
                )
                self._costmap_visual_hold = visual_hold
            live_visual_context = (
                "" if costmap_snapshot is None else costmap_snapshot.frame_id
            )
            # Only bridge the expected gap created by Costmap publication
            # cadence/freshness. Motion-limit, clock, frame, or TF failures
            # invalidate the projection instead of reusing old image pixels.
            if visual_hold is not None and costmap_projection is not None:
                display = visual_hold.confirm(
                    costmap_projection, live_visual_context
                )
                self._last_costmap_visual_context = live_visual_context
            elif costmap_visual_projection is not None:
                # Unlike the former pixel hold, this projection was recomputed
                # with exact TF at this image timestamp. Its old Costmap
                # contents are visibly labelled and never enter semantics.
                visual_hold.clear()
                display = CostmapVisualDisplay(
                    costmap_visual_projection,
                    "hold",
                    float(costmap_visual_age_sec or 0.0),
                )
            else:
                if visual_hold is not None:
                    visual_hold.clear()
                    self._last_costmap_visual_context = ""
                display = None
            if display is not None and display.projection is not None:
                costmap_display_projection = display.projection
                costmap_display_mode = display.mode
                if display.mode == "hold":
                    costmap_display_hold_age = display.age_sec
            display_reason = (
                costmap_reason
                if costmap_display_mode == "hidden"
                else (
                    "exact_time_synchronized"
                    if costmap_display_mode == "live"
                    else "last_exact_time_projection_visual_only"
                )
            )
            if costmap_enabled:
                self._diagnostics.costmap_visual_display(
                    costmap_display_mode,
                    0.0 if costmap_display_hold_age is None
                    else costmap_display_hold_age,
                    display_reason,
                )

            explanation_enabled = bool(
                getattr(self, "_perception_explanation_enabled", False)
                and ui_state["_perception_explanation_visible"]
                and debug_overlay
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
                elif costmap_path_points_base is None:
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

            minimap_enabled = robot_view_enabled
            if (
                minimap_enabled
                and robot_view_expanded
                and costmap_analysis_points_base is not None
                and costmap_snapshot is not None
                and localization is not None
                and localization.stable
            ):
                costmap_minimap = CostmapMinimap(
                    points_in_base=costmap_analysis_points_base,
                    costs=costmap_snapshot.analysis_costs,
                    path_points_base=(
                        costmap_path_points_base
                        if state in ("ACTIVE", "PATH OUT OF VIEW")
                        else None
                    ),
                    high_cost_min=self._costmap_settings.high_cost_min,
                    lethal_cost_min=self._costmap_settings.lethal_cost_min,
                    reference_points_base=costmap_reference_points_base,
                )

            alert_enabled = bool(
                getattr(self, "_perception_alert_enabled", False)
                and ui_state["_perception_alert_visible"]
            )
            if alert_enabled:
                now_sec = self._ros_now_sec()
                path_updated_event = bool(
                    state in ("ACTIVE", "PATH OUT OF VIEW")
                    and now_sec <= getattr(self, "_path_updated_until", 0.0)
                )
                semantic_inputs_available = not (
                    costmap_analysis_points_base is None
                    or costmap_snapshot is None
                    or localization is None
                    or not localization.stable
                )
                candidate = None
                if semantic_inputs_available:
                    # Robot-proximity evidence is a property of the fresh,
                    # exact-time Local Costmap.  It does not require a real
                    # Path, so a temporarily unavailable /plan must not hide
                    # a nearby lethal-cost cell.  This remains a neutral
                    # Costmap statement, not a person/collision claim.
                    candidate, costmap_alert_reason = (
                        evaluate_robot_proximity_alert(
                            costmap_analysis_points_base,
                            costmap_snapshot.analysis_costs,
                            lethal_cost_min=(
                                self._costmap_settings.lethal_cost_min
                            ),
                            settings=self._costmap_alert_settings,
                        )
                    )
                    if candidate is None:
                        # Earlier amber notice, from the SAME exact-time,
                        # fresh Costmap used above. Never infer a person.
                        candidate, costmap_alert_reason = (
                            evaluate_forward_cost_alert(
                                costmap_analysis_points_base,
                                costmap_snapshot.analysis_costs,
                                high_cost_min=self._costmap_settings.high_cost_min,
                                lethal_cost_min=self._costmap_settings.lethal_cost_min,
                                settings=self._costmap_alert_settings,
                            )
                        )
                if candidate is not None:
                    costmap_alert = alert_gate.update(candidate)
                    costmap_alert_evaluation = "unconfirmed"
                    if costmap_alert is None:
                        costmap_alert_reason = "confirmation_pending"
                    else:
                        costmap_alert_evaluation = "active"
                elif path_updated_event:
                    # Preserve the existing /plan event even when perception
                    # is unavailable. It reports only a confirmed generation
                    # change and makes no Costmap-causality claim. Fresh near-
                    # robot lethal evidence above takes precedence.
                    costmap_alert = CostmapAlert(
                        "PATH UPDATED", "path_updated",
                        "confirmed_real_plan_generation_change",
                    )
                    costmap_alert_reason = "confirmed_real_plan_update"
                    alert_gate.clear()
                elif not semantic_inputs_available:
                    # Missing image-time evidence is not evidence of
                    # safety, nor may it produce a live alert. A short
                    # gap can retain only a *pending* confirmation.
                    if (image_timing.usable
                            and localization is not None
                            and localization.stable and tf_available):
                        alert_gate.suspend()
                    else:
                        alert_gate.clear()
                    costmap_alert_reason = "synchronized_inputs_unavailable"
                elif (
                    state not in ("ACTIVE", "PATH OUT OF VIEW")
                    or costmap_path_points_base is None
                ):
                    alert_gate.clear()
                    costmap_alert_reason = (
                        "robot_proximity_clear_path_inputs_unavailable"
                    )
                else:
                    candidate, costmap_alert_reason = evaluate_path_cost_alert(
                        costmap_analysis_points_base,
                        costmap_snapshot.analysis_costs,
                        costmap_path_points_base,
                        high_cost_min=self._costmap_settings.high_cost_min,
                        lethal_cost_min=self._costmap_settings.lethal_cost_min,
                        settings=self._costmap_alert_settings,
                    )
                    if candidate is None:
                        # A complete exact-time robot/path evaluation found no
                        # alert. Any historical banner is explicitly marked
                        # CLEARED by CostmapAlertHistoryGate.
                        alert_gate.clear()
                        costmap_alert = None
                        costmap_alert_evaluation = "cleared"
                    else:
                        costmap_alert = alert_gate.update(candidate)
                        costmap_alert_evaluation = "unconfirmed"
                    if candidate is not None and costmap_alert is None:
                        costmap_alert_reason = "confirmation_pending"
                    elif costmap_alert is not None:
                        costmap_alert_evaluation = "active"
                history_gate = getattr(
                    self, "_costmap_alert_history_gate", None
                )
                if history_gate is None:
                    history_gate = CostmapAlertHistoryGate(
                        5.0, clock=self._ros_now_sec
                    )
                    self._costmap_alert_history_gate = history_gate
                costmap_alert_history = history_gate.update(
                    costmap_alert,
                    evaluation_status=costmap_alert_evaluation,
                    goal_identity=follower_goal_identity(
                        getattr(self, "_waypoint_progress_gate", None),
                        self._ros_now_sec(),
                    ),
                )
                if costmap_alert is not None:
                    costmap_alert_history = None
                self._diagnostics.costmap_alert(
                    costmap_alert is not None,
                    costmap_alert_reason,
                    "" if costmap_alert is None else costmap_alert.detail,
                )
            else:
                alert_gate = getattr(self, "_costmap_alert_gate", None)
                if alert_gate is not None:
                    alert_gate.clear()
                history_gate = getattr(
                    self, "_costmap_alert_history_gate", None
                )
                if history_gate is not None:
                    history_gate.clear()

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
                    "visible="
                    f"{0 if costmap_projection is None else costmap_projection.visible_points}"
                )
                if debug_overlay:
                    costmap_summary += (
                        f" image_stamp={image_timing.stamp_sec:.6f} "
                        "costmap_stamp="
                        f"{0.0 if costmap_snapshot is None else costmap_snapshot.stamp_sec:.6f} "
                        "path_gen="
                        f"{0 if snapshot is None else snapshot.generation}"
                    )
                detail = (
                    f"{detail}; {costmap_summary}"
                    if detail else costmap_summary
                )
            if explanation_enabled and debug_overlay:
                explanation_summary = (
                    "explanation="
                    f"{costmap_explanation_reason}"
                )
                detail = (
                    f"{detail}; {explanation_summary}"
                    if detail else explanation_summary
                )
            if retry_delayed_frame:
                # Keep this exact image/timestamp pair in the bounded delay
                # queue.  Do not draw paths or perception data using a newer
                # pose while waiting for the requested TF sample.
                state = "FRAME SYNC WAIT"
                detail = f"waiting_for_exact_image_time_tf; {costmap_reason}"
                segments = []
                reference_segments = []
                guidance = None
                waypoint_status = ""
                final_waypoint_status = ""
                navigation_status = ""
                costmap_projection = None
                costmap_display_projection = None
                costmap_display_hold_age = None
                costmap_explanation = None
                costmap_minimap = None
                costmap_alert = None
                # Keep the exact image/timestamp pair queued for a later TF
                # retry, but never freeze that old image on screen.  The live
                # preview remains visible with every unsynchronized overlay
                # suppressed.
                image = preview_image
                display_image_timing = captured_image_timing
                camera_pipeline_mode = "preview_tf_wait"
            elif delayed_frame is not None:
                frame_delay.consume(delayed_frame)
            if state == "FRAME SYNC WAIT":
                # A newer Path or any other image-time mismatch can reach this
                # state after a delayed candidate was selected.  Keep the
                # camera live and clear every overlay rather than presenting
                # an old synchronized image as though it were current.
                image = preview_image
                display_image_timing = captured_image_timing
                segments = []
                reference_segments = []
                guidance = None
                waypoint_status = ""
                final_waypoint_status = ""
                navigation_status = ""
                costmap_projection = None
                costmap_display_projection = None
                costmap_display_hold_age = None
                costmap_explanation = None
                costmap_minimap = None
                costmap_alert = None
                if camera_pipeline_mode == "synchronized":
                    camera_pipeline_mode = "preview_path_wait"
            if (self._waypoint_progress_gate.navigation_hold_status(
                    self._ros_now_sec()) == "ROUTE COMPLETE"):
                # Only explicit follower activity 9 authorizes this.
                state = "NAVIGATION HOLD"
                detail = "follower_reported_finished"
                navigation_status = "ROUTE COMPLETE"
                waypoint_status = ""
                final_waypoint_status = ""
                segments = []
                reference_segments = []
                guidance = None
            final_success = getattr(self, "_final_waypoint_success", None)
            if final_success is not None:
                success_age = self._ros_now_sec() - final_success[0]
                if not 0.0 <= success_age <= 20.0:
                    self._final_waypoint_success = None
                elif (navigation_status != "ROUTE COMPLETE"
                      and state not in ("ACTIVE", "PATH OUT OF VIEW")):
                    # The Follower's arrival threshold is not evidence
                    # that the next map or the entire route was completed.
                    state = "NAVIGATION HOLD"
                    detail = "follower_final_threshold_reached; route_unconfirmed"
                    navigation_status = "FINAL WP REACHED"
                    waypoint_status = ""
                    final_waypoint_status = ""
                    segments = []
                    reference_segments = []
                    guidance = None
            travel_distance_status = ''
            if navigation_status in ('ROUTE COMPLETE', 'FINAL WP REACHED'):
                meter = getattr(self, '_odom_distance_meter', None)
                distance = None if meter is None else meter.total_m()
                if distance is not None:
                    travel_distance_status = f'ODOM TRAVELED: ~{distance:.1f} m'
            minimap_status = ""
            if robot_view_enabled and robot_view_expanded:
                costmap_minimap, minimap_status = (
                    ArPathViewerNode._display_minimap_snapshot(
                        self, costmap_minimap, now=time.monotonic(),
                        allow_recent=bool(
                            localization is not None and localization.stable
                            and state in (
                                "ACTIVE", "PATH OUT OF VIEW",
                                "FRAME SYNC WAIT", "REPLANNING",
                            )
                        ),
                    )
                )
            render_started_monotonic = time.monotonic()
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
                final_waypoint_status=final_waypoint_status,
                navigation_status=navigation_status,
                observed_rotation_status=observed_rotation_status,
                travel_distance_status=travel_distance_status,
                debug_overlay=debug_overlay,
                path_alpha=getattr(self, "_path_alpha", 1.0),
                costmap_projection=(
                    costmap_display_projection
                    if getattr(self, "_perception_costmap_camera_overlay_enabled", True)
                    else None
                ),
                costmap_alpha=getattr(self, "_costmap_alpha", 0.42),
                costmap_radius_px=getattr(self, "_costmap_radius", 2),
                costmap_sync_hold_age_sec=(
                    costmap_display_hold_age
                    if getattr(self, "_perception_costmap_camera_overlay_enabled", True)
                    else None
                ),
                costmap_explanation=costmap_explanation,
                costmap_minimap=costmap_minimap,
                costmap_minimap_size_px=getattr(
                    self, "_costmap_minimap_size", 142
                ),
                costmap_minimap_range_m=getattr(
                    self, "_costmap_minimap_range", 3.0
                ),
                costmap_minimap_status=minimap_status,
                costmap_alert=costmap_alert,
                costmap_alert_history=costmap_alert_history,
                localization_warning=localization_warning,
                robot_view_enabled=robot_view_enabled,
                robot_view_expanded=robot_view_expanded,
            )
            render_duration_sec = time.monotonic() - render_started_monotonic
        self._diagnostics.state_changed(state, detail)
        recorded = False
        writer_duration_sec = 0.0
        if self._writer is not None:
            writer_started = time.monotonic()
            write_result = self._writer.write(frame)
            # Existing test/custom writers follow cv2 and return None; only an
            # explicit False from the bounded async writer means rejection.
            recorded = write_result is not False
            writer_duration_sec = time.monotonic() - writer_started
        ArPathViewerNode._publish_display_frame(
            self,
            frame,
            getattr(
                display_image_timing,
                "source_timestamp_monotonic_sec",
                None,
            ),
        )
        self._diagnostics.camera_pipeline(
            read_success=captured_image is not None,
            timestamp_usable=captured_image_timing.usable,
            timestamp_reason=captured_image_timing.reason,
            mode=(
                "unavailable" if captured_image is None
                else camera_pipeline_mode
            ),
            pending_frames=(
                0 if frame_delay is None else frame_delay.pending_count
            ),
            discarded_frames=(
                0 if frame_delay is None else frame_delay.discarded
            ),
            recorded=recorded,
            source_metrics=source_read_metrics,
            tick_interval_sec=tick_interval_sec,
            tick_duration_sec=time.monotonic() - capture_started_monotonic,
            pre_render_duration_sec=max(
                0.0, render_started_monotonic - capture_finished_monotonic
            ),
            render_duration_sec=render_duration_sec,
            writer_duration_sec=writer_duration_sec,
            target_fps=getattr(self, "_target_fps", 20.0),
            recording_output_frames=(
                0 if self._writer is None
                else getattr(self._writer, "frames_written", 0)
            ),
            recording_duplicate_frames=(
                0 if self._writer is None
                else getattr(self._writer, "duplicate_frames_written", 0)
            ),
            recording_writer_metrics=self._writer,
        )

    def _request_shutdown(self, reason: str) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._log_info(f"AR Path Viewer shutdown requested: {reason}")
        # Resource destruction (especially OpenCV HighGUI/Tk) belongs to the
        # creating main thread.  main() stops the executor, then destroys the
        # node there.  Shutting down the context wakes executor.spin().
        shutdown_ros_context()

    def _release_resources(self) -> None:
        ArPathViewerNode._require_main_thread(self)
        if self._resources_released:
            return
        self._resources_released = True
        display_frames = getattr(self, "_display_frames", None)
        if display_frames is not None:
            display_frames.close()
        self._image_source.close()
        if bool(getattr(self._image_source, "worker_alive", False)):
            self._log_warning(
                "Camera capture worker did not stop within the bounded timeout"
            )
        if self._writer is not None:
            self._writer.release()
            writer_error = str(getattr(self._writer, "error_reason", ""))
            if writer_error:
                self._log_warning(
                    f"Video writer shutdown status: {writer_error}"
                )
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


def perception_toggle_key(key_code: int):
    """Map review-only UI keys without changing any Nav2 setting."""
    if key_code is None or key_code < 0:
        return None
    code = int(key_code) & 0xFFFFFFFF
    mapping = {
        ord('c'): ("_perception_costmap_visible", "COSTMAP AR"),
        ord('C'): ("_perception_costmap_visible", "COSTMAP AR"),
        ord('m'): ("_perception_minimap_visible", "MINIMAP"),
        ord('M'): ("_perception_minimap_visible", "MINIMAP"),
        ord('e'): ("_perception_explanation_visible", "EXPLANATION"),
        ord('E'): ("_perception_explanation_visible", "EXPLANATION"),
        ord('a'): ("_perception_alert_visible", "COST ALERT"),
        ord('A'): ("_perception_alert_visible", "COST ALERT"),
    }
    return mapping.get(code)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArPathViewerNode()
    # HighGUI requires window creation, imshow/waitKey, and destruction on one
    # thread (the process main thread on Qt/GTK).  ROS callbacks therefore run
    # in this background executor while the main thread pumps the GUI below.
    # Two executor workers remain necessary: camera/rendering uses the node's
    # mutually-exclusive default callback group, while TransformListener's
    # reentrant callbacks can continue ingesting 30 Hz TF on the other worker.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_done = threading.Event()
    executor_failure: list[tuple[BaseException, str]] = []

    def spin_ros_callbacks() -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            # q/F10 or a sibling launch shutdown closes the ROS context.
            pass
        except BaseException as exc:  # reported and re-raised on main below
            executor_failure.append((exc, traceback.format_exc()))
        finally:
            executor_done.set()

    executor_thread = threading.Thread(
        target=spin_ros_callbacks,
        name="ar-path-viewer-ros-executor",
        daemon=False,
    )
    executor_thread.start()
    pending_failure: BaseException | None = None
    try:
        node.run_main_thread_ui(executor_done)
        if executor_failure:
            pending_failure, failure_traceback = executor_failure[0]
            node._log_error(
                "Unhandled ROS executor exception: "
                f"{type(pending_failure).__name__}: {pending_failure}\n"
                f"{failure_traceback}"
            )
    except KeyboardInterrupt:
        if rclpy.ok(context=node.context):
            node._request_shutdown("Ctrl+C")
    except Exception as exc:
        node._log_error(
            f"Unhandled exception: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        pending_failure = exc
    finally:
        # Wake spin() before waiting for the worker.  Resource release is
        # deliberately postponed until the executor has stopped, so camera
        # read/render/write cannot race close()/release().
        shutdown_ros_context()
        try:
            executor.shutdown(timeout_sec=2.0)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        executor_thread.join(timeout=2.0)
        if executor_thread.is_alive():
            node._log_error(
                "ROS executor did not stop within 2.0s; a camera read or ROS "
                "callback may be blocked, and forced cleanup is not a normal "
                "shutdown result"
            )
            if pending_failure is None:
                pending_failure = RuntimeError("ROS executor shutdown timed out")
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            # launch may forward a second SIGINT while rclpy is already
            # destroying internal services. Viewer resources were released
            # first by the destroy_node override, so this is normal unwind.
            pass
        try:
            shutdown_ros_context()
        except KeyboardInterrupt:
            pass
    if pending_failure is not None:
        raise pending_failure


def shutdown_ros_context() -> None:
    """Idempotently shut down rclpy after node destruction."""
    rclpy.try_shutdown()
