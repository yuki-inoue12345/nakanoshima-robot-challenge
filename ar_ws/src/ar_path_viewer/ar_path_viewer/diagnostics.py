"""Rate-limited diagnostics for Path reception and viewer state transitions."""

from collections import deque
import math
import threading
import time
from typing import Callable, Optional


class DiagnosticReporter:
    """Collect counters even when logs are throttled, without ROS dependencies."""

    def __init__(
        self,
        enabled: bool,
        logger,
        min_interval_sec: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        record_sink: Callable[[str, str], None] | None = None,
    ) -> None:
        if min_interval_sec <= 0.0:
            raise ValueError("diagnostic_log_interval_sec must be positive")
        self.enabled = bool(enabled)
        self.logger = logger
        self.min_interval_sec = float(min_interval_sec)
        self.clock = clock
        self.record_sink = record_sink
        self.path_receive_count = 0
        self.reference_receive_count = 0
        self.costmap_receive_count = 0
        self.camera_read_count = 0
        self.camera_read_success_count = 0
        self.camera_timestamp_usable_count = 0
        self.camera_display_frame_count = 0
        self.camera_synchronized_frame_count = 0
        self.camera_recorded_frame_count = 0
        self.costmap_live_display_count = 0
        self.costmap_hold_display_count = 0
        self.costmap_hidden_display_count = 0
        self._camera_transport_age_sec = deque(maxlen=200)
        self._camera_read_duration_sec = deque(maxlen=200)
        self._camera_dequeue_duration_sec = deque(maxlen=200)
        self._camera_decode_duration_sec = deque(maxlen=200)
        self._camera_tick_interval_sec = deque(maxlen=200)
        self._camera_tick_duration_sec = deque(maxlen=200)
        self._camera_pre_render_duration_sec = deque(maxlen=200)
        self._camera_render_duration_sec = deque(maxlen=200)
        self._camera_writer_duration_sec = deque(maxlen=200)
        self._camera_source_interval_sec = deque(maxlen=200)
        self._camera_gui_interval_sec = deque(maxlen=200)
        self._camera_gui_new_interval_sec = deque(maxlen=200)
        self._camera_display_age_sec = deque(maxlen=200)
        self._camera_previous_source_stamp = None
        self._camera_previous_gui_time = None
        self._camera_previous_gui_new_time = None
        self.camera_source_gap_frames = 0
        self.camera_gui_present_count = 0
        self.camera_gui_new_frame_count = 0
        self.camera_gui_duplicate_count = 0
        self.camera_gui_deadline_miss_count = 0
        self._camera_gui_lock = threading.Lock()
        self.last_path_received_at: Optional[float] = None
        self.last_path_outcome = "no Path received since startup"
        self.current_state: Optional[str] = None
        self._last_localization_state: Optional[str] = None
        self._last_emitted = {}

    def _write_log(self, level: str, message: str) -> None:
        """Call each rclpy severity from a stable, severity-specific call site."""
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level in ("warning", "warn"):
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "fatal":
            self.logger.fatal(message)
        else:
            raise ValueError(f"unsupported diagnostic log level: {level!r}")

    def _emit(self, level: str, key: str, message: str, force: bool = False) -> None:
        if not self.enabled:
            return
        now = self.clock()
        previous = self._last_emitted.get(key)
        if not force and previous is not None and now - previous < self.min_interval_sec:
            return
        self._last_emitted[key] = now
        if self.record_sink is not None:
            try:
                self.record_sink(level, message)
            except Exception:
                # File/UI diagnostics must never interrupt ROS callbacks or rendering.
                pass
        self._write_log(level, message)

    def path_received(
        self, frame_id: str, point_count: int, header_stamp_sec: float | None = None
    ) -> None:
        self.path_receive_count += 1
        self.last_path_received_at = self.clock()
        stamp = "unavailable" if header_stamp_sec is None else f"{header_stamp_sec:.9f}"
        self._emit(
            "info",
            "path_received",
            "plan_rx "
            f"count={self.path_receive_count} "
            f"received_monotonic={self.last_path_received_at:.6f} "
            f"header_stamp={stamp} frame_id={frame_id!r} points={point_count}",
            force=self.path_receive_count == 1,
        )

    def path_discarded(self, reason: str) -> None:
        self.last_path_outcome = f"discarded: {reason}"
        self._emit(
            "warning",
            f"path_discarded:{reason}",
            f"plan_discarded count={self.path_receive_count} reason={reason}",
        )

    def path_accepted(self, frame_id: str, point_count: int, generation: int, updated: bool) -> None:
        action = "updated" if updated else "adopted"
        self.last_path_outcome = (
            f"valid Path {action}: generation={generation}, "
            f"frame_id={frame_id!r}, points={point_count}"
        )
        self._emit(
            "info",
            "path_accepted",
            f"plan_{action} count={self.path_receive_count} generation={generation} "
            f"frame_id={frame_id!r} points={point_count}",
            force=not updated,
        )

    def tf_failure(self, path_frame: str, base_frame: str, error: Exception) -> None:
        self._emit(
            "warning",
            f"tf_failure:{path_frame}:{base_frame}:{type(error).__name__}",
            f"tf_unavailable source_frame={path_frame!r} target_frame={base_frame!r} "
            f"error={type(error).__name__}: {error}",
        )

    def path_out_of_view(self, frame_id: str, point_count: int) -> None:
        self._emit(
            "info",
            "path_out_of_view",
            f"path_out_of_view frame_id={frame_id!r} points={point_count} "
            "reason=no clipped segment intersects the camera frustum",
        )

    def projection_update(
        self, frame_id: str, stats_detail: str, out_of_view_reason: str = ""
    ) -> None:
        event = "path_out_of_view" if out_of_view_reason else "path_projection"
        reason = f" reason={out_of_view_reason}" if out_of_view_reason else ""
        self._emit(
            "info",
            f"projection:{event}",
            f"{event} frame_id={frame_id!r} {stats_detail}{reason}",
        )

    def offscreen_guidance(self, frame_id: str, detail: str) -> None:
        self._emit(
            "info",
            "offscreen_guidance",
            f"offscreen_guidance frame_id={frame_id!r} {detail}",
        )

    def reference_update(self, event: str, valid: bool, detail: str) -> None:
        """Rate-limit reference-path diagnostics independently from /plan."""
        self.reference_receive_count += int(event == "path_state")
        level = "info" if valid else "warning"
        self._emit(
            level,
            f"reference:{event}:{valid}:{detail}",
            "reference_path_state "
            f"event={event} valid={str(valid).lower()} "
            f"reference_rx_count={self.reference_receive_count} detail={detail}",
            force=event == "path_state" and self.reference_receive_count == 1,
        )

    def reference_projection(
        self, frame_id: str, valid: bool, detail: str, reason: str
    ) -> None:
        level = "info" if valid else "warning"
        self._emit(
            level,
            f"reference_projection:{valid}:{reason}",
            "reference_path_projection "
            f"valid={str(valid).lower()} frame_id={frame_id!r} "
            f"reason={reason} {detail}",
        )

    def costmap_update(self, valid: bool, detail: str) -> None:
        """Record Local Costmap reception independently from path state."""
        self.costmap_receive_count += 1
        level = "info" if valid else "warning"
        self._emit(
            level,
            f"costmap_update:{valid}:{detail}",
            "perception_costmap_state "
            f"valid={str(valid).lower()} "
            f"receive_count={self.costmap_receive_count} detail={detail}",
            force=self.costmap_receive_count == 1,
        )

    def costmap_projection(
        self,
        valid: bool,
        frame_id: str,
        visible_points: int,
        sampled_points: int,
        reason: str,
    ) -> None:
        """Explain why the perception layer is visible or suppressed."""
        level = "info" if valid and visible_points > 0 else "warning"
        # tf2 exception text embeds changing timestamps.  Key on the stable
        # category so a 20 fps retry cannot bypass diagnostic throttling.
        category = str(reason).split(" ", 1)[0]
        self._emit(
            level,
            f"costmap_projection:{valid}:{category}",
            "perception_costmap_projection "
            f"valid={str(valid).lower()} frame_id={frame_id!r} "
            f"visible_points={int(visible_points)} "
            f"sampled_points={int(sampled_points)} reason={reason}",
        )

    def costmap_sync(
        self, state: str, pending: int, resolved: int, expired: int, reason: str
    ) -> None:
        level = "info" if state == "ACTIVE" else "warning"
        category = str(reason).split(" ", 1)[0]
        self._emit(
            level,
            f"costmap_sync:{state}:{category}",
            "perception_costmap_sync "
            f"state={state} pending={int(pending)} resolved={int(resolved)} "
            f"expired={int(expired)} reason={reason}",
        )

    def costmap_visual_display(
        self, mode: str, hold_age_sec: float, reason: str
    ) -> None:
        """Count live, display-only hold, and hidden perception frames."""
        if mode == "live":
            self.costmap_live_display_count += 1
        elif mode == "hold":
            self.costmap_hold_display_count += 1
        else:
            mode = "hidden"
            self.costmap_hidden_display_count += 1
        level = "info" if mode in ("live", "hold") else "warning"
        category = str(reason).split(" ", 1)[0]
        self._emit(
            level,
            f"costmap_visual:{mode}:{category}",
            "perception_costmap_visual "
            f"mode={mode} hold_age_sec={max(0.0, float(hold_age_sec)):.3f} "
            f"live_frames={self.costmap_live_display_count} "
            f"hold_frames={self.costmap_hold_display_count} "
            f"hidden_frames={self.costmap_hidden_display_count} "
            f"reason={reason}",
        )

    def costmap_alert(self, visible: bool, reason: str, detail: str = "") -> None:
        level = "info" if visible else "warning"
        suffix = f" {detail}" if detail else ""
        self._emit(
            level,
            f"costmap_alert:{visible}:{reason}",
            "perception_costmap_alert "
            f"visible={str(visible).lower()} reason={reason}{suffix}",
        )

    def costmap_explanation(
        self, visible: bool, reason: str, detail: str = ""
    ) -> None:
        """Record the evidence gate without claiming planner causality."""
        suffix = f" {detail}" if detail else ""
        level = "info" if visible else "warning"
        self._emit(
            level,
            f"costmap_explanation:{visible}:{reason}",
            "perception_path_explanation "
            f"visible={str(visible).lower()} reason={reason}{suffix}",
        )

    def camera_configuration(self, detail: str) -> None:
        """Record the effective projection model once for field-log correlation."""
        self._emit(
            "info",
            "camera_configuration",
            f"camera_configuration {detail}",
            force=True,
        )

    def camera_pipeline(
        self,
        *,
        read_success: bool,
        timestamp_usable: bool,
        timestamp_reason: str,
        mode: str,
        pending_frames: int,
        discarded_frames: int,
        recorded: bool,
        source_metrics=None,
        tick_interval_sec: float | None = None,
        tick_duration_sec: float | None = None,
        pre_render_duration_sec: float | None = None,
        render_duration_sec: float | None = None,
        writer_duration_sec: float | None = None,
        target_fps: float | None = None,
        recording_output_frames: int = 0,
        recording_duplicate_frames: int = 0,
        recording_writer_metrics=None,
    ) -> None:
        """Expose bounded camera/synchronization progress for field logs."""
        self.camera_read_count += 1
        self.camera_read_success_count += int(bool(read_success))
        self.camera_timestamp_usable_count += int(bool(timestamp_usable))
        self.camera_display_frame_count += 1
        self.camera_synchronized_frame_count += int(mode == "synchronized")
        self.camera_recorded_frame_count += int(bool(recorded))
        self._append_camera_metric(
            self._camera_transport_age_sec,
            getattr(source_metrics, "transport_age_sec", None),
        )
        self._append_camera_metric(
            self._camera_read_duration_sec,
            getattr(source_metrics, "read_duration_sec", None),
        )
        self._append_camera_metric(
            self._camera_dequeue_duration_sec,
            getattr(source_metrics, "dequeue_duration_sec", None),
        )
        self._append_camera_metric(
            self._camera_decode_duration_sec,
            getattr(source_metrics, "decode_duration_sec", None),
        )
        self._append_camera_metric(
            self._camera_tick_interval_sec, tick_interval_sec
        )
        self._append_camera_metric(
            self._camera_tick_duration_sec, tick_duration_sec
        )
        self._append_camera_metric(
            self._camera_pre_render_duration_sec, pre_render_duration_sec
        )
        self._append_camera_metric(
            self._camera_render_duration_sec, render_duration_sec
        )
        self._append_camera_metric(
            self._camera_writer_duration_sec, writer_duration_sec
        )
        source_stamp = getattr(
            source_metrics, "frame_timestamp_monotonic_sec", None
        )
        explicit_source_interval = getattr(
            source_metrics, "source_interval_sec", None
        )
        sequence_gap = max(
            0, int(getattr(source_metrics, "sequence_gap", 0) or 0)
        )
        if (
            explicit_source_interval is not None
            and math.isfinite(float(explicit_source_interval))
            and float(explicit_source_interval) > 0.0
        ):
            self._append_camera_metric(
                self._camera_source_interval_sec,
                float(explicit_source_interval),
            )
        elif source_stamp is not None and math.isfinite(float(source_stamp)):
            source_stamp = float(source_stamp)
            if (
                self._camera_previous_source_stamp is not None
                and source_stamp > self._camera_previous_source_stamp
            ):
                self._append_camera_metric(
                    self._camera_source_interval_sec,
                    (source_stamp - self._camera_previous_source_stamp)
                    / (sequence_gap + 1),
                )
            self._camera_previous_source_stamp = source_stamp
        self.camera_source_gap_frames += sequence_gap
        reason = str(timestamp_reason).split(" ", 1)[0]
        level = "info" if read_success else "warning"
        usable_rate = (
            100.0 * self.camera_timestamp_usable_count
            / max(1, self.camera_read_count)
        )
        sync_rate = (
            100.0 * self.camera_synchronized_frame_count
            / max(1, self.camera_read_count)
        )
        source_detail = ""
        if source_metrics is not None:
            source_detail = (
                f" v4l2_sequence={getattr(source_metrics, 'sequence', None)}"
                " v4l2_stamp_mono="
                f"{getattr(source_metrics, 'frame_timestamp_monotonic_sec', None)}"
                " dq_mono="
                f"{getattr(source_metrics, 'dequeue_monotonic_sec', None)}"
                f" sequence_gap={getattr(source_metrics, 'sequence_gap', 0)}"
                f" dequeued={getattr(source_metrics, 'dequeued_buffers', 0)}"
                f" source_discarded={getattr(source_metrics, 'discarded_buffers', 0)}"
                " source_discarded_total="
                f"{getattr(source_metrics, 'total_discarded_buffers', 0)}"
                " source_captured_total="
                f"{getattr(source_metrics, 'capture_total', 0)}"
                " handoff_discarded_total="
                f"{getattr(source_metrics, 'handoff_discarded_total', 0)}"
                " source_gap_total="
                f"{getattr(source_metrics, 'source_gap_total', 0)}"
                " requested_capture_fps="
                f"{getattr(source_metrics, 'requested_capture_fps', None)}"
                " effective_capture_fps="
                f"{getattr(source_metrics, 'effective_capture_fps', None)}"
                " capture_rate_reason="
                f"{getattr(source_metrics, 'capture_rate_reason', '') or 'unknown'}"
                f" source_error={getattr(source_metrics, 'error_reason', '') or 'none'}"
            )
        with self._camera_gui_lock:
            gui_present_count = self.camera_gui_present_count
            gui_new_count = self.camera_gui_new_frame_count
            gui_duplicate_count = self.camera_gui_duplicate_count
            gui_deadline_miss_count = self.camera_gui_deadline_miss_count
            gui_intervals = tuple(self._camera_gui_interval_sec)
            gui_new_intervals = tuple(self._camera_gui_new_interval_sec)
            display_ages = tuple(self._camera_display_age_sec)
        deadline_pct = (
            100.0 * gui_deadline_miss_count / max(1, gui_new_count - 1)
        )
        writer_detail = ""
        if recording_writer_metrics is not None:
            writer_detail = (
                " recording_submitted="
                f"{getattr(recording_writer_metrics, 'submitted_frames', 0)}"
                " recording_replaced="
                f"{getattr(recording_writer_metrics, 'replaced_frames', 0)}"
                " encoder_write_ms="
                f"{1000.0 * float(getattr(recording_writer_metrics, 'encode_duration_sec', 0.0)):.1f}"
                " encoder_write_max_ms="
                f"{1000.0 * float(getattr(recording_writer_metrics, 'max_encode_duration_sec', 0.0)):.1f}"
                " recording_error="
                f"{getattr(recording_writer_metrics, 'error_reason', '') or 'none'}"
            )
        self._emit(
            level,
            f"camera_pipeline:{mode}:{reason}",
            "camera_pipeline "
            f"reads={self.camera_read_count} "
            f"read_success={self.camera_read_success_count} "
            f"timestamp_usable={self.camera_timestamp_usable_count} "
            f"rendered={self.camera_display_frame_count} "
            f"synchronized={self.camera_synchronized_frame_count} "
            f"recorded={self.camera_recorded_frame_count} "
            f"pending={int(pending_frames)} "
            f"discarded={int(discarded_frames)} "
            f"timestamp_usable_rate_pct={usable_rate:.1f} "
            f"synchronized_rate_pct={sync_rate:.1f} "
            f"transport_age_ms={self._camera_metric_summary(self._camera_transport_age_sec)} "
            f"read_ms={self._camera_metric_summary(self._camera_read_duration_sec)} "
            f"dequeue_ms={self._camera_metric_summary(self._camera_dequeue_duration_sec)} "
            f"decode_ms={self._camera_metric_summary(self._camera_decode_duration_sec)} "
            f"tick_interval_ms={self._camera_metric_summary(self._camera_tick_interval_sec)} "
            f"tick_duration_ms={self._camera_metric_summary(self._camera_tick_duration_sec)} "
            "pre_render_ms="
            f"{self._camera_metric_summary(self._camera_pre_render_duration_sec)} "
            f"render_ms={self._camera_metric_summary(self._camera_render_duration_sec)} "
            f"writer_ms={self._camera_metric_summary(self._camera_writer_duration_sec)} "
            f"source_interval_ms={self._camera_metric_summary(self._camera_source_interval_sec)} "
            f"gui_interval_ms={self._camera_metric_summary(gui_intervals)} "
            f"gui_new_interval_ms={self._camera_metric_summary(gui_new_intervals)} "
            f"display_age_ms={self._camera_metric_summary(display_ages)} "
            f"source_gap_frames={self.camera_source_gap_frames} "
            f"gui_presents={gui_present_count} gui_new_frames={gui_new_count} "
            f"gui_duplicates={gui_duplicate_count} "
            f"gui_deadline_misses={gui_deadline_miss_count} "
            f"gui_deadline_miss_pct={deadline_pct:.1f} "
            f"target_fps={target_fps} "
            f"recording_output_frames={int(recording_output_frames)} "
            f"recording_duplicates={int(recording_duplicate_frames)} "
            f"mode={mode} timestamp_reason={timestamp_reason}"
            f"{source_detail}{writer_detail}",
            force=self.camera_read_count == 1,
        )

    def camera_frame_presented(
        self,
        *,
        new_frame: bool,
        source_timestamp_monotonic_sec: float | None,
        target_fps: float,
        now_monotonic_sec: float | None = None,
    ) -> None:
        """Record calls through HighGUI; this is not monitor scanout time."""
        now = self.clock() if now_monotonic_sec is None else float(
            now_monotonic_sec
        )
        if not math.isfinite(now):
            return
        target_interval = 1.0 / max(1.0, float(target_fps))
        with self._camera_gui_lock:
            if self._camera_previous_gui_time is not None:
                self._append_camera_metric(
                    self._camera_gui_interval_sec,
                    now - self._camera_previous_gui_time,
                )
            self._camera_previous_gui_time = now
            self.camera_gui_present_count += 1
            if not new_frame:
                self.camera_gui_duplicate_count += 1
                return
            if self._camera_previous_gui_new_time is not None:
                interval = now - self._camera_previous_gui_new_time
                self._append_camera_metric(
                    self._camera_gui_new_interval_sec, interval
                )
                if interval > target_interval + 0.002:
                    self.camera_gui_deadline_miss_count += 1
            self._camera_previous_gui_new_time = now
            self.camera_gui_new_frame_count += 1
            stamp = source_timestamp_monotonic_sec
            if stamp is not None and math.isfinite(float(stamp)):
                age = now - float(stamp)
                if age >= 0.0:
                    self._append_camera_metric(
                        self._camera_display_age_sec, age
                    )

    @staticmethod
    def _append_camera_metric(values, value) -> None:
        if value is None:
            return
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0.0:
            values.append(numeric)

    @staticmethod
    def _camera_metric_summary(values) -> str:
        if not values:
            return "unavailable"
        ordered = sorted(values)
        count = len(ordered)
        median = ordered[(count - 1) // 2]
        p95 = ordered[max(0, math.ceil(count * 0.95) - 1)]
        mean = sum(ordered) / count
        return (
            f"mean:{mean * 1000.0:.1f},p50:{median * 1000.0:.1f},"
            f"p95:{p95 * 1000.0:.1f},max:{ordered[-1] * 1000.0:.1f}"
        )

    def localization_update(
        self, state: str, detail: str, receive_count: int
    ) -> None:
        changed = state != self._last_localization_state
        self._last_localization_state = state
        level = "info" if state == "LOCALIZATION OK" else "warning"
        self._emit(
            level,
            f"localization:{state}",
            f"localization_state state={state!r} amcl_rx_count={receive_count} "
            f"detail={detail}",
            force=changed,
        )

    def state_changed(self, state: str, detail: str = "") -> None:
        if state == self.current_state:
            return
        previous = self.current_state or "STARTUP"
        self.current_state = state
        suffix = f" detail={detail}" if detail else ""
        level = "info" if state == "ACTIVE" else "warning"
        self._emit(
            level,
            f"state:{previous}->{state}:{self.path_receive_count}",
            f"viewer_state previous={previous!r} current={state!r} "
            f"plan_rx_count={self.path_receive_count}{suffix}",
            force=True,
        )

    def no_path_detail(self) -> str:
        if self.last_path_received_at is None:
            return self.last_path_outcome
        return (
            f"{self.last_path_outcome}; last_rx_monotonic="
            f"{self.last_path_received_at:.6f}"
        )
