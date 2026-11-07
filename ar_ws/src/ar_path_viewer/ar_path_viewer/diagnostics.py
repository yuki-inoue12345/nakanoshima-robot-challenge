"""Rate-limited diagnostics for Path reception and viewer state transitions."""

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

    def camera_configuration(self, detail: str) -> None:
        """Record the effective projection model once for field-log correlation."""
        self._emit(
            "info",
            "camera_configuration",
            f"camera_configuration {detail}",
            force=True,
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
