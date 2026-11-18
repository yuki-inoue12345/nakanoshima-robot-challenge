"""Bounded camera handoff that keeps V4L2 capture off the ROS timer."""

import threading
import time
from dataclasses import replace


class LatestImageSource:
    """Continuously capture into one replaceable, timestamped slot."""

    def __init__(self, source, *, unavailable_after_sec=0.5):
        self._source = source
        self._unavailable_after_sec = float(unavailable_after_sec)
        if self._unavailable_after_sec <= 0.0:
            raise ValueError("camera unavailable timeout must be positive")
        self._condition = threading.Condition()
        self._stop = False
        self._generation = 0
        self._consumed_generation = 0
        self._latest = None
        self._latest_success_monotonic = None
        self._created_monotonic = time.monotonic()
        self._handoff_discarded_total = 0
        self._capture_total = 0
        self._source_gap_total = 0
        self._previous_capture_stamp = None
        self._worker_error = ""
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ""
        self.last_read_metrics = getattr(source, "last_read_metrics", None)
        self._latest_failure_metrics = self.last_read_metrics
        self.requested_capture_fps = getattr(
            source, "requested_capture_fps", None
        )
        self.effective_capture_fps = getattr(
            source, "effective_capture_fps", None
        )
        self.capture_rate_reason = getattr(
            source, "capture_rate_reason", "unavailable"
        )
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="ar-camera-capture",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _with_pipeline_counts(
        metrics, capture_total, discarded_total, source_interval_sec,
        source_gap_total,
    ):
        if metrics is None:
            return None
        names = getattr(metrics, "__dataclass_fields__", {})
        updates = {}
        if "capture_total" in names:
            updates["capture_total"] = int(capture_total)
        if "handoff_discarded_total" in names:
            updates["handoff_discarded_total"] = int(discarded_total)
        if "source_interval_sec" in names:
            updates["source_interval_sec"] = source_interval_sec
        if "source_gap_total" in names:
            updates["source_gap_total"] = int(source_gap_total)
        return replace(metrics, **updates) if updates else metrics

    def _capture_loop(self):
        while True:
            with self._condition:
                if self._stop:
                    return
            try:
                image = self._source.read()
                stamp = getattr(
                    self._source, "frame_timestamp_monotonic", None
                )
                kind = getattr(self._source, "frame_timestamp_kind", "")
                metrics = getattr(self._source, "last_read_metrics", None)
            except BaseException as exc:
                with self._condition:
                    self._worker_error = (
                        f"camera_capture_exception_{type(exc).__name__}"
                    )
                    self._condition.notify_all()
                return
            now = time.monotonic()
            with self._condition:
                if self._stop:
                    return
                if image is not None:
                    if self._generation > self._consumed_generation:
                        self._handoff_discarded_total += 1
                    self._capture_total += 1
                    self._generation += 1
                    sequence_gap = max(
                        0, int(getattr(metrics, "sequence_gap", 0) or 0)
                    )
                    self._source_gap_total += sequence_gap
                    source_interval = None
                    if stamp is not None and self._previous_capture_stamp is not None:
                        delta = float(stamp) - self._previous_capture_stamp
                        if delta > 0.0:
                            source_interval = delta / (sequence_gap + 1)
                    if stamp is not None:
                        self._previous_capture_stamp = float(stamp)
                    metrics = self._with_pipeline_counts(
                        metrics,
                        self._capture_total,
                        self._handoff_discarded_total,
                        source_interval,
                        self._source_gap_total,
                    )
                    # Image, SOE timestamp and metrics move as one snapshot.
                    self._latest = (image, stamp, kind, metrics)
                    self._latest_success_monotonic = now
                else:
                    self._latest_failure_metrics = metrics
                self._condition.notify_all()
            if image is None:
                time.sleep(0.005)

    @property
    def temporary_empty(self):
        with self._condition:
            if self._worker_error:
                return False
            if self._latest_success_monotonic is None:
                return (
                    time.monotonic() - self._created_monotonic
                    < self._unavailable_after_sec
                )
            return (
                time.monotonic() - self._latest_success_monotonic
                < self._unavailable_after_sec
            )

    @property
    def handoff_discarded_total(self):
        with self._condition:
            return self._handoff_discarded_total

    @property
    def worker_alive(self):
        return self._thread.is_alive()

    def read(self):
        """Return only a new frame; never replay pixels with a new time."""
        with self._condition:
            if self._generation <= self._consumed_generation:
                self.frame_timestamp_monotonic = None
                self.frame_timestamp_kind = ""
                self.last_read_metrics = self._latest_failure_metrics
                return None
            self._consumed_generation = self._generation
            image, stamp, kind, metrics = self._latest
            self.frame_timestamp_monotonic = stamp
            self.frame_timestamp_kind = kind
            self.last_read_metrics = metrics
            return image

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._source.close()
        else:
            self._worker_error = "camera_capture_shutdown_timeout"
