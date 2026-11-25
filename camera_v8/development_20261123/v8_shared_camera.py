"""Search-only, read-only ROS image intake for Camera V8.

One AR-owned camera provides UNANNOTATED BGR frames on a ROS topic only
while a person-area search is active. V8 NEVER opens a /dev/video device.
No video recording is implemented here. Photo evidence is kept by MS2.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ar_path_viewer.pipeline_metrics import (
    ProcessResourceSampler, StageMetrics, metric_ms,
)
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class RosFrameFeed:
    """One ROS subscriber, latest frame only; bounded RAM use, no video disk I/O."""

    def __init__(self, node, topic: str = '/koko/person_search/raw_image',
                 max_staleness_sec: float = 0.75,
                 pipeline_diagnostics: bool = True,
                 process_every: int = 5):
        self._node = node
        self._condition = threading.Condition()
        self._last_frame: Optional[np.ndarray] = None
        self._last_monotonic = 0.0
        self._sequence = 0
        self._last_consumed_sequence = 0
        self._closed = False
        self._max_staleness = float(max_staleness_sec)
        self._last_header_stamp = None
        self._metrics = StageMetrics(enabled=pipeline_diagnostics)
        self._resource_sampler = ProcessResourceSampler()
        self._pipeline_diagnostics = bool(pipeline_diagnostics)
        self._search_active = False
        self._process_every = max(1, int(process_every))
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self._subscription = node.create_subscription(Image, topic, self._on_frame, qos)
        self._diagnostic_timer = node.create_timer(5.0, self._safe_log_pipeline)
        node.get_logger().info(f'V8 receives unannotated frames ONLY from {topic}; camera is never opened')

    def _on_frame(self, message: Image) -> None:
        callback_started = time.perf_counter()
        self._metrics.count('received')
        try:
            if (message.encoding != 'bgr8' or message.is_bigendian or
                    message.width <= 0 or message.height <= 0 or
                    message.width > 1920 or message.height > 1080 or
                    message.step != int(message.width) * 3 or
                    len(message.data) != int(message.step) * int(message.height)):
                self._metrics.count('rejected')
                return
            copy_started = time.perf_counter()
            frame = np.frombuffer(message.data, dtype=np.uint8).reshape(
                int(message.height), int(message.width), 3).copy()
            self._metrics.observe(
                'callback_copy_sec', time.perf_counter() - copy_started)
            now = time.monotonic()
            stamp = (float(message.header.stamp.sec) +
                     1e-9 * float(message.header.stamp.nanosec))
            recv_ros = float(self._node.get_clock().now().nanoseconds) * 1e-9
            if stamp > 0.0:
                self._metrics.observe('transport_age_sec', recv_ros - stamp)
                if self._last_header_stamp is not None and stamp > self._last_header_stamp:
                    self._metrics.observe('source_interval_sec', stamp - self._last_header_stamp)
                self._last_header_stamp = stamp
            with self._condition:
                if self._closed:
                    return
                if self._sequence > self._last_consumed_sequence:
                    self._metrics.count('overwritten')
                self._last_frame = frame
                self._last_monotonic = now
                self._sequence += 1
                self._metrics.count('accepted')
                self._condition.notify_all()
        except (ValueError, TypeError, BufferError):
            self._metrics.count('rejected')
            return
        finally:
            self._metrics.observe(
                'callback_sec', time.perf_counter() - callback_started)

    def set_search_active(self, active: bool) -> None:
        self._search_active = bool(active)

    def record_model_stage(self, stage: str, duration_sec: float) -> None:
        stage = str(stage).lower()
        if stage not in ('yolo', 'vlm'):
            return
        self._metrics.count(f'{stage}_calls')
        self._metrics.observe(f'{stage}_sec', duration_sec)

    def _log_pipeline(self) -> None:
        if not self._pipeline_diagnostics:
            return
        data = self._metrics.snapshot()
        counts = data['counters']
        rates = data['rates']
        resources = self._resource_sampler.sample()
        self._node.get_logger().info(
            'v8_image_pipeline '
            f"interval_sec={data['interval_sec']:.3f} "
            f"search_active={str(self._search_active).lower()} "
            f"process_every={self._process_every} "
            'camera_fps_bookkeeping=15.0 camera_fps_is_measured=false '
            f"received={counts.get('received', 0)} "
            f"receive_fps={rates.get('received', 0.0):.2f} "
            f"accepted={counts.get('accepted', 0)} rejected={counts.get('rejected', 0)} "
            f"overwritten={counts.get('overwritten', 0)} "
            f"read_requests={counts.get('read_requests', 0)} "
            f"consumed={counts.get('consumed', 0)} "
            f"consume_fps={rates.get('consumed', 0.0):.2f} "
            f"read_timeouts={counts.get('read_timeouts', 0)} "
            f"yolo_calls={counts.get('yolo_calls', 0)} "
            f"yolo_fps={rates.get('yolo_calls', 0.0):.2f} "
            f"vlm_calls={counts.get('vlm_calls', 0)} "
            f"callback_ms_p50={metric_ms(data, 'callback_sec', 'p50'):.3f} "
            f"callback_ms_p95={metric_ms(data, 'callback_sec'):.3f} "
            f"copy_ms_p95={metric_ms(data, 'callback_copy_sec'):.3f} "
            f"reader_copy_ms_p95={metric_ms(data, 'reader_copy_sec'):.3f} "
            f"reader_wait_ms_p95={metric_ms(data, 'reader_wait_sec'):.3f} "
            f"transport_age_ms_p95={metric_ms(data, 'transport_age_sec'):.3f} "
            f"source_interval_ms_p95={metric_ms(data, 'source_interval_sec'):.3f} "
            f"yolo_ms_p50={metric_ms(data, 'yolo_sec', 'p50'):.3f} "
            f"yolo_ms_p95={metric_ms(data, 'yolo_sec'):.3f} "
            f"vlm_ms_p50={metric_ms(data, 'vlm_sec', 'p50'):.3f} "
            f"vlm_ms_p95={metric_ms(data, 'vlm_sec'):.3f} "
            f"cpu_pct={resources['cpu_pct']:.1f} rss_mib={resources['rss_mib']:.1f} "
            f"threads={resources['threads']}"
        )

    def _safe_log_pipeline(self) -> None:
        try:
            self._log_pipeline()
        except Exception as exc:
            # A diagnostic snapshot/logger failure cannot stop person search.
            try:
                self._node.get_logger().warning(
                    f'v8_image_pipeline diagnostics_failed={type(exc).__name__}'
                )
            except Exception:
                pass

    def has_recent_frame(self, after_monotonic: float = 0.0) -> bool:
        with self._condition:
            age = time.monotonic() - self._last_monotonic
            return (not self._closed and self._last_frame is not None and
                    self._last_monotonic >= after_monotonic and
                    0 <= age <= self._max_staleness)

    def wait_for_fresh_frame(self, *, since: float, timeout: float = 8.0,
                             stop_event: Optional[threading.Event] = None) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed and not (stop_event and stop_event.is_set()):
                age = time.monotonic() - self._last_monotonic
                if (self._last_frame is not None and self._last_monotonic >= since
                        and 0 <= age <= self._max_staleness):
                    return True
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                self._condition.wait(min(0.2, remain))
            return False

    def new_reader(self):
        with self._condition:
            return RosFrameReader(self, self._sequence)

    def close(self):
        with self._condition:
            self._closed = True
            self._last_frame = None
            self._condition.notify_all()


class RosFrameReader:
    """cv2.VideoCapture-like reader supported by the UNCHANGED V7 MS2 engine."""

    def __init__(self, feed: RosFrameFeed, initial_sequence: int):
        self._feed = feed
        self._sequence = initial_sequence
        self._released = False
        self.source_label = 'ar_ros_unannotated_camera'

    def read(self):
        started = time.perf_counter()
        self._feed._metrics.count('read_requests')
        if self._released:
            return False, None
        deadline = time.monotonic() + 0.5
        with self._feed._condition:
            while not self._released and not self._feed._closed:
                age = time.monotonic() - self._feed._last_monotonic
                if (self._feed._sequence > self._sequence and
                    self._feed._last_frame is not None and
                    0 <= age <= self._feed._max_staleness):
                    skipped = max(0, self._feed._sequence - self._sequence - 1)
                    self._sequence = self._feed._sequence
                    self._feed._last_consumed_sequence = self._sequence
                    copy_started = time.perf_counter()
                    output = self._feed._last_frame.copy()
                    self._feed._metrics.observe(
                        'reader_copy_sec', time.perf_counter() - copy_started)
                    self._feed._metrics.observe(
                        'reader_wait_sec', time.perf_counter() - started)
                    self._feed._metrics.count('reader_skipped', skipped)
                    self._feed._metrics.count('consumed')
                    return True, output
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._feed._condition.wait(min(0.1, remaining))
        self._feed._metrics.observe(
            'reader_wait_sec', time.perf_counter() - started)
        self._feed._metrics.count('read_timeouts')
        return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 15.0  # only for MS2 frame-time bookkeeping, not measured capture FPS
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._sequence)
        return 0.0

    def set(self, _prop, _value):
        return False

    def isOpened(self):
        return not self._released and not self._feed._closed

    def release(self):
        self._released = True
        with self._feed._condition:
            self._feed._condition.notify_all()


class V8FrameCoordinator:
    """Admission gate.  Never owns camera, resumes camera, or changes Nav2."""

    def __init__(self, feed: RosFrameFeed, shutdown_event: threading.Event):
        self.feed = feed
        self._shutdown = shutdown_event
        self._admitted = False

    def pause_for_search(self, timeout_sec: float = 8.0) -> bool:
        self._admitted = False
        # AR can start publishing slightly before V8's arrival callback.
        since = time.monotonic() - 0.5
        self._admitted = self.feed.wait_for_fresh_frame(
            since=since, timeout=timeout_sec, stop_event=self._shutdown)
        self.feed.set_search_active(self._admitted)
        return self._admitted

    def create_search_reader(self):
        if not self._admitted or self._shutdown.is_set() or not self.feed.has_recent_frame():
            raise RuntimeError('AR raw-frame stream unavailable or stale; V8 search denied')
        return self.feed.new_reader()

    def resume_after_search(self):
        self._admitted = False
        self.feed.set_search_active(False)
        return True

    def shutdown(self, timeout_sec: float = 8.0):
        self._shutdown.set()
        self._admitted = False
        self.feed.set_search_active(False)
        self.feed.close()
        return True


class V8ModelStageDiagnostics(AbstractContextManager):
    """Time unchanged V7 YOLO/VLM calls without changing their decisions."""

    _VLM_NAMES = ('call_person_vlm_for_candidate', 'call_front_photo_vlm')

    def __init__(self, ms2_module, feed: RosFrameFeed, *, enabled: bool = True):
        self._module = ms2_module
        self._feed = feed
        self._enabled = bool(enabled)
        self._originals = []

    def _wrap(self, owner, name: str, stage: str) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            return

        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._feed.record_model_stage(
                    stage, time.perf_counter() - started)

        self._originals.append((owner, name, original))
        setattr(owner, name, measured)

    def __enter__(self):
        if not self._enabled:
            return self
        base = getattr(self._module, 'base', None)
        if base is not None:
            self._wrap(base, 'detect_persons_with_tracking', 'yolo')
        for name in self._VLM_NAMES:
            self._wrap(self._module, name, 'vlm')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        return False
