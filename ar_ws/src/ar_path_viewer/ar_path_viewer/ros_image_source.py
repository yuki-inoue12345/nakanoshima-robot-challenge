"""Read-only, latest-only ROS camera input for AR Viewer.

This module never opens a camera and never creates a ROS publisher.
A camera node in another PROCESS owns V4L2 and publishes verified frames.
"""
import math
import threading
import time
from dataclasses import dataclass

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import Image


@dataclass(frozen=True)
class RosImageReadMetrics:
    frame_timestamp_monotonic_sec: float | None
    transport_age_sec: float | None
    source_interval_sec: float | None
    read_duration_sec: float
    callback_duration_sec: float
    callback_copy_duration_sec: float
    consumer_copy_duration_sec: float
    received_total: int
    accepted_total: int
    rejected_total: int
    overwritten_total: int
    consumed_total: int
    sequence: int
    sequence_gap: int = 0
    dequeued_buffers: int = 1
    discarded_buffers: int = 0
    total_discarded_buffers: int = 0
    capture_total: int = 0
    handoff_discarded_total: int = 0
    source_gap_total: int = 0
    dequeue_monotonic_sec: float | None = None
    dequeue_duration_sec: float = 0.0
    decode_duration_sec: float = 0.0
    requested_capture_fps: float | None = None
    effective_capture_fps: float | None = None
    capture_rate_reason: str = "ros_external_v4l2_timestamp"
    error_reason: str = ""


class RosLatestImageSource:
    """Non-blocking source with a strict frame-age/timestamp contract.

    All incoming images use ROS time in header.stamp.  The camera source must
    label kernel V4L2 SOE timestamps with a dedicated frame_id.  Because both
    processes run on the SAME NUC, a host ROS/monotonic pair can reconstruct a
    local monotonic source time for the existing AR Costmap safety gate.
    Missing/unverified stamps are NEVER silently upgraded to verified stamps.
    """

    def __init__(self, node, topic, width, height, *, max_age_sec=0.45,
                 metrics_enabled=True):
        if not isinstance(topic, str) or not topic.startswith('/'):
            raise ValueError('ROS camera input needs an absolute topic')
        self._node = node
        self._width = int(width)
        self._height = int(height)
        self._max_age_sec = float(max_age_sec)
        self._metrics_enabled = bool(metrics_enabled)
        if (not math.isfinite(self._max_age_sec) or
                not 0.05 <= self._max_age_sec <= 1.0):
            raise ValueError('invalid ROS camera max frame age')
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._latest = None
        self._received_at = None
        self._generation = 0
        self._consumed = 0
        self._closed = False
        self._received_total = 0
        self._accepted_total = 0
        self._rejected_total = 0
        self._overwritten_total = 0
        self._last_ros_stamp = None
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ''
        self.last_read_metrics = None
        self.requested_capture_fps = None
        self.effective_capture_fps = None
        self.capture_rate_reason = 'ros_external_v4l2_timestamp'
        self.worker_alive = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # The render timer uses the default mutually-exclusive group.
        # The camera callback must still receive fresh frames while rendering.
        self._camera_group = ReentrantCallbackGroup()
        self._subscription = node.create_subscription(
            Image, topic, self._on_image, qos,
            callback_group=self._camera_group
        )

    def _on_image(self, msg):
        callback_started = time.perf_counter() if self._metrics_enabled else 0.0
        if self._metrics_enabled:
            with self._lock:
                self._received_total += 1
        if (msg.encoding != 'bgr8' or bool(msg.is_bigendian) or
                int(msg.width) != self._width or
                int(msg.height) != self._height or
                int(msg.step) != self._width * 3 or
                len(msg.data) != self._height * self._width * 3):
            if self._metrics_enabled:
                with self._lock:
                    self._rejected_total += 1
            return
        try:
            copy_started = time.perf_counter() if self._metrics_enabled else 0.0
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                self._height, self._width, 3).copy()
            copy_duration = (
                time.perf_counter() - copy_started
                if self._metrics_enabled else 0.0
            )
            stamp = (float(msg.header.stamp.sec) +
                     1e-9 * float(msg.header.stamp.nanosec))
            recv_mono = time.monotonic()
            recv_ros = float(self._node.get_clock().now().nanoseconds) * 1e-9
            age = recv_ros - stamp
            verified = (
                msg.header.frame_id == 'koko_c920_v4l2_soe'
                and stamp > 0.0 and math.isfinite(stamp)
                and 0 <= age <= self._max_age_sec
            )
            # For an unverified timestamp, still show raw pixels.  Costmap
            # and navigation overlays will fail closed via viewer_node timing.
            mono_stamp = recv_mono - age if verified else None
            kind = 'v4l2_monotonic_soe' if verified else ''
        except (ValueError, TypeError, BufferError, OverflowError):
            if self._metrics_enabled:
                with self._lock:
                    self._rejected_total += 1
            return
        callback_duration = (
            time.perf_counter() - callback_started
            if self._metrics_enabled else 0.0
        )
        with self._lock:
            if self._closed:
                return
            if self._metrics_enabled and self._generation > self._consumed:
                self._overwritten_total += 1
            self._generation += 1
            source_interval = None
            if (self._metrics_enabled and verified
                    and self._last_ros_stamp is not None
                    and stamp > self._last_ros_stamp):
                source_interval = stamp - self._last_ros_stamp
            if self._metrics_enabled and verified:
                self._last_ros_stamp = stamp
            if self._metrics_enabled:
                self._accepted_total += 1
            self._latest = (
                image, mono_stamp, kind, age if verified else None,
                source_interval, callback_duration, copy_duration,
            )
            self._received_at = time.monotonic()

    @property
    def temporary_empty(self):
        with self._lock:
            since = (self._received_at if self._received_at is not None
                     else self._started)
            return not self._closed and time.monotonic() - since <= 0.6

    def read(self):
        read_started = time.perf_counter() if self._metrics_enabled else 0.0
        with self._lock:
            self.frame_timestamp_monotonic = None
            self.frame_timestamp_kind = ''
            if (self._closed or self._latest is None
                    or self._generation <= self._consumed):
                return None
            if (self._received_at is None or
                    time.monotonic() - self._received_at > self._max_age_sec):
                return None
            self._consumed = self._generation
            (frame, stamp, kind, transport_age, source_interval,
             callback_duration, callback_copy_duration) = self._latest
            self.frame_timestamp_monotonic = stamp
            self.frame_timestamp_kind = kind
            generation = self._generation
            received_total = self._received_total
            accepted_total = self._accepted_total
            rejected_total = self._rejected_total
            overwritten_total = self._overwritten_total
            consumed_total = self._consumed
        copy_started = time.perf_counter() if self._metrics_enabled else 0.0
        output = frame.copy()
        if self._metrics_enabled:
            consumer_copy_duration = time.perf_counter() - copy_started
            self.last_read_metrics = RosImageReadMetrics(
                frame_timestamp_monotonic_sec=stamp,
                transport_age_sec=transport_age,
                source_interval_sec=source_interval,
                read_duration_sec=time.perf_counter() - read_started,
                callback_duration_sec=callback_duration,
                callback_copy_duration_sec=callback_copy_duration,
                consumer_copy_duration_sec=consumer_copy_duration,
                received_total=received_total,
                accepted_total=accepted_total,
                rejected_total=rejected_total,
                overwritten_total=overwritten_total,
                consumed_total=consumed_total,
                sequence=generation,
                capture_total=accepted_total,
                handoff_discarded_total=overwritten_total,
                total_discarded_buffers=overwritten_total,
            )
        return output

    def close(self):
        with self._lock:
            self._closed = True
            self._latest = None
        self._node.destroy_subscription(self._subscription)
