"""Search-only, read-only ROS image intake for Camera V8.

One AR-owned camera provides UNANNOTATED BGR frames on a ROS topic only
while a person-area search is active. V8 NEVER opens a /dev/video device.
No video recording is implemented here. Photo evidence is kept by MS2.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class RosFrameFeed:
    """One ROS subscriber, latest frame only; bounded RAM use, no video disk I/O."""

    def __init__(self, node, topic: str = '/koko/person_search/raw_image',
                 max_staleness_sec: float = 0.75):
        self._node = node
        self._condition = threading.Condition()
        self._last_frame: Optional[np.ndarray] = None
        self._last_monotonic = 0.0
        self._sequence = 0
        self._closed = False
        self._max_staleness = float(max_staleness_sec)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self._subscription = node.create_subscription(Image, topic, self._on_frame, qos)
        node.get_logger().info(f'V8 receives unannotated frames ONLY from {topic}; camera is never opened')

    def _on_frame(self, message: Image) -> None:
        try:
            if (message.encoding != 'bgr8' or message.is_bigendian or
                    message.width <= 0 or message.height <= 0 or
                    message.width > 1920 or message.height > 1080 or
                    message.step != int(message.width) * 3 or
                    len(message.data) != int(message.step) * int(message.height)):
                return
            frame = np.frombuffer(message.data, dtype=np.uint8).reshape(
                int(message.height), int(message.width), 3).copy()
            now = time.monotonic()
            with self._condition:
                if self._closed:
                    return
                self._last_frame = frame
                self._last_monotonic = now
                self._sequence += 1
                self._condition.notify_all()
        except (ValueError, TypeError, BufferError):
            return

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
        if self._released:
            return False, None
        deadline = time.monotonic() + 0.5
        with self._feed._condition:
            while not self._released and not self._feed._closed:
                age = time.monotonic() - self._feed._last_monotonic
                if (self._feed._sequence > self._sequence and
                    self._feed._last_frame is not None and
                    0 <= age <= self._feed._max_staleness):
                    self._sequence = self._feed._sequence
                    return True, self._feed._last_frame.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._feed._condition.wait(min(0.1, remaining))
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
        return self._admitted

    def create_search_reader(self):
        if not self._admitted or self._shutdown.is_set() or not self.feed.has_recent_frame():
            raise RuntimeError('AR raw-frame stream unavailable or stale; V8 search denied')
        return self.feed.new_reader()

    def resume_after_search(self):
        self._admitted = False
        return True

    def shutdown(self, timeout_sec: float = 8.0):
        self._shutdown.set()
        self._admitted = False
        self.feed.close()
        return True
