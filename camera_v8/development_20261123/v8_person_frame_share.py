"""Separate display-only camera-frame publisher node; NEVER commands Nav2.

The AR Viewer itself remains subscriber-only. The sibling bridge publishes
unannotated BGR camera frames ONLY after a fresh person_area arrival event and
stops on a terminal V8 detail, a false arrival, or a bounded timeout.
"""
from __future__ import annotations

import json
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class V8PersonFrameShare(Node):
    def __init__(self, *, max_active_sec=180.0):
        super().__init__('koko_ar_v8_camera_frame_share')
        self._active = False
        self._started = 0.0
        self._max_active = float(max_active_sec)
        self._frame_count = 0
        # VOLATILE avoids replaying an old latched true on AR restart.
        arrival_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.VOLATILE)
        detail_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Bool, '/person_area_arrived', self._on_arrival, arrival_qos)
        self.create_subscription(String, '/person_search_done_detail', self._on_result, detail_qos)
        # Bool is also a terminal publication from the unchanged V8 engine.
        # It is useful if a large detail message is delayed or not delivered.
        self.create_subscription(Bool, '/person_search_done', self._on_done, detail_qos)
        self._pub = self.create_publisher(Image, '/koko/person_search/raw_image', image_qos)
        self.get_logger().info('V8 camera share ready: publishes UNANNOTATED pixels ONLY during arrival search')

    def _on_arrival(self, msg):
        if msg.data:
            self._active = True
            self._started = time.monotonic()
            self._frame_count = 0
            self.get_logger().info('V8 raw frames enabled by person_area arrival')
        else:
            if self._active:
                self.get_logger().info(
                    f'V8 raw frames stopped at false arrival; sent={self._frame_count}'
                )
            self._active = False

    def _on_done(self, msg):
        # The existing V8 server publishes done only for terminal results.
        # False means unsafe to advance; camera publication still stops.
        if not self._active:
            return
        self._active = False
        self.get_logger().info(
            f'V8 raw frames stopped at terminal done; sent={self._frame_count}; '
            f'safe_to_advance={bool(msg.data)}'
        )

    def _on_result(self, msg):
        if not self._active:
            return
        try:
            detail = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        # BUSY and other non-terminal diagnostics MUST NOT terminate capture.
        if isinstance(detail, dict) and detail.get('terminal') is True:
            self._active = False
            self.get_logger().info(f'V8 raw frames stopped at terminal result; sent={self._frame_count}')

    def publish_unannotated(self, bgr, stamp):
        if not self._active:
            return False
        if time.monotonic() - self._started > self._max_active:
            self._active = False
            self.get_logger().error('V8 raw-frame share deadline exceeded; stopped (no navigation action)')
            return False
        if self._pub.get_subscription_count() < 1:
            return False
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or
                bgr.ndim != 3 or bgr.shape[2] != 3):
            return False
        frame = np.ascontiguousarray(bgr)
        h, w = frame.shape[:2]
        if not 0 < h <= 1080 or not 0 < w <= 1920:
            return False
        msg = Image()
        msg.header.stamp = stamp  # ROS processing time; not claimed exposure time
        msg.header.frame_id = 'koko_c920_raw_image'
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()
        self._pub.publish(msg)
        self._frame_count += 1
        return True

    @property
    def active(self):
        return bool(self._active)

    @property
    def frame_count(self):
        return int(self._frame_count)
