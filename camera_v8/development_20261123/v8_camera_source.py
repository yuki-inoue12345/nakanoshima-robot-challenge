#!/usr/bin/env python3
"""Standalone camera owner for AR preview and on-demand V8.

This is a separate process, NOT part of the AR viewer.  It never publishes
navigation commands or modifies Nav2, Renya Launcher2, or the Follower.
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from ar_path_viewer.latest_image_source import LatestImageSource
from ar_path_viewer.v4l2_timestamp_source import V4L2TimestampCameraSource
from v8_person_frame_share import V8PersonFrameShare


class SharedCameraSource(Node):
    """Own the one C920 fd and deliver raw BGR8 to isolated ROS consumers."""

    def __init__(self, device, width=640, height=480, fps=30.0):
        super().__init__('koko_v8_camera_source')
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._previous_stamp = 0.0
        self._frames = 0
        self._share = V8PersonFrameShare()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._ar_pub = self.create_publisher(
            Image, '/koko/c920/raw_image', qos)
        self._source = LatestImageSource(V4L2TimestampCameraSource(
            device, self._width, self._height, fps=self._fps))
        self._timer = self.create_timer(1.0 / self._fps, self._tick)
        self.get_logger().info(
            'C920 owner ready: AR receives ROS; V8 receives on arrival only')

    def _tick(self):
        frame = self._source.read()
        if frame is None:
            return
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.shape != (self._height, self._width, 3)):
            return
        stamp_mono = self._source.frame_timestamp_monotonic
        source_kind = self._source.frame_timestamp_kind
        now_mono = time.monotonic()
        age = (now_mono - stamp_mono) if stamp_mono is not None else 1e9
        now_ros_ns = self.get_clock().now().nanoseconds
        if not 0 <= age <= 0.45:
            # No playback of buffered images with a newly forged stamp.
            return
        ros_stamp_ns = int(now_ros_ns - age * 1e9)
        if ros_stamp_ns <= self._previous_stamp:
            return
        self._previous_stamp = ros_stamp_ns
        msg = Image()
        msg.header.stamp.sec = ros_stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = ros_stamp_ns % 1_000_000_000
        # Preserve whether the kernel timestamp was SOE or untrusted EOF.
        msg.header.frame_id = (
            'koko_c920_v4l2_soe' if source_kind == 'v4l2_monotonic_soe'
            else 'koko_c920_unverified'
        )
        msg.height = self._height
        msg.width = self._width
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = self._width * 3
        msg.data = np.ascontiguousarray(frame).tobytes()
        # The AR feed stays alive for the entire recording session.
        self._ar_pub.publish(msg)
        # Search stream is the UNANNOTATED original and remains disabled
        # except when /person_area_arrived has activated the sibling node.
        self._share.publish_unannotated(frame, msg.header.stamp)
        self._frames += 1

    @property
    def share_node(self):
        return self._share

    def close_source(self):
        self._source.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--fps', type=float, choices=(20., 30.), default=30.)
    args = parser.parse_args()
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = SharedCameraSource(args.device, fps=args.fps)
        executor.add_node(node)
        executor.add_node(node.share_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.close_source()
            node.share_node.destroy_node()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
