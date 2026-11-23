#!/usr/bin/env python3
"""Still-photo preview for an already running independent C920 ROS owner.

Never opens /dev/video*, publishes driving topics, or records video.
Only the preview process owns its window, and stopping it never stops AR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


IMAGE_TOPIC = '/koko/c920/raw_image'


def safe_prefix(prefix):
    value = re.sub(r'[^A-Za-z0-9_-]', '_', str(prefix))[:48]
    return value or 'ui_capture'


def decode_bgr(msg):
    if (msg.encoding != 'bgr8' or msg.is_bigendian or
            msg.width < 1 or msg.height < 1 or msg.width > 1920 or
            msg.height > 1080 or msg.step != msg.width * 3 or
            len(msg.data) != msg.step * msg.height):
        return None
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, 3).copy()


class SharedStillPreview(Node):
    def __init__(self, request_topic, done_topic, out_dir):
        super().__init__('koko_v8_shared_still_preview')
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos)
        self.create_subscription(String, request_topic, self._on_request, 10)
        self.done_pub = self.create_publisher(String, done_topic, 10)
        self.out_dir = out_dir
        self.latest = None
        self.last_recv = 0.0
        self.pending = None
        self.seq = 0
        self.get_logger().info('V8 photo preview: ROS image subscriber only')

    def _on_image(self, msg):
        arr = decode_bgr(msg)
        if arr is None:
            return
        self.latest = arr
        self.last_recv = time.monotonic()
        if self.pending is not None:
            request = self.pending
            self.pending = None
            self._capture(arr, request)

    def _on_request(self, msg):
        if self.pending is not None:
            self.get_logger().warning('Still capture request busy')
            return
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                data = {}
        except (TypeError, ValueError):
            data = {}
        self.pending = {
            'prefix': safe_prefix(data.get('prefix', 'ui_capture')),
            'area_id': str(data.get('area_id', 'manual'))[:32],
            'received_at': time.monotonic(),
        }

    def capture_by_key(self):
        if self.pending is None:
            self.pending = {'prefix': 'manual', 'area_id': 'manual',
                            'received_at': time.monotonic()}

    def _capture(self, frame, request):
        self.seq += 1
        path = self.out_dir / (
            f"{request['prefix']}_{time.strftime('%Y%m%d_%H%M%S')}_{self.seq:04d}.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), frame)
        data = {
            'status': 'SAVED' if ok else 'ERROR',
            'success': bool(ok),
            'image_path': str(path) if ok else '',
            'saved_path': str(path) if ok else '',
            'area_id': request['area_id'],
        }
        done = String()
        done.data = json.dumps(data, ensure_ascii=False)
        self.done_pub.publish(done)
        self.get_logger().info('Still result: '+done.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-topic', required=True)
    parser.add_argument('--done-topic', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if not args.request_topic.startswith('/') or not args.done_topic.startswith('/'):
        raise SystemExit('ROS topics must be absolute')
    # Bound file output to the V8 app's own directory, not an arbitrary request path.
    app = Path(__file__).resolve().parent
    candidate = (app / args.out_dir).resolve()
    if not candidate.is_relative_to(app.resolve()):
        raise SystemExit('Unsafe photo output directory')
    rclpy.init()
    node = SharedStillPreview(args.request_topic, args.done_topic, candidate)
    title = 'Camera V8: C920 shared preview (C/SPACE photo, Q/F10 close)'
    started = time.monotonic()
    try:
        # Match the original V7 preview window without modifying ROS frames.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(title, 1440, 810)
        cv2.moveWindow(title, 240, 120)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.035)
            now = time.monotonic()
            if now - started > 10.0 and (node.latest is None or now-node.last_recv > 10.0):
                node.get_logger().error('No AR shared image for 10s; closing preview')
                break
            if node.pending is not None and now-node.pending['received_at'] > 5:
                node.get_logger().error('Still photo request timed out without a new image')
                node.pending = None
            if node.latest is not None and now-node.last_recv < 1.5:
                cv2.imshow(title, node.latest)
                k = cv2.waitKey(1)
                if k in (ord('q'), ord('Q'), 27, 65479, 0xFFC7, 16777273):
                    break
                if k in (ord('c'), ord('C'), 32):
                    node.capture_by_key()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Normal ROS shutdown initiated by the V8 UI.
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
