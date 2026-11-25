#!/usr/bin/env python3
"""Standalone camera owner for AR preview and on-demand V8.

This is a separate process, NOT part of the AR viewer.  It never publishes
navigation commands or modifies Nav2, Renya Launcher2, or the Follower.
"""
import argparse
import time

from v8_parent_death_guard import arm_parent_death_guard

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from ar_path_viewer.latest_image_source import LatestImageSource
from ar_path_viewer.pipeline_metrics import (
    ProcessResourceSampler, StageMetrics, metric_ms,
)
from ar_path_viewer.v4l2_timestamp_source import V4L2TimestampCameraSource
from v8_person_frame_share import V8PersonFrameShare


class SharedCameraSource(Node):
    """Own the one C920 fd and deliver raw BGR8 to isolated ROS consumers."""

    def __init__(self, device, width=640, height=480, fps=30.0,
                 pipeline_diagnostics=True):
        super().__init__('koko_v8_camera_source')
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._previous_stamp = 0.0
        self._frames = 0
        self._pipeline_metrics = StageMetrics(enabled=pipeline_diagnostics)
        self._resource_sampler = ProcessResourceSampler()
        self._pipeline_diagnostics = bool(pipeline_diagnostics)
        self._next_pipeline_log = time.monotonic() + 5.0
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
        tick_started = time.perf_counter()
        self._pipeline_metrics.count('timer_ticks')
        read_started = time.perf_counter()
        frame = self._source.read()
        self._pipeline_metrics.observe(
            'owner_read_sec', time.perf_counter() - read_started)
        if frame is None:
            self._pipeline_metrics.count('owner_read_empty')
            self._safe_log_pipeline()
            return
        self._pipeline_metrics.count('owner_read_success')
        source_metrics = self._source.last_read_metrics
        self._pipeline_metrics.observe(
            'source_interval_sec',
            getattr(source_metrics, 'source_interval_sec', None),
        )
        self._pipeline_metrics.observe(
            'transport_age_sec',
            getattr(source_metrics, 'transport_age_sec', None),
        )
        self._pipeline_metrics.observe(
            'dequeue_sec',
            getattr(source_metrics, 'dequeue_duration_sec', None),
        )
        self._pipeline_metrics.observe(
            'decode_sec',
            getattr(source_metrics, 'decode_duration_sec', None),
        )
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.shape != (self._height, self._width, 3)):
            self._pipeline_metrics.count('owner_frame_invalid')
            self._safe_log_pipeline()
            return
        stamp_mono = self._source.frame_timestamp_monotonic
        source_kind = self._source.frame_timestamp_kind
        now_mono = time.monotonic()
        age = (now_mono - stamp_mono) if stamp_mono is not None else 1e9
        now_ros_ns = self.get_clock().now().nanoseconds
        if not 0 <= age <= 0.45:
            # No playback of buffered images with a newly forged stamp.
            self._pipeline_metrics.count('owner_timestamp_rejected')
            self._safe_log_pipeline()
            return
        ros_stamp_ns = int(now_ros_ns - age * 1e9)
        if ros_stamp_ns <= self._previous_stamp:
            self._pipeline_metrics.count('owner_timestamp_nonmonotonic')
            self._safe_log_pipeline()
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
        serialize_started = time.perf_counter()
        msg.data = np.ascontiguousarray(frame).tobytes()
        self._pipeline_metrics.observe(
            'ar_serialize_sec', time.perf_counter() - serialize_started)
        # The AR feed stays alive for the entire recording session.
        publish_started = time.perf_counter()
        self._ar_pub.publish(msg)
        self._pipeline_metrics.observe(
            'ar_publish_sec', time.perf_counter() - publish_started)
        self._pipeline_metrics.count('ar_published')
        # Search stream is the UNANNOTATED original and remains disabled
        # except when /person_area_arrived has activated the sibling node.
        share_started = time.perf_counter()
        shared = self._share.publish_unannotated(frame, msg.header.stamp)
        self._pipeline_metrics.observe(
            'v8_share_sec', time.perf_counter() - share_started)
        if shared:
            self._pipeline_metrics.count('v8_published')
        self._frames += 1
        self._pipeline_metrics.observe(
            'owner_tick_sec', time.perf_counter() - tick_started)
        self._safe_log_pipeline()

    def _safe_log_pipeline(self):
        try:
            self._maybe_log_pipeline()
        except Exception as exc:
            # Scalar diagnostics must never interrupt C920 delivery.
            try:
                self.get_logger().warning(
                    f'camera_owner_pipeline diagnostics_failed={type(exc).__name__}'
                )
            except Exception:
                pass

    def _maybe_log_pipeline(self):
        now = time.monotonic()
        if not self._pipeline_diagnostics or now < self._next_pipeline_log:
            return
        self._next_pipeline_log = now + 5.0
        data = self._pipeline_metrics.snapshot(now=now)
        counts = data['counters']
        rates = data['rates']
        resources = self._resource_sampler.sample()
        source = self._source.last_read_metrics
        self.get_logger().info(
            'camera_owner_pipeline '
            f"interval_sec={data['interval_sec']:.3f} target_fps={self._fps:.1f} "
            f"timer_ticks={counts.get('timer_ticks', 0)} "
            f"source_capture_total={getattr(source, 'capture_total', 0)} "
            f"source_handoff_discarded_total={getattr(source, 'handoff_discarded_total', 0)} "
            f"owner_read_success={counts.get('owner_read_success', 0)} "
            f"owner_read_fps={rates.get('owner_read_success', 0.0):.2f} "
            f"ar_published={counts.get('ar_published', 0)} "
            f"ar_publish_fps={rates.get('ar_published', 0.0):.2f} "
            f"v8_active={str(self._share.active).lower()} "
            f"v8_published={counts.get('v8_published', 0)} "
            f"v8_publish_fps={rates.get('v8_published', 0.0):.2f} "
            f"transport_age_ms={1000.0 * float(getattr(source, 'transport_age_sec', 0.0) or 0.0):.2f} "
            f"source_interval_ms_p50={metric_ms(data, 'source_interval_sec', 'p50'):.3f} "
            f"source_interval_ms_p95={metric_ms(data, 'source_interval_sec'):.3f} "
            f"transport_age_ms_p50={metric_ms(data, 'transport_age_sec', 'p50'):.3f} "
            f"transport_age_ms_p95={metric_ms(data, 'transport_age_sec'):.3f} "
            f"dequeue_ms_p95={metric_ms(data, 'dequeue_sec'):.3f} "
            f"decode_ms_p95={metric_ms(data, 'decode_sec'):.3f} "
            f"read_ms_p50={metric_ms(data, 'owner_read_sec', 'p50'):.3f} "
            f"read_ms_p95={metric_ms(data, 'owner_read_sec'):.3f} "
            f"ar_serialize_ms_p50={metric_ms(data, 'ar_serialize_sec', 'p50'):.3f} "
            f"ar_serialize_ms_p95={metric_ms(data, 'ar_serialize_sec'):.3f} "
            f"ar_publish_ms_p95={metric_ms(data, 'ar_publish_sec'):.3f} "
            f"v8_share_ms_p95={metric_ms(data, 'v8_share_sec'):.3f} "
            f"tick_ms_p95={metric_ms(data, 'owner_tick_sec'):.3f} "
            f"cpu_pct={resources['cpu_pct']:.1f} rss_mib={resources['rss_mib']:.1f} "
            f"threads={resources['threads']}"
        )

    @property
    def share_node(self):
        return self._share

    def close_source(self):
        self._source.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--fps', type=float, choices=(20., 30.), default=30.)
    parser.add_argument('--expected-parent-pid', type=int, default=0)
    parser.add_argument(
        '--pipeline-diagnostics', action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.expected_parent_pid != 0:
        arm_parent_death_guard(args.expected_parent_pid)
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = SharedCameraSource(
            args.device, fps=args.fps,
            pipeline_diagnostics=args.pipeline_diagnostics,
        )
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
