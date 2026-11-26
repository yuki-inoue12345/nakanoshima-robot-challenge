from types import SimpleNamespace
import threading

import numpy as np
from sensor_msgs.msg import Image

from ar_path_viewer.pipeline_metrics import StageMetrics
from ar_path_viewer.ros_image_source import RosLatestImageSource


class _Clock:
    def __init__(self, seconds=100.0):
        self.seconds = float(seconds)

    def now(self):
        return SimpleNamespace(nanoseconds=int(self.seconds * 1e9))


class _Node:
    def __init__(self):
        self.clock = _Clock()
        self.callback = None

    def create_subscription(self, _type, _topic, callback, _qos, **_kwargs):
        self.callback = callback
        return object()

    def destroy_subscription(self, _subscription):
        pass

    def get_clock(self):
        return self.clock


def _image(value, stamp=99.98):
    msg = Image()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
    msg.header.frame_id = 'koko_c920_v4l2_soe'
    msg.height = 2
    msg.width = 3
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = 9
    msg.data = np.full((2, 3, 3), value, dtype=np.uint8).tobytes()
    return msg


def test_ros_latest_metrics_preserve_latest_pixels_and_matching_timestamp():
    node = _Node()
    source = RosLatestImageSource(node, '/camera', 3, 2)
    node.callback(_image(10, 99.96))
    node.callback(_image(20, 99.98))

    output = source.read()
    metrics = source.last_read_metrics

    assert np.all(output == 20)
    assert source.frame_timestamp_monotonic is not None
    assert source.frame_timestamp_kind == 'v4l2_monotonic_soe'
    assert metrics.received_total == 2
    assert metrics.accepted_total == 2
    assert metrics.overwritten_total == 1
    assert metrics.consumed_total == 2
    assert metrics.sequence == 2
    assert abs(metrics.transport_age_sec - 0.02) < 1e-6
    assert abs(metrics.source_interval_sec - 0.02) < 1e-6
    assert metrics.consumer_copy_duration_sec == 0.0


def test_ros_latest_transfers_owned_pixels_without_reusing_or_mixing_stamp():
    node = _Node()
    source = RosLatestImageSource(node, '/camera', 3, 2)
    node.callback(_image(31, 99.96))

    first = source.read()
    first_stamp = source.frame_timestamp_monotonic
    assert np.all(first == 31)
    assert source.read() is None

    # Mutating the consumed frame is safe: the callback creates a separate
    # owner copy for the next ROS message, and its SOE time stays paired.
    first[:] = 255
    node.callback(_image(72, 99.98))
    second = source.read()
    second_stamp = source.frame_timestamp_monotonic

    assert np.all(second == 72)
    assert second_stamp is not None and first_stamp is not None
    assert second_stamp > first_stamp


def test_ros_latest_consumer_pixels_stay_owned_while_producer_overwrites():
    """A held consumer frame must not alias a later callback buffer.

    The producer is deliberately allowed to overwrite the latest-only slot
    while the consumer repeatedly reads the already transferred ndarray.  A
    missing ownership copy or a split image/timestamp update makes this test
    observe torn pixels or a mismatched final stamp.
    """
    node = _Node()
    source = RosLatestImageSource(node, '/camera', 3, 2)
    node.callback(_image(17, 99.90))
    held = source.read()
    held_stamp = source.frame_timestamp_monotonic
    start = threading.Event()

    def produce():
        start.wait(timeout=1.0)
        for index in range(1, 201):
            node.callback(_image(index % 256, 99.90 + index * 0.0004))

    producer = threading.Thread(target=produce)
    producer.start()
    start.set()
    while producer.is_alive():
        assert np.all(held == 17)
    producer.join(timeout=1.0)
    assert not producer.is_alive()

    latest = source.read()
    latest_stamp = source.frame_timestamp_monotonic
    assert np.all(held == 17)
    assert np.all(latest == 200)
    assert held_stamp is not None and latest_stamp is not None
    assert latest_stamp > held_stamp
    assert source.last_read_metrics.sequence == 201
    assert source.last_read_metrics.overwritten_total == 199


def test_ros_latest_rejects_unverified_timestamp_for_ar_but_keeps_preview():
    node = _Node()
    source = RosLatestImageSource(node, '/camera', 3, 2)
    msg = _image(42)
    msg.header.frame_id = 'not_verified'
    node.callback(msg)

    assert np.all(source.read() == 42)
    assert source.frame_timestamp_monotonic is None
    assert source.last_read_metrics.transport_age_sec is None


def test_ros_latest_metrics_can_be_disabled_without_changing_pixels_or_stamp():
    node = _Node()
    source = RosLatestImageSource(
        node, '/camera', 3, 2, metrics_enabled=False)
    node.callback(_image(77))
    assert np.all(source.read() == 77)
    assert source.frame_timestamp_monotonic is not None
    assert source.frame_timestamp_kind == 'v4l2_monotonic_soe'
    assert source.last_read_metrics is None


def test_stage_metrics_disabled_has_no_samples_and_enabled_is_bounded():
    disabled = StageMetrics(enabled=False, max_samples=20)
    for _ in range(100):
        disabled.count('frames')
        disabled.observe('duration', 0.001)
    assert disabled.snapshot()['counters'] == {}
    assert disabled.snapshot()['samples'] == {}

    enabled = StageMetrics(max_samples=20)
    for value in range(100):
        enabled.count('frames')
        enabled.observe('duration', value / 1000.0)
    result = enabled.snapshot()
    assert result['counters']['frames'] == 100
    assert result['samples']['duration']['count'] <= 20
    assert result['samples']['duration']['p95'] > 0.0


def test_metrics_do_not_retain_image_objects():
    metrics = StageMetrics()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics.count('received')
    metrics.observe('copy_sec', 0.001)
    del frame
    snapshot = metrics.snapshot()
    assert set(snapshot) == {'interval_sec', 'counters', 'rates', 'samples'}
    assert snapshot['counters']['received'] == 1
