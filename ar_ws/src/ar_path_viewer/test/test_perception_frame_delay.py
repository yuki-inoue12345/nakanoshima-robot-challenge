"""ROS-free adversarial tests for bounded image-time TF arrival margin."""

from types import SimpleNamespace

import pytest

from ar_path_viewer.perception_frame_delay import PerceptionFrameDelay


def _timing(sec, usable=True):
    return SimpleNamespace(monotonic_finished_sec=sec, usable=usable,
                           stamp_sec=sec - 80.0)


def test_tf_arrival_margin_keeps_original_timestamp_and_image():
    queue = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30)
    timing = _timing(100.0)
    queue.add('frame0', timing)
    assert queue.take(100.05) is None
    item = queue.take(100.12)
    assert item.image == 'frame0'
    assert item.timing is timing
    assert queue.take(100.15) is None


def test_multiple_frames_released_in_capture_order():
    q = PerceptionFrameDelay(0.10, 0.30)
    q.add('a', _timing(100.0))
    q.add('b', _timing(100.05))
    assert q.take(100.11).image == 'a'
    assert q.take(100.11) is None
    assert q.take(100.16).image == 'b'


def test_expired_frame_dropped_not_assigned_current_pose():
    q = PerceptionFrameDelay(0.12, 0.30)
    q.add('expired', _timing(1.0))
    q.add('recent', _timing(1.25))
    assert q.take(1.4).image == 'recent'
    assert q.discarded == 1


def test_invalid_or_missing_frame_flushes_pending():
    q = PerceptionFrameDelay()
    q.add('good', _timing(100.0))
    q.add('bad', _timing(100.03, usable=False))
    assert q.take(100.20) is None
    q.add('fresh', _timing(100.3))
    q.add(None, _timing(100.35))
    assert q.take(100.5) is None


def test_monotonic_clock_reset_flushes_old_epoch():
    q = PerceptionFrameDelay()
    q.add('before', _timing(200.0))
    q.add('after', _timing(1.0))
    assert q.take(1.15).image == 'after'
    assert q.take(200.5) is None


def test_no_future_frames_allowed():
    q = PerceptionFrameDelay()
    q.add('future', _timing(100.0))
    assert q.take(99.99) is None
    assert q.take(100.2) is None


def test_bounded_queue_memory_and_expiration():
    q = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30, max_frames=3)
    for i in range(10):
        q.add(str(i), _timing(100 + 0.01 * i))
    assert q.discarded == 7
    assert q.take(100.23).image == '7'


@pytest.mark.parametrize('delay,maxage,limit', [
    (0.4, 0.3, 8), (-0.1, 0.3, 8), (0.12, 0.5, 8),
    (0.1, 0.3, 1), (float('nan'), 0.3, 8),
])
def test_invalid_delay_configuration_rejected(delay, maxage, limit):
    with pytest.raises(ValueError):
        PerceptionFrameDelay(delay, maxage, limit)
