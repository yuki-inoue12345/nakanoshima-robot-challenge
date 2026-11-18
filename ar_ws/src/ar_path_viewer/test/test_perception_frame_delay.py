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


def test_peek_retries_same_frame_without_changing_capture_timestamp():
    queue = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30)
    timing = _timing(100.0)
    queue.add('frame0', timing)

    first = queue.peek(100.12)
    second = queue.peek(100.16)

    assert first is second
    assert second.timing is timing
    assert queue.pending_count == 1
    assert queue.consume(second)
    assert queue.pending_count == 0


def test_20fps_frames_wait_for_30hz_tf_without_timestamp_replacement():
    """Model TF arriving 28 ms after each image timestamp."""
    queue = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30)
    released = []
    latest_tf_stamp = 0.0
    now = 100.0
    for index in range(20):
        capture = 100.0 + index * 0.05
        queue.add(index, _timing(capture))
        now = capture + 0.12
        # 30 Hz TF has advanced past the original exposure by the time the
        # delayed frame becomes eligible.
        latest_tf_stamp = max(latest_tf_stamp, now - 0.028)
        item = queue.peek(now)
        if item is not None and latest_tf_stamp >= item.timing.stamp_sec:
            released.append((item.image, item.timing.stamp_sec))
            queue.consume(item)

    assert released
    assert all(stamp == _timing(100.0 + index * 0.05).stamp_sec
               for index, stamp in released)
    assert all(released[i][1] < released[i + 1][1]
               for i in range(len(released) - 1))


def test_30fps_queue_holds_four_frames_for_120ms_without_overflow():
    queue = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30)
    frames = []
    for index in range(5):
        timing = _timing(100.0 + index / 30.0)
        image = object()
        frames.append((image, timing))
        queue.add(image, timing)

    assert queue.pending_count == 5
    released = queue.peek(100.12)
    assert released.image is frames[0][0]
    assert released.timing is frames[0][1]
    assert queue.discarded == 0


def test_retry_still_expires_when_tf_never_arrives():
    queue = PerceptionFrameDelay(delay_sec=0.12, max_age_sec=0.30)
    queue.add('frame0', _timing(100.0))
    assert queue.peek(100.12).image == 'frame0'
    assert queue.peek(100.31) is None
    assert queue.pending_count == 0
    assert queue.discarded == 1


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


def test_latest_eligible_favors_fresh_exposure_without_timestamp_rewrite():
    queue = PerceptionFrameDelay(0.12, 0.30)
    for index in range(6):
        queue.add(index, _timing(100.0 + index * 0.03))
    selected = queue.peek_latest_eligible(100.25)
    assert selected.image == 4
    assert selected.timing is not None
    assert selected.timing.monotonic_finished_sec == pytest.approx(100.12)
    assert queue.discarded == 4
    assert queue.consume(selected)
    assert queue.peek_latest_eligible(100.25) is None


def test_latest_eligible_never_uses_unmatured_or_expired_frame():
    queue = PerceptionFrameDelay(0.12, 0.30)
    queue.add("old", _timing(100.0))
    queue.add("new", _timing(100.20))
    selected = queue.peek_latest_eligible(100.25)
    assert selected.image == "old"
    assert queue.consume(selected)
    assert queue.peek_latest_eligible(100.25) is None
    assert queue.peek_latest_eligible(100.51) is None
