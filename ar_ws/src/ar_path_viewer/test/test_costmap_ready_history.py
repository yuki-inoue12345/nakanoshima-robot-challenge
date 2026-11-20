"""Unit-level contract for camera-first Costmap selection."""
from types import SimpleNamespace

from ar_path_viewer.costmap_ready_history import CostmapReadyHistory


def ready(stamp, frame="robot/odom"):
    return SimpleNamespace(
        snapshot=SimpleNamespace(stamp_sec=stamp, frame_id=frame)
    )


def choose(history, stamp, now=100.5):
    return history.for_image(
        stamp, now,
        maximum_image_age_sec=0.7,
        maximum_ros_age_sec=1.5,
    )


def test_latest_costmap_newer_than_camera_uses_previous_exact_ready():
    history = CostmapReadyHistory()
    older, newer = ready(100.0), ready(100.6)
    assert history.add(older)
    assert history.add(newer)
    assert choose(history, 100.58, 100.65) is older
    assert choose(history, 100.65, 100.68) is newer


def test_no_future_map_and_no_fake_extension():
    history = CostmapReadyHistory()
    history.add(ready(100.6))
    assert choose(history, 100.58, 100.65) is None
    assert choose(history, 101.4, 101.4) is None


def test_exact_epoch_transition_drops_other_frame():
    history = CostmapReadyHistory()
    history.add(ready(100.0))
    other = ready(100.6, "map")
    assert history.add(other)
    assert history.count == 1
    assert choose(history, 100.5, 100.6) is None
    assert choose(history, 100.65, 100.7) is other


def test_out_of_order_and_duplicates_do_not_revive_history():
    history = CostmapReadyHistory()
    fresh = ready(100.6)
    assert history.add(fresh)
    assert not history.add(ready(100.5))
    assert not history.add(ready(100.6))
    assert history.count == 1


def test_clock_rollback_purges_ready_items():
    history = CostmapReadyHistory()
    history.add(ready(100.6))
    assert choose(history, 100.8, 99.5) is None
    assert history.count == 0


def test_history_capacity_bounded():
    history = CostmapReadyHistory(capacity=4)
    for i in range(30):
        history.add(ready(100.0 + i * 0.1))
    assert history.count == 4
