"""P1-③: a delayed Costmap may not borrow a later /plan or goal epoch."""
from types import SimpleNamespace

import numpy as np
import pytest

from ar_path_viewer.path_adapter import PathStore
from ar_path_viewer.costmap_path_sync import (
    capture_costmap_path, check_costmap_path_pair,
    follower_goal_identity,
)


GOAL_A = ("a" * 64, 2, 22, "to_pose", 4)
GOAL_B = ("a" * 64, 3, 23, "to_pose", 5)


def _store():
    return PathStore(clock=lambda: 100.0)


def _path(store, stamp=100.0):
    return store.replace(
        np.array(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))), "map",
        header_stamp_sec=stamp, received_ros_sec=100.0,
    )


def _check(link, path, goal=GOAL_A, costmap_stamp=100.1, frame="robot/odom"):
    return check_costmap_path_pair(
        link, costmap_stamp_sec=costmap_stamp,
        costmap_frame_id=frame, current_path=path,
        current_follower_identity=goal,
    )


def test_path_snapshot_preserves_header_stamp_and_receipt_ros_time():
    store = _store()
    path = _path(store)
    assert path.header_stamp_sec == 100.0
    assert path.received_ros_sec == 100.0
    assert path.generation == 1


def test_frozen_path_is_used_when_goal_and_path_generation_unchanged():
    store = _store()
    path = _path(store)
    pair = capture_costmap_path(100.1, "robot/odom", path, GOAL_A)
    accepted, reason = _check(pair, store.snapshot())
    assert accepted is path
    assert reason == "path_costmap_pair_matched"


def test_delayed_tf_can_never_compare_old_costmap_to_new_plan():
    store = _store()
    old = _path(store)
    pair = capture_costmap_path(100.1, "robot/odom", old, GOAL_A)
    new = store.replace(np.array(((0., 2., 0.),)), "map", header_stamp_sec=100.15)
    accepted, reason = _check(pair, new)
    assert accepted is None
    assert reason == "real_path_generation_changed"
    assert pair.path is old
    assert np.array_equal(old.points[0], (1., 0., 0.))


def test_goal_change_without_any_path_update_still_suppresses_alert():
    store = _store()
    path = _path(store)
    pair = capture_costmap_path(100.1, "robot/odom", path, GOAL_A)
    accepted, reason = _check(pair, path, goal=GOAL_B)
    assert accepted is None
    assert reason == "follower_goal_generation_changed"


def test_new_goal_and_plan_require_new_costmap_binding():
    store = _store()
    old = _path(store)
    previous = capture_costmap_path(100.1, "robot/odom", old, GOAL_A)
    newer = store.replace(np.array(((0., 2., 0.),)), "map", header_stamp_sec=100.2)
    assert _check(previous, newer, goal=GOAL_B)[0] is None
    fresh = capture_costmap_path(100.25, "robot/odom", newer, GOAL_B)
    assert _check(fresh, newer, goal=GOAL_B, costmap_stamp=100.25)[0] is newer


def test_new_path_arrives_before_costmap_but_header_stamp_is_newer():
    store = _store()
    newer = _path(store, stamp=101.0)
    binding = capture_costmap_path(100.1, "robot/odom", newer, GOAL_A)
    accepted, reason = _check(binding, newer)
    assert accepted is None
    assert "stamp_mismatch" in reason


def test_old_path_cannot_be_used_with_far_newer_costmap():
    store = _store()
    old = _path(store, stamp=95.0)
    binding = capture_costmap_path(100.1, "robot/odom", old, GOAL_A)
    assert _check(binding, old)[0] is None


@pytest.mark.parametrize('bad_stamp', [None, float('nan'), float('inf')])
def test_unknown_header_stamp_disables_path_alert_not_path_display(bad_stamp):
    store = _store()
    if bad_stamp is None:
        path = store.replace(np.array(((1., 0., 0.),)), "map")
    else:
        # Invalid stamps are rejected at the PathStore boundary.
        with pytest.raises(ValueError):
            store.replace(np.array(((1., 0., 0.),)), "map", header_stamp_sec=bad_stamp)
        return
    binding = capture_costmap_path(100.1, "robot/odom", path, GOAL_A)
    assert _check(binding, path)[1] == "real_path_header_stamp_unknown"
    assert store.state()[0] == "ACTIVE"


@pytest.mark.parametrize('kind', ['costmap_stamp', 'costmap_frame', 'missing_binding'])
def test_costmap_pair_rejects_wrong_occupancygrid(kind):
    store = _store()
    path = _path(store)
    binding = capture_costmap_path(100.1, "robot/odom", path, GOAL_A)
    if kind == 'costmap_stamp':
        result = _check(binding, path, costmap_stamp=100.2)
    elif kind == 'costmap_frame':
        result = _check(binding, path, frame='map')
    else:
        result = _check(None, path)
    assert result[0] is None


@pytest.mark.parametrize('at_capture, at_render', [(None, GOAL_A), (GOAL_A, None), (None, None)])
def test_unknown_follower_goal_disables_path_alert(at_capture, at_render):
    store = _store()
    path = _path(store)
    binding = capture_costmap_path(100.1, "robot/odom", path, at_capture)
    assert _check(binding, path, goal=at_render)[0] is None


def test_path_clear_while_tf_waiting_disables_alert():
    store = _store()
    path = _path(store)
    binding = capture_costmap_path(100.1, "robot/odom", path, GOAL_A)
    store.clear()
    assert _check(binding, store.snapshot())[1] == "real_path_missing"


def test_follower_identity_rejects_stale_data():
    view = SimpleNamespace(
        route_sha256=GOAL_A[0], next_index=2, next_number=22,
        navigation_mode='to_pose', goal_generation=4, stamp_sec=100.0,
    )
    gate = SimpleNamespace(_view=view, timeout_sec=2.5)
    assert follower_goal_identity(gate, 101.0) == GOAL_A
    assert follower_goal_identity(gate, 102.6) is None
    assert follower_goal_identity(gate, 99.0) is None
    assert follower_goal_identity(None, 101.0) is None


def test_maximum_path_age_boundary_is_explicit():
    store = _store()
    path = _path(store, stamp=97.0)
    binding = capture_costmap_path(100.0, 'robot/odom', path, GOAL_A)
    assert _check(binding, path, costmap_stamp=100.0)[0] is path
    binding2 = capture_costmap_path(100.0001, 'robot/odom', path, GOAL_A)
    assert _check(binding2, path, costmap_stamp=100.0001)[0] is None
