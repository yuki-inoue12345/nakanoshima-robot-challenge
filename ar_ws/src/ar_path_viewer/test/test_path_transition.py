import numpy as np
import pytest

from ar_path_viewer.path_transition import PathTransitionGate


def _path(points):
    return np.asarray([(x, y, 0.0) for x, y in points], dtype=float)


def test_compatible_progress_is_visible_immediately():
    gate = PathTransitionGate()
    first = _path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    progressed = _path([(0.3, 0.0), (1.3, 0.0), (2.3, 0.0)])

    assert gate.update(1, first, 10.0).visible
    decision = gate.update(2, progressed, 10.1)

    assert decision.visible
    assert "compatible" in decision.detail


def test_progress_into_existing_corner_is_not_a_replan():
    gate = PathTransitionGate()
    first = _path([
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)
    ])
    progressed = _path([(1.8, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)])

    assert gate.update(1, first, 10.0).visible
    assert gate.update(2, progressed, 10.1).visible


def test_large_turn_hides_new_path_then_accepts_latest_without_old_path_hold():
    gate = PathTransitionGate(hold_sec=0.35)
    straight = _path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    left = _path([(0.0, 0.0), (0.4, 0.4), (0.4, 1.6)])

    assert gate.update(1, straight, 10.0).visible
    pending = gate.update(2, left, 10.1)
    assert not pending.visible
    assert pending.status == "PATH CHANGE CHECK"
    assert "heading_deg=" in pending.detail
    assert not gate.update(2, left, 10.44).visible
    assert gate.update(2, left, 10.46).visible


def test_repeated_new_disruptive_shape_restarts_confirmation():
    gate = PathTransitionGate(hold_sec=0.35)
    straight = _path([(0.0, 0.0), (2.0, 0.0)])
    left = _path([(0.0, 0.0), (0.0, 2.0)])
    right = _path([(0.0, 0.0), (0.0, -2.0)])

    gate.update(1, straight, 1.0)
    assert not gate.update(2, left, 1.1).visible
    assert not gate.update(3, right, 1.3).visible
    assert not gate.update(3, right, 1.5).visible
    assert gate.update(3, right, 1.66).visible


def test_compatible_updates_to_pending_replan_keep_original_hold_window():
    gate = PathTransitionGate(hold_sec=0.35)
    straight = _path([(0.0, 0.0), (2.0, 0.0)])
    left = _path([(0.0, 0.0), (0.0, 2.0)])
    left_refined = _path([(0.0, 0.0), (0.05, 1.0), (0.05, 2.0)])

    gate.update(1, straight, 1.0)
    assert not gate.update(2, left, 1.1).visible
    assert not gate.update(3, left_refined, 1.2).visible
    assert not gate.update(3, left_refined, 1.44).visible
    assert gate.update(3, left_refined, 1.46).visible


def test_invalid_thresholds_fail_closed():
    with pytest.raises(ValueError):
        PathTransitionGate(max_displacement_m=0.0)
    with pytest.raises(ValueError):
        PathTransitionGate(hold_sec=-0.1)
