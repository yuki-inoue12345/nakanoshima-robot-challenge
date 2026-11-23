"""Verify measured turn continuity without inventing rotation."""

import math

import pytest

from ar_path_viewer.rotation_motion import OdomRotationMotionGate


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("slowdown", [False, True])
def test_full_180_turn_stays_visible_after_detection(sign, slowdown):
    gate = OdomRotationMotionGate()
    dt = 1.0 / 30.0
    angle = 0.0
    detected = False
    missing = 0
    slow_frames = 0
    expected = "ROTATE LEFT" if sign > 0 else "ROTATE RIGHT"

    for i in range(600):
        t = 1.0 + i * dt
        slow = slowdown and 20 <= i % 60 < 32
        rate = 0.15 if slow else 0.384
        slow_frames += int(slow)
        angle += rate * dt

        gate.update(0.002, sign * rate, t)
        shown = gate.direction(t) == expected

        if shown:
            detected = True
        elif detected:
            missing += 1

        if angle >= math.pi:
            break

    assert detected
    assert math.degrees(angle) >= 180.0
    assert missing == 0
    if slowdown:
        assert slow_frames > 0


def test_slow_rotation_alone_cannot_start_guidance():
    gate = OdomRotationMotionGate()
    for i in range(300):
        t = 1.0 + i / 30.0
        assert gate.update(0.002, 0.15, t) is None
        assert gate.direction(t) is None


def _verified_gate():
    gate = OdomRotationMotionGate()
    for i in range(15):
        gate.update(0.002, 0.384, 1.0 + i / 30.0)
    last = 1.0 + 14.0 / 30.0
    assert gate.direction(last) == "ROTATE LEFT"
    return gate, last


def test_stop_clears_verified_rotation():
    gate, last = _verified_gate()
    t = last + 1.0 / 30.0
    assert gate.update(0.0, 0.0, t) is None
    assert gate.direction(t) is None


def test_reverse_direction_clears_previous_guidance():
    gate, last = _verified_gate()
    t = last + 1.0 / 30.0
    assert gate.update(0.0, -0.384, t) is None
    assert gate.direction(t) is None


def test_forward_motion_clears_rotation():
    gate, last = _verified_gate()
    t = last + 1.0 / 30.0
    assert gate.update(0.13, 0.384, t) is None
    assert gate.direction(t) is None


def test_stale_data_cannot_continue_rotation():
    gate, last = _verified_gate()
    assert gate.direction(last + 0.23) is None
    assert gate.update(0.002, 0.15, last + 0.26) is None
    assert gate.direction(last + 0.26) is None


def test_below_continuation_threshold_clears_rotation():
    gate, last = _verified_gate()
    t = last + 1.0 / 30.0
    assert gate.update(0.002, 0.11, t) is None
    assert gate.direction(t) is None
