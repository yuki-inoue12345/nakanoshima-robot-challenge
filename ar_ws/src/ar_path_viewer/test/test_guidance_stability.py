from ar_path_viewer.guidance_stability import GuidanceStabilityGate
from ar_path_viewer.offscreen_guidance import OffscreenGuidance


def guide(direction):
    bearing = {'LEFT': 30.0, 'STRAIGHT': 0.0, 'RIGHT': -30.0}[direction]
    return OffscreenGuidance(direction, bearing, 2.0, 1.8)


def test_initial_direction_waits_for_current_path_to_remain_stable():
    gate = GuidanceStabilityGate(0.35)
    assert gate.update(guide('STRAIGHT'), 10.0).status == 'DIRECTION CHECK'
    assert gate.update(guide('STRAIGHT'), 10.34).guidance is None
    accepted = gate.update(guide('STRAIGHT'), 10.35)
    assert accepted.guidance.direction == 'STRAIGHT'


def test_direction_change_removes_old_arrow_instead_of_smoothing_paths():
    gate = GuidanceStabilityGate(0.2)
    gate.update(guide('LEFT'), 1.0)
    assert gate.update(guide('LEFT'), 1.2).guidance.direction == 'LEFT'

    changed = gate.update(guide('RIGHT'), 1.3)
    assert changed.guidance is None
    assert changed.status == 'DIRECTION CHECK'
    # Another actual Path change restarts the hold and never revives LEFT.
    assert gate.update(guide('STRAIGHT'), 1.4).guidance is None
    assert gate.update(guide('STRAIGHT'), 1.6).guidance.direction == 'STRAIGHT'


def test_missing_or_invalid_guidance_clears_pending_state():
    gate = GuidanceStabilityGate(0.1)
    gate.update(guide('RIGHT'), 1.0)
    assert gate.update(None, 1.05).guidance is None
    assert gate.update(guide('RIGHT'), 1.2).guidance is None
