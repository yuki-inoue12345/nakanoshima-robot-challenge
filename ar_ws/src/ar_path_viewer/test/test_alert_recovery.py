"""Transient missing camera-time evidence must never itself be a live alert."""

from ar_path_viewer.costmap_alerts import CostmapAlert, CostmapAlertGate


def test_short_sync_gap_keeps_confirmation_but_not_visible_warning():
    clock = [0.0]
    gate = CostmapAlertGate(0.5, clock=lambda: clock[0])
    candidate = CostmapAlert("COSTMAP ALERT: NEARBY", "nearby", "")
    assert gate.update(candidate) is None
    clock[0] = 0.35
    assert gate.update(candidate) is None
    clock[0] = 0.50
    gate.suspend()
    # No warning is shown by suspend. The second validated observation
    # is needed before the visible alert can return.
    clock[0] = 0.61
    assert gate.update(candidate) == candidate


def test_long_gap_clears_old_candidate_and_requires_confirmation_again():
    clock = [0.0]
    gate = CostmapAlertGate(0.5, clock=lambda: clock[0])
    alert = CostmapAlert("FORWARD COSTMAP CAUTION", "forward", "")
    assert gate.update(alert) is None
    clock[0] = 0.7
    gate.suspend()
    assert gate.update(alert) is None
    clock[0] = 1.21
    assert gate.update(alert) == alert
