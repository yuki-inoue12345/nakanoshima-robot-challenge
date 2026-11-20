"""Evidence and rendering tests for neutral Costmap alerts/minimap."""

import cv2
import numpy as np

from ar_path_viewer.costmap_alerts import (
    CostmapAlert,
    CostmapAlertGate,
    CostmapAlertHistoryGate,
    evaluate_costmap_alert,
    evaluate_path_cost_alert,
    evaluate_forward_cost_alert,
    evaluate_robot_proximity_alert,
)
from ar_path_viewer.costmap_minimap import CostmapMinimap, draw_costmap_minimap
from ar_path_viewer.rendering import render_frame


def _path():
    return np.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))


def test_alert_precedence_distinguishes_nearby_lethal_and_high_path_cost():
    points = np.asarray(((0.4, 0.1, 0.0), (2.0, 0.1, 0.0), (2.0, 1.0, 0.0)))
    nearby, reason = evaluate_costmap_alert(
        points, np.asarray((100, 70, 0)), _path(),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert nearby.label == "COSTMAP ALERT: NEARBY"
    assert reason == "lethal_cost_near_robot"

    high, reason = evaluate_costmap_alert(
        points[1:], np.asarray((70, 0)), _path(),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert high.label == "HIGH COST NEAR PATH"
    assert reason == "high_cost_near_path"
    assert "danger" not in high.label.lower()


def test_alert_is_suppressed_for_invalid_path_and_debounced():
    points = np.asarray(((2.0, 0.1, 0.0),))
    alert, reason = evaluate_costmap_alert(
        points, np.asarray((100,)), np.empty((0, 3)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert alert is None
    assert reason == "real_path_unavailable"

    now = [0.0]
    gate = CostmapAlertGate(0.25, clock=lambda: now[0])
    alert, _ = evaluate_costmap_alert(
        points, np.asarray((100,)), _path(),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert gate.update(alert) is None
    now[0] = 0.25
    assert gate.update(alert).label == "LETHAL COST NEAR PATH"
    assert gate.update(None) is None


def test_robot_proximity_alert_does_not_require_a_real_path():
    """A fresh exact-time lethal cell near the footprint is independent of /plan."""
    points = np.asarray(((0.55, 0.15, 0.0), (2.0, 0.0, 0.0)))
    costs = np.asarray((100, 0), dtype=np.int16)

    nearby, reason = evaluate_robot_proximity_alert(
        points, costs, lethal_cost_min=100
    )
    assert nearby is not None
    assert nearby.kind == "nearby"
    assert reason == "lethal_cost_near_robot"

    path_alert, reason = evaluate_path_cost_alert(
        points, costs, np.empty((0, 3)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert path_alert is None
    assert reason == "real_path_unavailable"


def test_nearby_distance_uses_footprint_and_valid_release_hysteresis():
    points = np.asarray(((0.80, 0.29, 0.0),))
    alert, reason = evaluate_costmap_alert(
        points, np.asarray((100,)), _path(),
        high_cost_min=60, lethal_cost_min=100,
    )
    # Origin distance is > 0.6 m, footprint-edge distance is 0.5 m.
    assert alert.label == "COSTMAP ALERT: NEARBY"
    assert reason == "lethal_cost_near_robot"

    now = [0.0]
    gate = CostmapAlertGate(0.0, release_sec=0.4, clock=lambda: now[0])
    assert gate.update(alert) == alert
    now[0] = 0.3
    assert gate.update(None) == alert
    now[0] = 0.41
    assert gate.update(None) is None


def test_alert_history_is_distinct_expires_and_clears_on_goal_change():
    now = [10.0]
    gate = CostmapAlertHistoryGate(2.5, clock=lambda: now[0])
    alert = CostmapAlert(
        "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"
    )
    identity = ("a" * 64, 2, 14, "to_pose", 7)
    assert gate.update(
        alert, evaluation_status="active", goal_identity=identity
    ) is None

    now[0] = 10.2
    history = gate.update(
        None, evaluation_status="unavailable", goal_identity=identity
    )
    assert history.current_status == "UNCONFIRMED"
    assert history.alert == alert
    now[0] = 10.4
    history = gate.update(
        None, evaluation_status="cleared", goal_identity=identity
    )
    assert history.current_status == "CLEARED"

    # A different Follower goal must not inherit the old event.
    assert gate.update(
        None, evaluation_status="unavailable",
        goal_identity=("a" * 64, 3, 15, "to_pose", 8),
    ) is None


def test_alert_history_expires_without_extending_live_danger():
    now = [0.0]
    gate = CostmapAlertHistoryGate(2.5, clock=lambda: now[0])
    alert = CostmapAlert("HIGH COST NEAR PATH", "high_path", "path_distance_m=0.2")
    identity = ("a" * 64, 2, 14, "to_pose", 7)
    gate.update(alert, evaluation_status="active", goal_identity=identity)
    now[0] = 2.51
    assert gate.update(
        None, evaluation_status="unavailable", goal_identity=identity
    ) is None


def test_minimap_and_alert_keep_lower_navigation_path_visible():
    points = np.asarray(((0.5, -0.2, 0.0), (1.0, 0.3, 0.0), (2.0, 0.0, 0.0)))
    costs = np.asarray((-1, 70, 100))
    alert, _ = evaluate_costmap_alert(
        points, costs, _path(), high_cost_min=60, lethal_cost_min=100
    )
    minimap = CostmapMinimap(points, costs, _path(), 60, 100)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = render_frame(
        image, [((100, 420), (500, 300))], "ACTIVE",
        costmap_minimap=minimap, costmap_alert=alert,
        robot_view_enabled=True, robot_view_expanded=True,
    )
    assert cv2.countNonZero(cv2.cvtColor(frame[109:147, 10:380], cv2.COLOR_BGR2GRAY)) > 100
    assert cv2.countNonZero(cv2.cvtColor(frame[132:274, 488:630], cv2.COLOR_BGR2GRAY)) > 100
    assert cv2.countNonZero(frame[285:, :, 1]) > 20


def test_minimap_keeps_robot_left_on_image_left_and_cost_classes_distinct():
    # In base coordinates +y is robot-left.  The forward-up minimap must keep
    # that point left of center independently of robot yaw in the world.
    points = np.asarray((
        (0.5, 0.2, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, -0.5, 0.0),
    ))
    minimap = CostmapMinimap(
        points, np.asarray((-1, 70, 100)), None, 60, 100
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    draw_costmap_minimap(frame, minimap, size_px=142, range_m=3.0)

    assert tuple(frame[250, 551]) == (90, 80, 65)
    assert tuple(frame[231, 540]) == (20, 190, 230)
    assert tuple(frame[231, 578]) == (205, 45, 235)


def test_minimap_single_lethal_survives_many_unremarkable_cells():
    # Old stride sampling draws indices 0,2,4... and drops index 1.
    points = np.tile(np.array(((1.0, 0.0, 0.0),)), (3600, 1))
    points[1] = (1.0, 0.5, 0.0)
    costs = np.zeros(3600, dtype=np.int16)
    costs[1] = 100
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_costmap_minimap(
        frame, CostmapMinimap(points, costs, None, 60, 100),
        size_px=142, range_m=3.0,
    )
    assert tuple(frame[231, 540]) == (205, 45, 235)


def test_minimap_priority_keeps_each_sparse_cost_class():
    # 6000 visible cells, only three semantically significant.
    points = np.tile(np.array(((1.0, 0.0, 0.0),)), (6000, 1))
    targets = ((1, 0.5, 100), (2, -0.5, 80), (3, 0.0, -1))
    costs = np.zeros(6000, dtype=np.int16)
    for index, side, value in targets:
        points[index] = (1.0, side, 0.0)
        costs[index] = value
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_costmap_minimap(
        frame, CostmapMinimap(points, costs, None, 60, 100),
        size_px=142, range_m=3.0,
    )
    assert tuple(frame[231, 540]) == (205, 45, 235)
    assert tuple(frame[231, 578]) == (20, 190, 230)
    assert tuple(frame[231, 559]) == (90, 80, 65)


def test_minimap_budget_prefers_lethal_over_plentiful_high_cells():
    from ar_path_viewer.costmap_minimap import select_minimap_class_indices

    costs = np.full(6000, 80, dtype=np.int16)
    costs[1] = 100
    selected = select_minimap_class_indices(
        costs, high_cost_min=60, lethal_cost_min=100,
    )
    assert len(selected) == 2500
    assert len(set(selected)) == 2500
    assert 1 in selected
    assert np.count_nonzero(costs[selected] >= 100) == 1


def test_forward_costmap_caution_precedes_nearby_lethal_without_side_wall():
    points = np.asarray(((1.5, 0.05, 0.0), (1.3, 1.2, 0.0)))
    alert, reason = evaluate_forward_cost_alert(
        points, np.asarray((100, 100)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert alert is not None
    assert alert.kind == "forward"
    assert "forward_cost_m=" in alert.detail
    assert reason == "costmap_cost_in_forward_corridor"

    alert, reason = evaluate_forward_cost_alert(
        points[1:], np.asarray((100,)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert alert is None
    assert reason == "no_forward_costmap_evidence"

    alert, reason = evaluate_forward_cost_alert(
        points[:1], np.asarray((65,)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert alert is not None and alert.kind == "forward"


def test_forward_caution_never_claims_person_or_live_danger():
    points = np.asarray(((1.2, 0.0, 0.0),))
    alert, _ = evaluate_forward_cost_alert(
        points, np.asarray((100,)),
        high_cost_min=60, lethal_cost_min=100,
    )
    assert "PERSON" not in alert.label
    assert alert.kind != "nearby"


def test_review_forward_caution_rejects_sparse_or_wide_inflation():
    from ar_path_viewer.costmap_alerts import (
        CostmapAlertSettings, evaluate_forward_cost_alert,
    )
    settings = CostmapAlertSettings(
        forward_max_m=1.5, forward_half_width_m=0.30,
        forward_min_cost=85, forward_min_cells=3,
        nearby_lethal_distance_m=0.30,
    )
    # A lone high-cost cell is not evidence for a full-screen warning.
    points = np.asarray(((1.0, 0.0, 0.0), (1.1, 0.0, 0.0)))
    alert, reason = evaluate_forward_cost_alert(
        points, np.asarray((100, 90)), high_cost_min=60,
        lethal_cost_min=100, settings=settings,
    )
    assert alert is None and reason == "no_forward_costmap_evidence"
    # Wide/weak/too-distant inflation must not trigger the review banner.
    rejected = np.asarray(((1.0, 0.45, 0.0), (1.0, 0.0, 0.0),
                           (1.6, 0.0, 0.0), (1.0, 0.0, 0.0)))
    alert, _ = evaluate_forward_cost_alert(
        rejected, np.asarray((100, 60, 100, 70)), high_cost_min=60,
        lethal_cost_min=100, settings=settings,
    )
    assert alert is None
    # Multiple strong cells in the actual forward corridor can still warn.
    points = np.asarray(((0.95, -0.05, 0.0), (1.0, 0.0, 0.0),
                         (1.05, 0.05, 0.0)))
    alert, reason = evaluate_forward_cost_alert(
        points, np.asarray((85, 100, 90)), high_cost_min=60,
        lethal_cost_min=100, settings=settings,
    )
    assert alert is not None and alert.kind == "forward"
    assert reason == "costmap_cost_in_forward_corridor"


def test_review_caution_requires_longer_confirmation_than_nearby():
    clock = [0.0]
    gate = CostmapAlertGate(
        0.85, clock=lambda: clock[0], confirm_by_kind={"forward": 1.25},
    )
    caution = CostmapAlert("FORWARD COSTMAP CAUTION", "forward", "")
    red = CostmapAlert("COSTMAP ALERT: NEARBY", "nearby", "")
    assert gate.update(caution) is None
    clock[0] = 0.9
    assert gate.update(caution) is None
    clock[0] = 1.25
    assert gate.update(caution) == caution
    gate.clear()
    clock[0] = 2.0
    assert gate.update(red) is None
    clock[0] = 2.85
    assert gate.update(red) == red
    # Invalid inputs must never be turned into live danger via the gate.
    gate.clear()
    assert gate.update(None) is None


def test_review_red_nearby_requires_small_footprint_gap():
    from ar_path_viewer.costmap_alerts import CostmapAlertSettings
    settings = CostmapAlertSettings(nearby_lethal_distance_m=0.30)
    distant, reason = evaluate_robot_proximity_alert(
        np.asarray(((0.85, 0.0, 0.0),)), np.asarray((100,)),
        lethal_cost_min=100, settings=settings,
    )
    assert distant is None and reason == "no_lethal_cost_near_robot"
    close, reason = evaluate_robot_proximity_alert(
        np.asarray(((0.55, 0.0, 0.0),)), np.asarray((100,)),
        lethal_cost_min=100, settings=settings,
    )
    assert close is not None and reason == "lethal_cost_near_robot"


def test_side_wall_lethal_is_not_front_collision_but_front_lethal_is():
    from ar_path_viewer.costmap_alerts import CostmapAlertSettings
    settings = CostmapAlertSettings(
        nearby_lethal_distance_m=0.60, forward_half_width_m=0.40,
    )
    # Reconstructed from the actual 20:49:50 Costmap (right wall).
    side, reason = evaluate_robot_proximity_alert(
        np.asarray(((0.36, -0.87, 0.0),)), np.asarray((100,)),
        lethal_cost_min=100, settings=settings,
    )
    assert side is None and reason == 'no_lethal_cost_near_robot'
    # Reconstructed from the actual 20:49:42 Costmap (front region).
    front, reason = evaluate_robot_proximity_alert(
        np.asarray(((0.75, -0.15, 0.0),)), np.asarray((100,)),
        lethal_cost_min=100, settings=settings,
    )
    assert front is not None and reason == 'lethal_cost_near_robot'
    assert 'cell_x_m=0.750' in front.detail
    assert 'cell_y_m=-0.150' in front.detail
    # A lethal cell intersecting the current front half-footprint remains red.
    overlap, _ = evaluate_robot_proximity_alert(
        np.asarray(((0.10, 0.29, 0.0),)), np.asarray((100,)),
        lethal_cost_min=100, settings=settings,
    )
    assert overlap is not None
