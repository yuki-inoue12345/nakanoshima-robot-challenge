"""Evidence and rendering tests for neutral Costmap alerts/minimap."""

import cv2
import numpy as np

from ar_path_viewer.costmap_alerts import (
    CostmapAlertGate,
    evaluate_costmap_alert,
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
