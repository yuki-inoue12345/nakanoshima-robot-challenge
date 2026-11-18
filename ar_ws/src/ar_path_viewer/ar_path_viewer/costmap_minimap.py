"""Compact top-down Local Costmap review rendering."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CostmapMinimap:
    points_in_base: np.ndarray
    costs: np.ndarray
    path_points_base: np.ndarray | None
    high_cost_min: int
    lethal_cost_min: int
    reference_points_base: np.ndarray | None = None


MINIMAP_MAX_DRAWN_CELLS = 2500
ROBOT_VIEW_PANEL_WIDTH = 220
ROBOT_VIEW_TAB_WIDTH = 24


def robot_view_tab_rectangle(
    frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Stable source-pixel hit target shared by rendering and HighGUI."""
    center_y = int(frame_height) // 2
    return (
        max(0, int(frame_width) - ROBOT_VIEW_TAB_WIDTH),
        max(0, center_y - 42),
        max(0, int(frame_width) - 1),
        min(int(frame_height) - 1, center_y + 42),
    )


def draw_robot_view_tab(frame: np.ndarray, expanded: bool) -> None:
    """Draw the recorded, right-edge disclosure tab without changing geometry."""
    left, top, right, bottom = robot_view_tab_rectangle(
        frame.shape[1], frame.shape[0]
    )
    layer = frame.copy()
    cv2.rectangle(layer, (left, top), (right, bottom), (20, 20, 20), -1)
    cv2.addWeighted(layer, 0.88, frame, 0.12, 0.0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (190, 190, 190), 1)
    middle_y = (top + bottom) // 2
    if expanded:
        points = np.asarray(
            ((left + 7, middle_y), (right - 6, middle_y - 10),
             (right - 6, middle_y + 10)), np.int32,
        )
    else:
        points = np.asarray(
            ((right - 7, middle_y), (left + 6, middle_y - 10),
             (left + 6, middle_y + 10)), np.int32,
        )
    cv2.polylines(frame, [points], False, (245, 245, 245), 2, cv2.LINE_AA)


def draw_robot_view_panel(
    frame: np.ndarray,
    minimap: CostmapMinimap | None,
    *,
    range_m: float = 3.0,
    status: str = "",
) -> None:
    """Overlay an expanded review panel; never resize or reproject the camera."""
    width = min(ROBOT_VIEW_PANEL_WIDTH, max(100, frame.shape[1] - 40))
    left = frame.shape[1] - width
    top = min(126, max(44, frame.shape[0] // 4))
    bottom = frame.shape[0] - 8
    layer = frame.copy()
    cv2.rectangle(layer, (left, top), (frame.shape[1] - 1, bottom),
                  (10, 10, 10), -1)
    cv2.addWeighted(layer, 0.90, frame, 0.10, 0.0, frame)
    cv2.rectangle(frame, (left, top), (frame.shape[1] - 1, bottom),
                  (135, 135, 135), 1)
    if minimap is not None:
        size = min(190, bottom - top - 62, width - 20)
        draw_costmap_minimap(frame, minimap, size_px=max(80, size),
                             range_m=range_m)
    else:
        cv2.putText(frame, "ROBOT VIEW", (left + 12, top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, "PERCEPTION UNAVAILABLE", (left + 12, top + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 180, 255), 1,
                    cv2.LINE_AA)
        if status:
            cv2.putText(frame, str(status)[:28], (left + 12, top + 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1,
                        cv2.LINE_AA)


def select_minimap_class_indices(
    costs: np.ndarray, *, high_cost_min: int, lethal_cost_min: int,
    maximum: int = MINIMAP_MAX_DRAWN_CELLS,
) -> np.ndarray:
    """Select important visible cells before low-priority unknown coverage.

    Lethal > high > unknown, each spatially distributed when over budget.
    Returned indices are ordered unknown -> high -> lethal so more important
    points win when multiple grid cells happen to project to one pixel.
    """
    if maximum < 1:
        raise ValueError("minimap cell budget must be positive")
    costs = np.asarray(costs)
    remaining = maximum
    selected = {}
    masks = (
        ("lethal", costs >= lethal_cost_min),
        ("high", (costs >= high_cost_min) & (costs < lethal_cost_min)),
        ("unknown", costs < 0),
    )
    for name, mask in masks:
        indices = np.flatnonzero(mask)
        if len(indices) > remaining:
            if remaining > 0:
                chosen = np.linspace(0, len(indices) - 1, remaining, dtype=np.int64)
                indices = indices[chosen]
            else:
                indices = indices[:0]
        selected[name] = indices
        remaining -= len(indices)
    return np.concatenate((
        selected["unknown"], selected["high"], selected["lethal"]
    ))


def draw_costmap_minimap(
    frame: np.ndarray,
    minimap: CostmapMinimap | None,
    *,
    size_px: int = 142,
    range_m: float = 3.0,
    robot_half_length_m: float = 0.30,
    robot_half_width_m: float = 0.29,
) -> None:
    """Draw robot-forward-up geometry without claiming camera occlusion."""
    if minimap is None:
        return
    if size_px < 80 or not math.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("invalid costmap minimap geometry")
    points = np.asarray(minimap.points_in_base, dtype=np.float64)
    costs = np.asarray(minimap.costs, dtype=np.int16)
    if points.ndim != 2 or points.shape[1] != 3 or costs.shape != (len(points),):
        return
    height, width = frame.shape[:2]
    left = max(8, width - size_px - 10)
    # Retain the established right-column geometry so the top-down position
    # remains stable across Viewer versions.  The compact legend is placed
    # below the panel and does not move any Costmap cell.
    top = min(max(132, int(round(height * 0.29))), height - size_px - 10)
    right, bottom = left + size_px, top + size_px
    layer = frame.copy()
    cv2.rectangle(layer, (left, top), (right, bottom), (12, 12, 12), -1)
    cv2.addWeighted(layer, 0.82, frame, 0.18, 0.0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (160, 160, 160), 1)
    cv2.putText(frame, "ROBOT VIEW", (left + 7, top + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

    center_x = left + size_px // 2
    origin_y = bottom - 12
    scale = (size_px - 28) / range_m
    selected = (
        (points[:, 0] >= 0.0) & (points[:, 0] <= range_m)
        & (np.abs(points[:, 1]) <= range_m * 0.5)
    )
    selected_points = points[selected]
    selected_costs = costs[selected]
    indices = select_minimap_class_indices(
        selected_costs,
        high_cost_min=minimap.high_cost_min,
        lethal_cost_min=minimap.lethal_cost_min,
    )
    selected_points = selected_points[indices]
    selected_costs = selected_costs[indices]
    # Selection already orders unknown -> high -> lethal. Draw each class in
    # that same order so important classes still win at duplicate pixels, but
    # avoid thousands of Python/OpenCV calls in every 30 fps frame.
    if len(selected_points):
        pixel_x = np.rint(
            center_x - selected_points[:, 1] * scale
        ).astype(np.int32)
        pixel_y = np.rint(
            origin_y - selected_points[:, 0] * scale
        ).astype(np.int32)
        inside = (
            (pixel_x > left) & (pixel_x < right)
            & (pixel_y > top + 16) & (pixel_y < bottom)
        )
        class_masks = (
            (selected_costs < 0, (90, 80, 65)),
            (
                (selected_costs >= minimap.high_cost_min)
                & (selected_costs < minimap.lethal_cost_min),
                (20, 190, 230),
            ),
            (selected_costs >= minimap.lethal_cost_min, (205, 45, 235)),
        )
        for class_mask, color in class_masks:
            visible = inside & class_mask
            frame[pixel_y[visible], pixel_x[visible]] = color

    path = minimap.path_points_base
    if path is not None:
        path = np.asarray(path, dtype=np.float64)
        if path.ndim == 2 and path.shape[1] == 3:
            visible = (
                (path[:, 0] >= 0.0) & (path[:, 0] <= range_m)
                & (np.abs(path[:, 1]) <= range_m * 0.5)
            )
            path = path[visible]
            if len(path) >= 2:
                pixels = np.column_stack((
                    np.rint(center_x - path[:, 1] * scale),
                    np.rint(origin_y - path[:, 0] * scale),
                )).astype(np.int32)
                cv2.polylines(frame, [pixels], False,
                              (0, 255, 0), 2, cv2.LINE_AA)

    reference = minimap.reference_points_base
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim == 2 and reference.shape[1] == 3:
            visible = (
                (reference[:, 0] >= 0.0) & (reference[:, 0] <= range_m)
                & (np.abs(reference[:, 1]) <= range_m * 0.5)
            )
            reference = reference[visible]
            pixels = np.column_stack((
                np.rint(center_x - reference[:, 1] * scale),
                np.rint(origin_y - reference[:, 0] * scale),
            )).astype(np.int32)
            for pixel in pixels[::5]:
                cv2.circle(
                    frame, tuple(pixel), 1, (255, 255, 0), -1, cv2.LINE_AA
                )

    half_l = max(2, int(round(robot_half_length_m * scale)))
    half_w = max(2, int(round(robot_half_width_m * scale)))
    cv2.rectangle(frame, (center_x - half_w, origin_y - 2 * half_l),
                  (center_x + half_w, origin_y), (230, 230, 230), 1)
    cv2.arrowedLine(frame, (center_x, origin_y - 2),
                    (center_x, origin_y - 2 * half_l - 7),
                    (230, 230, 230), 1, cv2.LINE_AA, tipLength=0.4)
    _draw_costmap_legend(frame, left, bottom, size_px)


def _draw_costmap_legend(
    frame: np.ndarray, left: int, bottom: int, size_px: int
) -> None:
    """Label Nav2 cost classes without using a generic danger claim."""
    if bottom + 17 >= frame.shape[0]:
        return
    items = (
        ("UNK", (90, 80, 65)),
        ("HIGH", (20, 190, 230)),
        ("LETHAL", (205, 45, 235)),
    )
    layer = frame.copy()
    cv2.rectangle(
        layer, (left, bottom + 2), (left + size_px, bottom + 20),
        (12, 12, 12), -1,
    )
    cv2.addWeighted(layer, 0.82, frame, 0.18, 0.0, frame)
    x = left + 5
    for label, color in items:
        cv2.rectangle(frame, (x, bottom + 8), (x + 5, bottom + 13), color, -1)
        cv2.putText(
            frame, label, (x + 8, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX,
            0.24, color, 1, cv2.LINE_AA,
        )
        x += {"UNK": 38, "HIGH": 44, "LETHAL": 54}[label]


def draw_costmap_minimap_status(
    frame: np.ndarray,
    status: str,
    *,
    size_px: int = 142,
) -> None:
    """Show that perception is unavailable instead of implying free space."""
    if not status:
        return
    left = max(8, frame.shape[1] - size_px - 10)
    top = min(max(132, int(round(frame.shape[0] * 0.29))), frame.shape[0] - 38)
    right = left + size_px
    layer = frame.copy()
    cv2.rectangle(layer, (left, top), (right, top + 28), (18, 18, 18), -1)
    cv2.addWeighted(layer, 0.82, frame, 0.18, 0.0, frame)
    cv2.rectangle(frame, (left, top), (right, top + 28), (0, 180, 255), 1)
    cv2.putText(frame, "PERCEPTION UNAVAILABLE", (left + 5, top + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 180, 255), 1, cv2.LINE_AA)
