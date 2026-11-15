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
    colors = {
        "unknown": (90, 80, 65),
        "high": (20, 190, 230),
        "lethal": (205, 45, 235),
    }
    for point, cost in zip(selected_points, selected_costs):
        if cost < 0:
            color = colors["unknown"]
        elif cost >= minimap.lethal_cost_min:
            color = colors["lethal"]
        elif cost >= minimap.high_cost_min:
            color = colors["high"]
        else:
            continue
        x = int(round(center_x - point[1] * scale))
        y = int(round(origin_y - point[0] * scale))
        if left < x < right and top + 16 < y < bottom:
            frame[y, x] = color

    path = minimap.path_points_base
    if path is not None:
        path = np.asarray(path, dtype=np.float64)
        if path.ndim == 2 and path.shape[1] == 3:
            pixels = []
            for point in path:
                if 0.0 <= point[0] <= range_m and abs(point[1]) <= range_m * 0.5:
                    pixels.append((
                        int(round(center_x - point[1] * scale)),
                        int(round(origin_y - point[0] * scale)),
                    ))
            if len(pixels) >= 2:
                cv2.polylines(frame, [np.asarray(pixels, np.int32)], False,
                              (0, 255, 0), 2, cv2.LINE_AA)

    reference = minimap.reference_points_base
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim == 2 and reference.shape[1] == 3:
            pixels = []
            for point in reference:
                if 0.0 <= point[0] <= range_m and abs(point[1]) <= range_m * 0.5:
                    pixels.append((
                        int(round(center_x - point[1] * scale)),
                        int(round(origin_y - point[0] * scale)),
                    ))
            for pixel in pixels[::5]:
                cv2.circle(frame, pixel, 1, (255, 255, 0), -1, cv2.LINE_AA)

    half_l = max(2, int(round(robot_half_length_m * scale)))
    half_w = max(2, int(round(robot_half_width_m * scale)))
    cv2.rectangle(frame, (center_x - half_w, origin_y - 2 * half_l),
                  (center_x + half_w, origin_y), (230, 230, 230), 1)
    cv2.arrowedLine(frame, (center_x, origin_y - 2),
                    (center_x, origin_y - 2 * half_l - 7),
                    (230, 230, 230), 1, cv2.LINE_AA, tipLength=0.4)


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
