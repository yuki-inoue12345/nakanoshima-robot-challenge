"""OpenCV rendering for viewer states and real projected path segments."""

import math
import textwrap
from typing import Iterable, Tuple

import cv2
import numpy as np

from .costmap_overlay import ProjectedCostmap
from .costmap_explanation import CostmapExplanation
from .costmap_alerts import CostmapAlert
from .costmap_minimap import (
    CostmapMinimap,
    draw_costmap_minimap,
    draw_costmap_minimap_status,
)
from .offscreen_guidance import OffscreenGuidance


VALID_STATES = {
    "ACTIVE",
    "NO PATH",
    "PATH STALE",
    "TF UNAVAILABLE",
    "CAMERA UNAVAILABLE",
    "FRAME SYNC WAIT",
    "PATH OUT OF VIEW",
    "LOCALIZING",
    "LOCALIZATION UNSTABLE",
    "REPLANNING",
    "NAVIGATION HOLD",
    "INVALID CONFIG",
}


# These are Nav2 cost classes, not a claim about camera-visible object type or
# generic danger.  BGR colors are deliberately distinct from green real Path,
# cyan reference Path, and orange direction guidance.
COSTMAP_UNKNOWN_BGR = (155, 135, 105)
COSTMAP_HIGH_BGR = (20, 190, 230)
COSTMAP_LETHAL_BGR = (205, 45, 235)


def draw_costmap_explanation(
    frame: np.ndarray,
    explanation: CostmapExplanation | None,
) -> None:
    """Draw one compact evidence label without covering the navigation card."""
    if explanation is None:
        return
    accent = {
        "low_cost": (80, 210, 170),
        "side_cost": COSTMAP_HIGH_BGR,
        "replanning": COSTMAP_HIGH_BGR,
        "blocked": COSTMAP_LETHAL_BGR,
    }.get(explanation.kind, (210, 210, 210))
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = explanation.label[:38]
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, 0.50, 1
    )
    width = min(370, max(230, text_width + 22))
    left, top, height = 10, 50, 53
    layer = frame.copy()
    cv2.rectangle(
        layer, (left, top), (left + width, top + height), (18, 18, 18), -1
    )
    cv2.addWeighted(layer, 0.86, frame, 0.14, 0.0, frame)
    cv2.rectangle(frame, (left, top), (left + 5, top + height), accent, -1)
    cv2.putText(
        frame, "COSTMAP + PLAN", (left + 14, top + 16), font,
        0.34, (190, 190, 190), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, label, (left + 14, top + 38 + baseline // 2), font,
        0.50, accent, 1, cv2.LINE_AA,
    )


def draw_costmap_alert(
    frame: np.ndarray,
    alert: CostmapAlert | None,
) -> None:
    """Draw a neutral Costmap observation, never a generic danger claim."""
    if alert is None:
        return
    accent = (
        COSTMAP_LETHAL_BGR
        if alert.kind in ("nearby", "lethal_path")
        else COSTMAP_HIGH_BGR
    )
    label = alert.label[:38]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.48, 1)
    detail_label = ""
    for prefix, title in (
        ("nearest_lethal_m=", "LETHAL CELL GAP"),
        ("path_distance_m=", "PATH GAP"),
    ):
        if alert.detail.startswith(prefix):
            try:
                detail_label = f"{title} {float(alert.detail[len(prefix):]):.2f} m"
            except ValueError:
                detail_label = ""
            break
    if alert.detail == "confirmed_real_plan_generation_change":
        detail_label = "REAL /plan GENERATION CHANGED"
    left, top = 10, 109
    width = min(370, max(230, text_width + 22))
    if detail_label:
        width = min(370, max(width, 278))
    height = 38
    layer = frame.copy()
    cv2.rectangle(layer, (left, top), (left + width, top + height), (18, 18, 18), -1)
    cv2.addWeighted(layer, 0.86, frame, 0.14, 0.0, frame)
    cv2.rectangle(frame, (left, top), (left + 5, top + height), accent, -1)
    label_y = top + (15 if detail_label else 24) + baseline // 2
    cv2.putText(frame, label, (left + 14, label_y),
                font, 0.48, accent, 1, cv2.LINE_AA)
    if detail_label:
        cv2.putText(
            frame, detail_label, (left + 14, top + 32), font,
            0.31, (205, 205, 205), 1, cv2.LINE_AA,
        )


def draw_costmap_overlay(
    image: np.ndarray,
    projection: ProjectedCostmap | None,
    *,
    alpha: float = 0.42,
    radius_px: int = 2,
) -> np.ndarray:
    """Draw sparse semantic markers below every navigation overlay layer."""
    if projection is None or projection.visible_points == 0:
        return image.copy()
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("costmap overlay alpha must be in (0, 1]")
    if radius_px < 1:
        raise ValueError("costmap marker radius must be positive")
    frame = image.copy()
    layer = frame.copy()
    for x, y in projection.unknown_pixels:
        cv2.circle(
            layer, (int(x), int(y)), radius_px,
            COSTMAP_UNKNOWN_BGR, -1, cv2.LINE_AA,
        )
    for x, y in projection.high_cost_pixels:
        cv2.circle(
            layer, (int(x), int(y)), radius_px + 1,
            COSTMAP_HIGH_BGR, -1, cv2.LINE_AA,
        )
    for x, y in projection.lethal_cost_pixels:
        cv2.drawMarker(
            layer, (int(x), int(y)), COSTMAP_LETHAL_BGR,
            cv2.MARKER_DIAMOND, radius_px * 3 + 1, 2, cv2.LINE_AA,
        )
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, frame)
    return frame


def direction_arrow_segments(
    segments: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    spacing_px: float = 90.0,
    minimum_path_px: float = 18.0,
    arrow_length_px: float = 18.0,
) -> list[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Place arrow shafts along visible projected segments without moving the path."""
    segment_list = list(segments)
    chains: list[list[Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    current: list[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for start, end in segment_list:
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 1.0:
            continue
        if current and math.hypot(
            start[0] - current[-1][1][0], start[1] - current[-1][1][1]
        ) > 2.0:
            chains.append(current)
            current = []
        current.append((start, end))
    if current:
        chains.append(current)

    arrows: list[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for chain in chains:
        lengths = [
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in chain
        ]
        total = sum(lengths)
        if total < minimum_path_px:
            continue
        targets = list(np.arange(spacing_px, total + 1e-9, spacing_px))
        if not targets:
            targets = [total]
        elif total - targets[-1] >= spacing_px * 0.65:
            targets.append(total)
        for target_distance in targets:
            walked = 0.0
            for (start, end), length in zip(chain, lengths):
                if walked + length + 1e-9 < target_distance:
                    walked += length
                    continue
                fraction = min(1.0, max(0.0, (target_distance - walked) / length))
                target = np.array(start, dtype=float) + fraction * (
                    np.array(end, dtype=float) - np.array(start, dtype=float)
                )
                unit = (np.array(end, dtype=float) - np.array(start, dtype=float)) / length
                origin = target - unit * min(arrow_length_px, target_distance)
                arrows.append(
                    (
                        tuple(np.rint(origin).astype(int)),
                        tuple(np.rint(target).astype(int)),
                    )
                )
                break
    return arrows


def dashed_line_segments(
    segments: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    dash_px: float = 14.0,
    gap_px: float = 9.0,
) -> list[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Split real projected segments into a stable dash pattern."""
    if dash_px <= 0.0 or gap_px < 0.0:
        raise ValueError("dash_px must be positive and gap_px non-negative")
    period = dash_px + gap_px
    phase = 0.0
    output = []
    for start, end in segments:
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(vector))
        if length < 1.0:
            continue
        unit = vector / length
        distance = 0.0
        while distance < length - 1e-9:
            in_period = phase % period
            step = min(length - distance, period - in_period)
            if in_period < dash_px:
                visible = min(step, dash_px - in_period)
                if visible > 0.5:
                    dash_start = np.asarray(start, dtype=float) + unit * distance
                    dash_end = dash_start + unit * visible
                    output.append(
                        (
                            tuple(np.rint(dash_start).astype(int)),
                            tuple(np.rint(dash_end).astype(int)),
                        )
                    )
            distance += step
            phase += step
        # Projection clipping may split one continuous path by rounding only;
        # retaining phase prevents every source segment looking like a solid line.
    return output


def offscreen_arrow_geometry(
    width: int,
    height: int,
    guidance: OffscreenGuidance,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Place a fixed UI arrow; it is deliberately not a floor projection."""
    center_x = width // 2
    start_y = max(32, height - 26)
    end_y = max(18, start_y - max(54, int(round(height * 0.16))))
    maximum_horizontal = max(30, int(round(width * 0.18)))
    horizontal = int(round(
        -guidance.bearing_deg / 80.0 * maximum_horizontal
    ))
    end_x = int(np.clip(center_x + horizontal, 12, width - 13))
    return (center_x, start_y), (end_x, end_y)


def _draw_offscreen_guidance(
    frame: np.ndarray,
    guidance: OffscreenGuidance,
    color_bgr: Tuple[int, int, int],
) -> None:
    # Use the same immediately recognisable navigation card whether the real
    # Path is visible or outside the camera view.  The state banner still says
    # PATH OUT OF VIEW; the card never pretends to be projected geometry.
    _draw_direction_panel(frame, guidance, color_bgr)


def _draw_bottom_badge(
    frame: np.ndarray,
    label: str,
    color_bgr: Tuple[int, int, int],
    *,
    align_right: bool = False,
) -> None:
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    x = frame.shape[1] - text_width - 14 if align_right else 14
    x = max(8, x)
    y = frame.shape[0] - 14
    cv2.rectangle(
        frame,
        (x - 6, y - text_height - 6),
        (min(frame.shape[1] - 1, x + text_width + 6), y + baseline + 5),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame, label, (x, y), font, scale, color_bgr,
        thickness, cv2.LINE_AA,
    )


def _draw_direction_panel(
    frame: np.ndarray,
    guidance: OffscreenGuidance,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw a prominent turn-shaped cue without inventing route geometry."""
    width, height = frame.shape[1], frame.shape[0]
    panel_width = min(228, max(188, int(round(width * 0.36))))
    panel_height = 74
    right = width - 10
    left = right - panel_width
    # Projected floor routes occupy the lower half of the C920 image.  Keeping
    # the fixed UI card below the state banner but above the floor preserves
    # both the green real Path and cyan reference Path.
    top = min(max(50, int(round(height * 0.11))), height - panel_height - 8)
    bottom = top + panel_height
    cv2.rectangle(frame, (left, top), (right, bottom), (18, 18, 18), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), color_bgr, 2, cv2.LINE_AA)

    icon_left = left + 12
    icon_right = left + 72
    icon_center = (icon_left + icon_right) // 2
    arrow_bottom = bottom - 12
    arrow_turn_y = top + 26
    arrow_top = top + 11
    if guidance.direction == 'LEFT':
        cv2.line(
            frame, (icon_center, arrow_bottom), (icon_center, arrow_turn_y),
            color_bgr, 6, cv2.LINE_AA,
        )
        cv2.arrowedLine(
            frame, (icon_center, arrow_turn_y), (icon_left, arrow_turn_y),
            color_bgr, 6, cv2.LINE_AA, tipLength=0.42,
        )
        label = 'TURN LEFT'
    elif guidance.direction == 'RIGHT':
        cv2.line(
            frame, (icon_center, arrow_bottom), (icon_center, arrow_turn_y),
            color_bgr, 6, cv2.LINE_AA,
        )
        cv2.arrowedLine(
            frame, (icon_center, arrow_turn_y), (icon_right, arrow_turn_y),
            color_bgr, 6, cv2.LINE_AA, tipLength=0.42,
        )
        label = 'TURN RIGHT'
    else:
        cv2.arrowedLine(
            frame, (icon_center, arrow_bottom), (icon_center, arrow_top),
            color_bgr, 6, cv2.LINE_AA, tipLength=0.30,
        )
        label = 'GO STRAIGHT'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = max(left + 82, right - text_width - 10)
    text_y = top + (panel_height + text_height) // 2 - 2
    cv2.putText(
        frame, label, (text_x, text_y), font, scale, color_bgr,
        thickness, cv2.LINE_AA,
    )


def _draw_navigation_status(frame: np.ndarray, label: str) -> None:
    if not label:
        return
    _draw_bottom_badge(frame, label, (180, 180, 180), align_right=True)


def render_frame(
    image: np.ndarray,
    segments: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    state: str,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
    detail: str = "",
    guidance: OffscreenGuidance | None = None,
    guidance_color_bgr: Tuple[int, int, int] = (0, 165, 255),
    reference_segments: Iterable[
        Tuple[Tuple[int, int], Tuple[int, int]]
    ] = (),
    reference_color_bgr: Tuple[int, int, int] = (255, 255, 0),
    reference_thickness: int = 3,
    reference_dash_px: float = 14.0,
    reference_gap_px: float = 9.0,
    display_mode_label: str = "",
    waypoint_status: str = "",
    navigation_status: str = "",
    debug_overlay: bool = True,
    path_alpha: float = 1.0,
    costmap_projection: ProjectedCostmap | None = None,
    costmap_alpha: float = 0.42,
    costmap_radius_px: int = 2,
    costmap_explanation: CostmapExplanation | None = None,
    costmap_minimap: CostmapMinimap | None = None,
    costmap_minimap_size_px: int = 142,
    costmap_minimap_range_m: float = 3.0,
    costmap_minimap_status: str = "",
    costmap_alert: CostmapAlert | None = None,
) -> np.ndarray:
    if state not in VALID_STATES:
        raise ValueError(f"unsupported viewer state: {state}")
    if not math.isfinite(path_alpha) or not 0.0 < path_alpha <= 1.0:
        raise ValueError("path_alpha must be finite and in (0, 1]")
    frame = draw_costmap_overlay(
        image,
        costmap_projection,
        alpha=costmap_alpha,
        radius_px=costmap_radius_px,
    )
    segment_list = list(segments)
    reference_list = list(reference_segments)
    reference_dashes = dashed_line_segments(
        reference_list, reference_dash_px, reference_gap_px
    )
    # A dark halo makes cyan dashes readable against white walls without
    # changing their position, length, or distinction from the green Path.
    for start, end in reference_dashes:
        cv2.line(
            frame, start, end, (18, 18, 18),
            reference_thickness + 3, cv2.LINE_AA,
        )
    for start, end in reference_dashes:
        cv2.line(
            frame, start, end, reference_color_bgr,
            reference_thickness, cv2.LINE_AA,
        )
    if state == "ACTIVE":
        route_layer = frame.copy()
        for start, end in segment_list:
            cv2.line(
                route_layer, start, end, color_bgr, thickness, cv2.LINE_AA
            )
        for start, end in direction_arrow_segments(segment_list):
            cv2.arrowedLine(
                route_layer, start, end, color_bgr, thickness, cv2.LINE_AA,
                tipLength=0.4,
            )
        if path_alpha < 1.0:
            cv2.addWeighted(
                route_layer, path_alpha, frame, 1.0 - path_alpha, 0.0, frame
            )
        else:
            frame = route_layer
        if guidance is not None:
            _draw_direction_panel(frame, guidance, guidance_color_bgr)
    elif state == "PATH OUT OF VIEW" and guidance is not None:
        _draw_offscreen_guidance(frame, guidance, guidance_color_bgr)
    if waypoint_status and state in ("ACTIVE", "PATH OUT OF VIEW"):
        _draw_bottom_badge(frame, waypoint_status, (230, 230, 230))
    if navigation_status and state in (
        "ACTIVE", "PATH OUT OF VIEW", "REPLANNING", "NAVIGATION HOLD"
    ):
        _draw_navigation_status(frame, navigation_status)
    draw_costmap_explanation(frame, costmap_explanation)
    draw_costmap_alert(frame, costmap_alert)
    draw_costmap_minimap(
        frame, costmap_minimap,
        size_px=costmap_minimap_size_px,
        range_m=costmap_minimap_range_m,
    )
    if costmap_minimap is None:
        draw_costmap_minimap_status(
            frame, costmap_minimap_status, size_px=costmap_minimap_size_px
        )
    label_color = (0, 220, 0) if state == "ACTIVE" else (0, 180, 255)
    detail_lines = (
        textwrap.wrap(detail, width=68)[:2]
        if debug_overlay and detail else []
    )
    box_bottom = 42 + 24 * len(detail_lines)
    cv2.rectangle(
        frame, (8, 8), (max(8, frame.shape[1] - 8), box_bottom), (20, 20, 20), -1
    )
    cv2.putText(frame, state, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                label_color, 2, cv2.LINE_AA)
    right_edge = frame.shape[1] - 18
    if display_mode_label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        mode_thickness = 1
        (mode_width, _), _ = cv2.getTextSize(
            display_mode_label, font, scale, mode_thickness
        )
        cv2.putText(
            frame,
            display_mode_label,
            (max(18, right_edge - mode_width), 34),
            font,
            scale,
            (210, 210, 210),
            mode_thickness,
            cv2.LINE_AA,
        )
        right_edge -= mode_width + 18
    if reference_list:
        label = "REFERENCE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        label_thickness = 2
        (label_width, _), _ = cv2.getTextSize(
            label, font, scale, label_thickness
        )
        cv2.putText(
            frame,
            label,
            (max(18, right_edge - label_width), 34),
            font,
            scale,
            reference_color_bgr,
            label_thickness,
            cv2.LINE_AA,
        )
        right_edge -= label_width + 18
    if costmap_projection is not None:
        label = "LOCAL COSTMAP"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.43
        label_thickness = 1
        (label_width, _), _ = cv2.getTextSize(
            label, font, scale, label_thickness
        )
        cv2.putText(
            frame,
            label,
            (max(18, right_edge - label_width), 34),
            font,
            scale,
            COSTMAP_HIGH_BGR,
            label_thickness,
            cv2.LINE_AA,
        )
    for index, line in enumerate(detail_lines):
        cv2.putText(frame, line, (18, 58 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return frame
