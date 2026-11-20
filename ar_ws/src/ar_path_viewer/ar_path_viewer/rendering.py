"""OpenCV rendering for viewer states and real projected path segments."""

import math
import textwrap
from typing import Iterable, Tuple

import cv2
import numpy as np

from .costmap_overlay import ProjectedCostmap
from .costmap_explanation import CostmapExplanation
from .costmap_alerts import CostmapAlert, CostmapAlertHistory
from .costmap_minimap import (
    CostmapMinimap,
    draw_robot_view_panel,
    draw_robot_view_tab,
)
from .display_window import draw_localized_label
from .localization_monitor import LocalizationWarning
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
    """Draw prominent, evidence-bound Nav2 Costmap information."""
    if alert is None:
        return
    accent = (
        (30, 30, 245)
        if alert.kind in ("nearby", "lethal_path")
        else COSTMAP_HIGH_BGR
    )
    if alert.kind == "path_updated":
        # This is an event, not Costmap danger evidence.  Keep it compact.
        label = alert.label[:38]
        cv2.rectangle(frame, (10, 109), (265, 145), (18, 18, 18), -1)
        cv2.rectangle(frame, (10, 109), (15, 145), COSTMAP_HIGH_BGR, -1)
        cv2.putText(frame, label, (24, 133), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, COSTMAP_HIGH_BGR, 1, cv2.LINE_AA)
        return
    label = alert.label[:38]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.48, 1)
    detail_label = ""
    for prefix, title in (
        ("nearest_lethal_m=", "LETHAL CELL GAP"),
        ("path_distance_m=", "PATH GAP"),
        ("forward_cost_m=", "FRONT RANGE"),
    ):
        if alert.detail.startswith(prefix):
            try:
                detail_label = f"{title} {float(alert.detail[len(prefix):]):.2f} m"
            except ValueError:
                detail_label = ""
            break
    if alert.detail == "confirmed_real_plan_generation_change":
        detail_label = "REAL /plan GENERATION CHANGED"
    left, top, width, height = 10, 104, min(465, frame.shape[1] - 20), 99
    layer = frame.copy()
    background = (20, 20, 95) if alert.kind in ("nearby", "lethal_path") else (10, 65, 85)
    cv2.rectangle(layer, (left, top), (left + width, top + height), background, -1)
    cv2.addWeighted(layer, 0.76, frame, 0.24, 0.0, frame)
    cv2.rectangle(frame, (left, top), (left + width, top + height), accent, 3)
    cv2.rectangle(frame, (left, top), (left + 8, top + height), accent, -1)
    # The triangle is font-independent; Japanese text has a safe ASCII fallback.
    triangle = np.asarray(((left + 19, top + 55), (left + 45, top + 9),
                           (left + 71, top + 55)), np.int32)
    cv2.fillConvexPoly(frame, triangle, accent, cv2.LINE_AA)
    cv2.putText(frame, "!", (left + 40, top + 46), font, 0.78,
                (20, 20, 20), 2, cv2.LINE_AA)
    lethal = alert.kind in ("nearby", "lethal_path")
    draw_localized_label(
        frame, (left + 82, top + 4, left + width - 8, top + 51),
        "危険！" if lethal else "注意",
        "LETHAL COSTMAP CELL" if lethal else "COSTMAP CAUTION",
        accent, 37,
    )
    reason_japanese = {
        "lethal_path": "経路付近に通行不可領域",
        "nearby": "ロボット近傍に通行不可領域",
        "high_path": "経路付近に高コスト領域",
        "forward": "ロボット前方に高コスト領域",
    }.get(alert.kind, "Nav2 Costmapによる判定")
    draw_localized_label(
        frame, (left + 12, top + 55, left + width - 8, top + 78),
        reason_japanese, label, (245, 245, 245), 19,
    )
    evidence_label = (
        f"NAV2 LOCAL COSTMAP | {detail_label} | NOT CAMERA DETECTION"
        if detail_label else "NAV2 LOCAL COSTMAP | NOT CAMERA DETECTION"
    )
    cv2.putText(
        frame, evidence_label, (left + 14, top + 93), font,
        0.29, (230, 230, 230), 1, cv2.LINE_AA,
    )


def draw_costmap_alert_history(
    frame: np.ndarray, history: CostmapAlertHistory | None
) -> None:
    """Draw a recent-event notice that cannot be mistaken for a live alert."""
    if history is None:
        return
    left, top = 10, 109
    width, height = min(370, frame.shape[1] - 20), 68
    accent = (80, 150, 205)
    layer = frame.copy()
    cv2.rectangle(
        layer, (left, top), (left + width, top + height), (28, 34, 42), -1
    )
    cv2.addWeighted(layer, 0.68, frame, 0.32, 0.0, frame)
    cv2.rectangle(frame, (left, top), (left + width, top + height), accent, 1)
    draw_localized_label(
        frame, (left + 12, top + 5, left + width - 8, top + 31),
        ("直前に注意を検出" if history.alert.kind in ("high_path", "forward")
         else "直前に危険を検出"),
        "PREVIOUS COSTMAP NOTICE", accent, 18,
    )
    status_ja = (
        "現在の判定：解除"
        if history.current_status == "CLEARED"
        else "現在の判定：未確認"
    )
    status_en = (
        "CURRENT: CLEARED"
        if history.current_status == "CLEARED"
        else "CURRENT: UNCONFIRMED"
    )
    draw_localized_label(
        frame, (left + 12, top + 32, left + width - 8, top + 57),
        status_ja, status_en, (215, 215, 215), 16,
    )
    cv2.putText(
        frame, f"HISTORY {history.age_sec:.1f}s", (left + width - 98, top + 64),
        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (150, 150, 150), 1, cv2.LINE_AA,
    )


def draw_localization_warning(
    frame: np.ndarray, warning: LocalizationWarning | None
) -> None:
    """Draw current and recovered localization states without implying control."""
    if warning is None:
        return
    active = warning.active
    left, top = 10, 207
    width = min(400, frame.shape[1] - 20)
    height = 72 if active else 58
    accent = (35, 45, 245) if active else (80, 150, 205)
    background = (18, 18, 92) if active else (28, 34, 42)
    layer = frame.copy()
    cv2.rectangle(
        layer, (left, top), (left + width, top + height), background, -1
    )
    cv2.addWeighted(layer, 0.78 if active else 0.66, frame,
                    0.22 if active else 0.34, 0.0, frame)
    cv2.rectangle(frame, (left, top), (left + width, top + height), accent,
                  3 if active else 1)
    cv2.rectangle(frame, (left, top), (left + 7, top + height), accent, -1)
    title = "LOCALIZATION WARNING" if active else "RECENT LOCALIZATION WARNING"
    draw_localized_label(
        frame, (left + 12, top + 3, left + width - 12, top + 30),
        "自己位置を確認できません" if active else "自己位置情報は復旧しました",
        title, accent, 20,
    )
    reason_japanese = (
        "自己位置情報の更新が遅れています"
        if "STALE" in warning.reason
        else "位置情報の正確さを確認してください"
    )
    draw_localized_label(
        frame, (left + 12, top + 29, left + width - 12, top + 52),
        reason_japanese,
        warning.reason[:43], (245, 245, 245), 16,
    )
    if active:
        detail = warning.detail.replace("\n", " ")[:57]
        cv2.putText(
            frame, detail, (left + 15, top + 64), cv2.FONT_HERSHEY_SIMPLEX,
            0.28, (210, 210, 210), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, f"RECOVERED {warning.age_sec:.1f}s AGO",
            (left + width - 144, top + 51), cv2.FONT_HERSHEY_SIMPLEX,
            0.27, (170, 170, 170), 1, cv2.LINE_AA,
        )


def draw_costmap_overlay(
    image: np.ndarray,
    projection: ProjectedCostmap | None,
    *,
    alpha: float = 0.42,
    radius_px: int = 2,
) -> np.ndarray:
    """Draw outlined ground patches below every navigation overlay layer."""
    if projection is None or projection.visible_points == 0:
        return image.copy()
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("costmap overlay alpha must be in (0, 1]")
    if radius_px < 1:
        raise ValueError("costmap marker radius must be positive")
    frame = image.copy()
    layer = frame.copy()
    outline = (12, 12, 12)

    # The previous implementation allocated and dilated six full 640x480
    # masks on every camera frame. Projection already bounds each class to a
    # small, configured sample, so drawing only those markers is materially
    # cheaper while preserving the same outlined ground-patch semantics.
    for x_value, y_value in projection.unknown_pixels:
        cv2.circle(
            layer, (int(x_value), int(y_value)), radius_px,
            COSTMAP_UNKNOWN_BGR, -1, cv2.LINE_8,
        )
    for x_value, y_value in projection.high_cost_pixels:
        center = (int(x_value), int(y_value))
        cv2.circle(layer, center, radius_px + 4, outline, -1, cv2.LINE_8)
        cv2.circle(
            layer, center, radius_px + 2,
            COSTMAP_HIGH_BGR, -1, cv2.LINE_8,
        )
    lethal_radius = radius_px * 2
    for x_value, y_value in projection.lethal_cost_pixels:
        center_x, center_y = int(x_value), int(y_value)
        for patch_radius, color in (
            (lethal_radius + 6, outline),
            (lethal_radius + 4, COSTMAP_LETHAL_BGR),
        ):
            diamond = np.asarray(
                (
                    (center_x, center_y - patch_radius),
                    (center_x + patch_radius, center_y),
                    (center_x, center_y + patch_radius),
                    (center_x - patch_radius, center_y),
                ),
                dtype=np.int32,
            )
            cv2.fillConvexPoly(layer, diamond, color, cv2.LINE_8)
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
        start_x, start_y = float(start[0]), float(start[1])
        delta_x = float(end[0]) - start_x
        delta_y = float(end[1]) - start_y
        length = math.hypot(delta_x, delta_y)
        if length < 1.0:
            continue
        unit_x, unit_y = delta_x / length, delta_y / length
        distance = 0.0
        while distance < length - 1e-9:
            in_period = phase % period
            step = min(length - distance, period - in_period)
            if in_period < dash_px:
                visible = min(step, dash_px - in_period)
                if visible > 0.5:
                    dash_start_x = start_x + unit_x * distance
                    dash_start_y = start_y + unit_y * distance
                    output.append(
                        (
                            (round(dash_start_x), round(dash_start_y)),
                            (
                                round(dash_start_x + unit_x * visible),
                                round(dash_start_y + unit_y * visible),
                            ),
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
    offset_y: int = 0,
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
    y = frame.shape[0] - 14 - offset_y
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
    if guidance.direction in ('ROTATE LEFT', 'ROTATE RIGHT'):
        # Circular-arrow symbol means observed turn in place, not a
        # proposed Nav2 floor-path segment.
        clockwise = guidance.direction == 'ROTATE RIGHT'
        cv2.ellipse(
            frame, (icon_center, top + 39), (19, 19), 0,
            40, 315, color_bgr, 4, cv2.LINE_AA,
        )
        tip_x = icon_center + (14 if clockwise else -14)
        tip_y = top + 24
        cv2.arrowedLine(
            frame, (icon_center, top + 20), (tip_x, tip_y),
            color_bgr, 4, cv2.LINE_AA, tipLength=0.55,
        )
        label = guidance.direction
    elif guidance.direction == 'LEFT':
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
    scale = 0.51 if guidance.direction.startswith('ROTATE') else 0.62
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
    final_waypoint_status: str = "",
    navigation_status: str = "",
    debug_overlay: bool = True,
    path_alpha: float = 1.0,
    costmap_projection: ProjectedCostmap | None = None,
    costmap_alpha: float = 0.42,
    costmap_radius_px: int = 2,
    costmap_sync_hold_age_sec: float | None = None,
    costmap_explanation: CostmapExplanation | None = None,
    costmap_minimap: CostmapMinimap | None = None,
    costmap_minimap_size_px: int = 142,
    costmap_minimap_range_m: float = 3.0,
    costmap_minimap_status: str = "",
    costmap_alert: CostmapAlert | None = None,
    costmap_alert_history: CostmapAlertHistory | None = None,
    localization_warning: LocalizationWarning | None = None,
    robot_view_enabled: bool = False,
    robot_view_expanded: bool = False,
) -> np.ndarray:
    if state not in VALID_STATES:
        raise ValueError(f"unsupported viewer state: {state}")
    if not math.isfinite(path_alpha) or not 0.0 < path_alpha <= 1.0:
        raise ValueError("path_alpha must be finite and in (0, 1]")
    hold_visible = costmap_sync_hold_age_sec is not None
    effective_costmap_alpha = (
        min(costmap_alpha * 0.46, 0.24)
        if hold_visible else costmap_alpha
    )
    frame = draw_costmap_overlay(
        image,
        costmap_projection,
        alpha=effective_costmap_alpha,
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
        if final_waypoint_status:
            _draw_bottom_badge(
                frame, final_waypoint_status, (230, 230, 230),
                offset_y=29,
            )
    if navigation_status and state in (
        "ACTIVE", "PATH OUT OF VIEW", "REPLANNING", "NAVIGATION HOLD"
    ):
        _draw_navigation_status(frame, navigation_status)
    if (state == "NAVIGATION HOLD"
            and navigation_status == "ROUTE COMPLETE"):
        # Only an explicit FINISHED message authorizes this panel.
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        left = max(8, center_x - 229)
        right = min(frame.shape[1] - 8, center_x + 229)
        top = max(58, center_y - 43)
        bottom = min(frame.shape[0] - 50, center_y + 43)
        layer = frame.copy()
        cv2.rectangle(layer, (left, top), (right, bottom),
                      (16, 45, 24), -1)
        cv2.addWeighted(layer, 0.87, frame, 0.13, 0.0, frame)
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (50, 240, 100), 2)
        cv2.putText(
            frame, "ROUTE COMPLETE", (left + 38, top + 39),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 255, 130), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, "FOLLOWER CONFIRMED FINISHED",
            (left + 42, top + 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.40, (210, 245, 218), 1, cv2.LINE_AA,
        )
    if (not debug_overlay and not segment_list and not reference_list
            and state in (
                "REPLANNING", "NO PATH", "PATH STALE",
                "NAVIGATION HOLD", "TF UNAVAILABLE",
            ) and not navigation_status):
        _draw_bottom_badge(
            frame, "NO VERIFIED LIVE ROUTE", (195, 195, 195),
            align_right=True,
        )
    if debug_overlay:
        draw_costmap_explanation(frame, costmap_explanation)
    draw_costmap_alert(frame, costmap_alert)
    if costmap_alert is None:
        draw_costmap_alert_history(frame, costmap_alert_history)
    draw_localization_warning(frame, localization_warning)
    if robot_view_enabled:
        if robot_view_expanded and costmap_minimap is not None:
            draw_robot_view_panel(
                frame, costmap_minimap, range_m=costmap_minimap_range_m,
                status=costmap_minimap_status,
            )
        draw_robot_view_tab(frame, robot_view_expanded)
    label_color = (0, 220, 0) if state == "ACTIVE" else (0, 180, 255)
    detail_lines = (
        textwrap.wrap(detail, width=68)[:2]
        if debug_overlay and detail else []
    )
    box_bottom = 42 + 24 * len(detail_lines)
    cv2.rectangle(
        frame, (8, 8), (max(8, frame.shape[1] - 8), box_bottom), (20, 20, 20), -1
    )
    if debug_overlay:
        cv2.putText(frame, state, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, label_color, 2, cv2.LINE_AA)
    else:
        # Normal navigation stays in English. Japanese is reserved for
        # urgent Costmap and localization warning panels below.
        cv2.putText(
            frame, "KOKO NAVI", (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.66, (235, 235, 235), 2, cv2.LINE_AA,
        )
        if state in ("CAMERA UNAVAILABLE", "INVALID CONFIG") and frame.shape[1] >= 500:
            cv2.putText(
                frame,
                "CAMERA ERROR" if state == "CAMERA UNAVAILABLE" else "CONFIG ERROR",
                (235, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (30, 170, 255), 1, cv2.LINE_AA,
            )
    right_edge = frame.shape[1] - 18
    if display_mode_label and debug_overlay:
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
    if reference_list and debug_overlay:
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
    if costmap_projection is not None and debug_overlay:
        if hold_visible:
            hold_age = max(0.0, float(costmap_sync_hold_age_sec))
            label = f"SYNC HOLD | LAST SYNC {hold_age:.2f} s"
        else:
            label = "LOCAL COSTMAP | LIVE SYNC"
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
