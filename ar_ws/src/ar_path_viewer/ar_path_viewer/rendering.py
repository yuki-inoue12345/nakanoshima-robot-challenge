"""OpenCV rendering for viewer states and real projected path segments."""

import math
import textwrap
from typing import Iterable, Tuple

import cv2
import numpy as np


VALID_STATES = {
    "ACTIVE",
    "NO PATH",
    "PATH STALE",
    "TF UNAVAILABLE",
    "CAMERA UNAVAILABLE",
    "PATH OUT OF VIEW",
    "LOCALIZING",
    "LOCALIZATION UNSTABLE",
    "INVALID CONFIG",
}


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


def render_frame(
    image: np.ndarray,
    segments: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
    state: str,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
    detail: str = "",
) -> np.ndarray:
    if state not in VALID_STATES:
        raise ValueError(f"unsupported viewer state: {state}")
    frame = image.copy()
    segment_list = list(segments)
    if state == "ACTIVE":
        for start, end in segment_list:
            cv2.line(frame, start, end, color_bgr, thickness, cv2.LINE_AA)
        for start, end in direction_arrow_segments(segment_list):
            cv2.arrowedLine(
                frame, start, end, color_bgr, thickness, cv2.LINE_AA, tipLength=0.4
            )
    label_color = (0, 220, 0) if state == "ACTIVE" else (0, 180, 255)
    detail_lines = textwrap.wrap(detail, width=68)[:2] if detail else []
    box_bottom = 42 + 24 * len(detail_lines)
    cv2.rectangle(
        frame, (8, 8), (max(8, frame.shape[1] - 8), box_bottom), (20, 20, 20), -1
    )
    cv2.putText(frame, state, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                label_color, 2, cv2.LINE_AA)
    for index, line in enumerate(detail_lines):
        cv2.putText(frame, line, (18, 58 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return frame
