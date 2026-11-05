"""OpenCV rendering for viewer states and real projected path segments."""

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
    label_color = (0, 220, 0) if state == "ACTIVE" else (0, 180, 255)
    cv2.rectangle(frame, (8, 8), (370, 66 if detail else 42), (20, 20, 20), -1)
    cv2.putText(frame, state, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                label_color, 2, cv2.LINE_AA)
    if detail:
        cv2.putText(frame, detail[:48], (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return frame
