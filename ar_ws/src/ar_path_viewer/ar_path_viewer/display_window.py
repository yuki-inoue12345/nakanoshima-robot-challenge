"""Resizable OpenCV display and mouse controls kept outside recorded pixels."""

from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "AR Path Viewer"
F11_KEY_CODES = {0x007A0000, 0x0000FFC8, 0x0100003A}
ACTION_VIEW_LOG = "view_log"
ACTION_COPY_LOG = "copy_log"
CONTROL_BAR_HEIGHT = 64
_JAPANESE_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def is_fullscreen_key(key_code: int) -> bool:
    if key_code is None or key_code < 0:
        return False
    return (int(key_code) & 0xFFFFFFFF) in F11_KEY_CODES


def fit_frame_to_window(
    frame: np.ndarray, window_width: int, window_height: int
) -> np.ndarray:
    """Resize with letterboxing so the source aspect ratio is preserved."""
    if frame.ndim != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("display frame must be a non-empty HxWxC image")
    width, height = int(window_width), int(window_height)
    if width < 1 or height < 1:
        return frame.copy()
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, min(width, int(round(frame.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(frame.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


@lru_cache(maxsize=1)
def _japanese_font():
    """Load an optional Japanese font once; callers retain an ASCII fallback."""
    freetype = getattr(cv2, "freetype", None)
    if freetype is None:
        return None
    for font_path in _JAPANESE_FONT_CANDIDATES:
        if not Path(font_path).is_file():
            continue
        try:
            font = freetype.createFreeType2()
            font.loadFontData(fontFileName=font_path, id=0)
            return font
        except (AttributeError, RuntimeError, cv2.error):
            continue
    return None


def _draw_button_label(
    canvas: np.ndarray,
    rectangle: tuple[int, int, int, int],
    japanese: str,
    fallback: str,
) -> None:
    x1, top, x2, bottom = rectangle
    font = _japanese_font()
    if font is not None:
        font_height = max(14, min(20, bottom - top - 14))
        try:
            (text_width, text_height), baseline = font.getTextSize(
                japanese, font_height, -1
            )
            origin = (
                x1 + max(2, (x2 - x1 - text_width) // 2),
                top + max(text_height + baseline, (bottom - top + text_height) // 2),
            )
            font.putText(
                canvas, japanese, origin, font_height, (245, 245, 245),
                -1, cv2.LINE_AA, False,
            )
            return
        except (AttributeError, RuntimeError, cv2.error):
            pass
    (text_width, text_height), _ = cv2.getTextSize(
        fallback, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    origin = (
        x1 + max(2, (x2 - x1 - text_width) // 2),
        top + max(text_height + 2, (bottom - top + text_height) // 2),
    )
    cv2.putText(
        canvas, fallback, origin, cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (245, 245, 245), 1, cv2.LINE_AA,
    )


def compose_display_frame(
    frame: np.ndarray,
    window_width: int,
    window_height: int,
    control_bar_height: int = CONTROL_BAR_HEIGHT,
    status_text: str = "",
    status_success: bool = True,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Fit a 4:3 video above a display-only control bar and return hit boxes."""
    width, height = max(1, int(window_width)), max(1, int(window_height))
    bar_height = min(max(36, int(control_bar_height)), max(36, height // 3))
    if height <= bar_height:
        bar_height = max(1, height // 3)
    video_height = max(1, height - bar_height)
    video = fit_frame_to_window(frame, width, video_height)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    canvas[:video_height] = video
    canvas[video_height:] = (36, 36, 36)
    cv2.line(canvas, (0, video_height), (width - 1, video_height), (90, 90, 90), 1)

    margin = max(6, min(16, width // 30))
    gap = max(6, min(14, width // 50))
    button_height = max(24, bar_height - 18)
    button_width = max(90, min(170, (width - margin * 2 - gap) // 3))
    y1 = video_height + max(4, (bar_height - button_height) // 2)
    y2 = min(height - 4, y1 + button_height)
    rectangles = {
        ACTION_VIEW_LOG: (margin, y1, margin + button_width, y2),
        ACTION_COPY_LOG: (
            margin + button_width + gap,
            y1,
            margin + button_width * 2 + gap,
            y2,
        ),
    }
    labels = {
        ACTION_VIEW_LOG: ("ログを見る", "LOG VIEW"),
        ACTION_COPY_LOG: ("ログをコピー", "LOG COPY"),
    }
    for action, (x1, top, x2, bottom) in rectangles.items():
        cv2.rectangle(canvas, (x1, top), (x2, bottom), (76, 76, 76), -1)
        cv2.rectangle(canvas, (x1, top), (x2, bottom), (150, 150, 150), 1)
        _draw_button_label(canvas, (x1, top, x2, bottom), *labels[action])
    if status_text:
        status_x = rectangles[ACTION_COPY_LOG][2] + gap + 4
        available = width - status_x - margin
        if available >= 70:
            max_letters = max(8, available // 8)
            text = status_text[:max_letters]
            color = (90, 230, 90) if status_success else (80, 170, 255)
            cv2.putText(
                canvas, text, (status_x, y1 + button_height // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA,
            )
    return canvas, rectangles


class DisplayWindow:
    """Own HighGUI state and degrade safely when optional backend APIs fail."""

    def __init__(
        self,
        normal_width: int = 960,
        normal_height: int = 720,
        cv_module: Any = cv2,
        clock=time.monotonic,
    ) -> None:
        if normal_width < 1 or normal_height < 1:
            raise ValueError("display dimensions must be positive")
        self.cv = cv_module
        self.normal_size = (int(normal_width), int(normal_height))
        self.fullscreen = False
        self.closed = False
        self._clock = clock
        self._button_rectangles: dict[str, tuple[int, int, int, int]] = {}
        self._pending_actions: list[str] = []
        self._status_text = ""
        self._status_success = True
        self._status_until = 0.0
        flags = getattr(self.cv, "WINDOW_NORMAL", 0) | getattr(
            self.cv, "WINDOW_KEEPRATIO", 0
        )
        self.cv.namedWindow(WINDOW_NAME, flags)
        self.cv.resizeWindow(WINDOW_NAME, *self.normal_size)
        set_mouse_callback = getattr(self.cv, "setMouseCallback", None)
        if set_mouse_callback is not None:
            set_mouse_callback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event, x, y, flags=None, param=None) -> None:
        if event != getattr(self.cv, "EVENT_LBUTTONUP", 4):
            return
        for action, (x1, y1, x2, y2) in self._button_rectangles.items():
            if x1 <= int(x) <= x2 and y1 <= int(y) <= y2:
                self._pending_actions.append(action)
                return

    def consume_actions(self) -> list[str]:
        actions = self._pending_actions
        self._pending_actions = []
        return actions

    def set_status(self, text: str, success: bool, duration_sec: float = 3.0) -> None:
        self._status_text = str(text)
        self._status_success = bool(success)
        self._status_until = self._clock() + max(0.1, float(duration_sec))

    def _image_rect_size(self) -> tuple[int, int]:
        get_rect = getattr(self.cv, "getWindowImageRect", None)
        if get_rect is None:
            return self.normal_size
        try:
            _, _, width, height = get_rect(WINDOW_NAME)
        except (AttributeError, RuntimeError, cv2.error):
            return self.normal_size
        if width < 1 or height < 1:
            return self.normal_size
        return int(width), int(height)

    def toggle_fullscreen(self) -> bool:
        target = not self.fullscreen
        try:
            value = (
                getattr(self.cv, "WINDOW_FULLSCREEN", 1)
                if target
                else getattr(self.cv, "WINDOW_NORMAL", 0)
            )
            self.cv.setWindowProperty(
                WINDOW_NAME, getattr(self.cv, "WND_PROP_FULLSCREEN", 0), value
            )
            if not target:
                self.cv.resizeWindow(WINDOW_NAME, *self.normal_size)
        except (AttributeError, RuntimeError, cv2.error):
            return False
        self.fullscreen = target
        return True

    def show_and_poll(self, frame: np.ndarray) -> int:
        width, height = self._image_rect_size()
        if self._clock() >= self._status_until:
            self._status_text = ""
        display_frame, self._button_rectangles = compose_display_frame(
            frame,
            width,
            height,
            status_text=self._status_text,
            status_success=self._status_success,
        )
        self.cv.imshow(WINDOW_NAME, display_frame)
        wait_key = getattr(self.cv, "waitKeyEx", None)
        if wait_key is None:
            wait_key = self.cv.waitKey
        key_code = int(wait_key(1))
        if is_fullscreen_key(key_code):
            self.toggle_fullscreen()
            return -1
        return key_code

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.cv.destroyWindow(WINDOW_NAME)
        except (AttributeError, RuntimeError, cv2.error):
            try:
                self.cv.destroyAllWindows()
            except (AttributeError, RuntimeError, cv2.error):
                pass
