"""Resizable OpenCV display that never changes recording or projection pixels."""

from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "AR Path Viewer"
F11_KEY_CODES = {0x007A0000, 0x0000FFC8, 0x0100003A}


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


class DisplayWindow:
    """Own HighGUI state and degrade safely when optional backend APIs fail."""

    def __init__(
        self,
        normal_width: int = 960,
        normal_height: int = 720,
        cv_module: Any = cv2,
    ) -> None:
        if normal_width < 1 or normal_height < 1:
            raise ValueError("display dimensions must be positive")
        self.cv = cv_module
        self.normal_size = (int(normal_width), int(normal_height))
        self.fullscreen = False
        self.closed = False
        flags = getattr(self.cv, "WINDOW_NORMAL", 0) | getattr(
            self.cv, "WINDOW_KEEPRATIO", 0
        )
        self.cv.namedWindow(WINDOW_NAME, flags)
        self.cv.resizeWindow(WINDOW_NAME, *self.normal_size)

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
        display_frame = fit_frame_to_window(frame, width, height)
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
