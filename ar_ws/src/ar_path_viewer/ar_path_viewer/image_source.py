"""Pluggable OpenCV image inputs with a deterministic simulation fallback."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ImageSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimulationSource(ImageSource):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._frame = self._make_background()

    def _make_background(self) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon = int(self.height * 0.40)
        image[:horizon] = (185, 145, 95)
        for row in range(horizon, self.height):
            fraction = (row - horizon) / max(1, self.height - horizon - 1)
            image[row, :] = (
                int(70 - 25 * fraction),
                int(80 - 30 * fraction),
                int(75 - 25 * fraction),
            )
        cv2.line(image, (0, horizon), (self.width - 1, horizon), (220, 210, 180), 2)
        cv2.putText(
            image,
            "SIMULATED CAMERA - NOT C920",
            (12, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return image

    def read(self) -> np.ndarray:
        return self._frame.copy()


class StaticImageSource(ImageSource):
    def __init__(self, path: str, width: int, height: int):
        image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image: {path}")
        self._image = cv2.resize(image, (width, height))

    def read(self) -> np.ndarray:
        return self._image.copy()


class CaptureSource(ImageSource):
    def __init__(self, source: str | int, width: int, height: int, loop: bool = False):
        self._source = source
        self._loop = loop
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            self._capture.release()
            raise OSError(f"cannot open capture source: {source}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._capture.read()
        if ok:
            return frame
        if self._loop and isinstance(self._source, str):
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                return frame
        return None

    def close(self) -> None:
        self._capture.release()


def make_image_source(kind: str, uri: str, width: int, height: int) -> ImageSource:
    normalized = kind.strip().lower()
    if normalized == "simulation":
        return SimulationSource(width, height)
    if normalized == "image":
        return StaticImageSource(uri, width, height)
    if normalized == "video":
        return CaptureSource(uri, width, height, loop=True)
    if normalized == "camera":
        source: str | int = int(uri) if uri.strip().isdigit() else uri
        return CaptureSource(source, width, height)
    raise ValueError(f"unsupported image input type: {kind}")
