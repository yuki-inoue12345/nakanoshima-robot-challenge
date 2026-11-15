"""Bounded exchange from the Viewer worker to the main-thread GUI."""

import threading
from typing import Optional, Tuple

import numpy as np


class LatestDisplayFrame:
    """Keep only the newest completed frame for nonblocking GUI delivery."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._frame: Optional[np.ndarray] = None
        self._closed = False

    def publish(self, frame: np.ndarray) -> bool:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("display frame must have shape (H, W, 3)")
        # Copy before taking the lock. The producer never mutates the object
        # visible to the GUI, and the critical section remains very short.
        immutable_copy = np.array(array, copy=True)
        with self._condition:
            if self._closed:
                return False
            self._sequence += 1
            self._frame = immutable_copy
            self._condition.notify_all()
            return True

    def wait_for_new(
        self, after_sequence: int, timeout_sec: float
    ) -> Tuple[int, Optional[np.ndarray]]:
        if timeout_sec < 0.0:
            raise ValueError("display wait timeout must not be negative")
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._sequence > after_sequence,
                timeout=float(timeout_sec),
            )
            if self._sequence <= after_sequence or self._frame is None:
                return self._sequence, None
            return self._sequence, self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frame = None
            self._condition.notify_all()
