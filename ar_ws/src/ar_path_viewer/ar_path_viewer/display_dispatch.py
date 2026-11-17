"""Bounded exchange from the Viewer worker to the main-thread GUI."""

from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DisplayFrameMetadata:
    """Timing attached to the exact pixels handed to HighGUI."""

    source_timestamp_monotonic_sec: float | None
    published_monotonic_sec: float


class LatestDisplayFrame:
    """Keep only the newest completed frame for nonblocking GUI delivery."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._frame: Optional[np.ndarray] = None
        self._metadata: DisplayFrameMetadata | None = None
        self._closed = False

    def publish(
        self, frame: np.ndarray,
        source_timestamp_monotonic_sec: float | None = None,
    ) -> bool:
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
            self._metadata = DisplayFrameMetadata(
                source_timestamp_monotonic_sec=(
                    None if source_timestamp_monotonic_sec is None
                    else float(source_timestamp_monotonic_sec)
                ),
                published_monotonic_sec=time.monotonic(),
            )
            self._condition.notify_all()
            return True

    def wait_for_new(
        self, after_sequence: int, timeout_sec: float
    ) -> Tuple[int, Optional[np.ndarray]]:
        sequence, frame, _ = self.wait_for_new_with_metadata(
            after_sequence, timeout_sec
        )
        return sequence, frame

    def wait_for_new_with_metadata(
        self, after_sequence: int, timeout_sec: float
    ) -> tuple[int, Optional[np.ndarray], DisplayFrameMetadata | None]:
        if timeout_sec < 0.0:
            raise ValueError("display wait timeout must not be negative")
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._sequence > after_sequence,
                timeout=float(timeout_sec),
            )
            if self._sequence <= after_sequence or self._frame is None:
                return self._sequence, None, None
            return self._sequence, self._frame, self._metadata

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frame = None
            self._metadata = None
            self._condition.notify_all()
