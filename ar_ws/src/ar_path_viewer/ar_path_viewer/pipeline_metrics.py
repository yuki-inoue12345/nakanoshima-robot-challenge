"""Small, dependency-free stage metrics for camera pipeline diagnostics.

The collector intentionally never owns or copies image data.  Producers only
record counters and scalar durations.  A snapshot resets the interval values,
which keeps both CPU use and memory bounded during long field runs.
"""
from __future__ import annotations

from collections import defaultdict
import math
import os
import threading
import time
from typing import Callable


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


class StageMetrics:
    """Thread-safe interval counters and bounded scalar observations."""

    def __init__(self, *, enabled: bool = True, max_samples: int = 600):
        self.enabled = bool(enabled)
        self.max_samples = max(20, int(max_samples))
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._counters: dict[str, int] = defaultdict(int)
        self._samples: dict[str, list[float]] = defaultdict(list)

    def count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[str(name)] += int(amount)

    def observe(self, name: str, value: float | None) -> None:
        if not self.enabled or value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            return
        with self._lock:
            samples = self._samples[str(name)]
            if len(samples) >= self.max_samples:
                # Evenly downsample the old interval instead of growing RAM.
                del samples[::2]
            samples.append(value)

    def snapshot(self, *, reset: bool = True, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            elapsed = max(1e-9, now - self._started)
            counters = dict(self._counters)
            samples = {key: tuple(values) for key, values in self._samples.items()}
            if reset:
                self._started = now
                self._counters.clear()
                self._samples.clear()
        summaries = {}
        for key, values_tuple in samples.items():
            values = list(values_tuple)
            summaries[key] = {
                "count": len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values, default=0.0),
                "mean": sum(values) / max(1, len(values)),
            }
        return {
            "interval_sec": elapsed,
            "counters": counters,
            "rates": {key: value / elapsed for key, value in counters.items()},
            "samples": summaries,
        }


class ProcessResourceSampler:
    """Measure this process only, without spawning ps/pidstat or image copies."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.process_time,
    ):
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._last_wall = clock()
        self._last_cpu = cpu_clock()

    def sample(self) -> dict[str, float | int]:
        wall = self._clock()
        cpu = self._cpu_clock()
        wall_delta = max(1e-9, wall - self._last_wall)
        cpu_delta = max(0.0, cpu - self._last_cpu)
        self._last_wall = wall
        self._last_cpu = cpu
        rss_bytes = 0
        threads = 0
        try:
            with open("/proc/self/statm", encoding="ascii") as stream:
                fields = stream.read().split()
            rss_bytes = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/self/status", encoding="ascii") as stream:
                fields = stream.read().splitlines()
            line = next(item for item in fields if item.startswith("Threads:"))
            threads = int(line.split(":", 1)[1].strip())
        except (OSError, ValueError, StopIteration):
            pass
        return {
            "cpu_pct": 100.0 * cpu_delta / wall_delta,
            "rss_mib": rss_bytes / (1024.0 * 1024.0),
            "threads": threads,
        }


def metric_ms(snapshot: dict, name: str, field: str = "p95") -> float:
    return 1000.0 * float(snapshot.get("samples", {}).get(name, {}).get(field, 0.0))
