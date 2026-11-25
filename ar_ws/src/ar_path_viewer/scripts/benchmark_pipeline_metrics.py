#!/usr/bin/env python3
"""Measure scalar instrumentation overhead without camera/image subscribers."""
import argparse
import json
import time

from ar_path_viewer.pipeline_metrics import StageMetrics


def run(enabled: bool, frames: int):
    metrics = StageMetrics(enabled=enabled)
    started = time.perf_counter()
    for index in range(frames):
        metrics.count('received')
        metrics.count('discarded', int(index % 7 == 0))
        metrics.observe('callback_sec', 0.0005 + (index % 5) * 0.00001)
        metrics.observe('render_sec', 0.004 + (index % 11) * 0.00002)
        if index % 150 == 149:  # one five-second snapshot at synthetic 30 Hz
            metrics.snapshot()
    elapsed = time.perf_counter() - started
    return {"enabled": enabled, "frames": frames, "elapsed_sec": elapsed,
            "microseconds_per_frame": 1e6 * elapsed / frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=18000)
    args = parser.parse_args()
    disabled = run(False, args.frames)
    enabled = run(True, args.frames)
    result = {"disabled": disabled, "enabled": enabled,
              "added_microseconds_per_frame":
                  enabled["microseconds_per_frame"] - disabled["microseconds_per_frame"]}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
