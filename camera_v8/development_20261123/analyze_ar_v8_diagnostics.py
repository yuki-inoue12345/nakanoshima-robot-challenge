#!/usr/bin/env python3
"""Summarize one field run and select the first optimization branch.

The parser accepts both rclpy epoch timestamps and the Viewer's ISO-8601
session-log timestamps. It refuses to invent missing pipeline stages or run
phases and reports INSUFFICIENT DATA when evidence is incomplete.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import re
from pathlib import Path


ROS_STAMP = re.compile(r"\[(\d{9,}(?:\.\d+)?)\]")
ISO_STAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))"
)
FIELD_START = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def line_timestamp(line: str):
    match = ROS_STAMP.search(line)
    if match:
        return float(match.group(1))
    match = ISO_STAMP.search(line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(
            match.group(1).replace("Z", "+00:00")
        ).timestamp()
    except ValueError:
        return None


def parse_fields(line: str):
    """Keep compound metric summaries intact until the next ``name=``."""
    matches = list(FIELD_START.finditer(line))
    output = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        output[match.group(1)] = line[match.end():end].strip()
    return output


def records(path: Path, marker: str, start_epoch=None, end_epoch=None):
    output = []
    if not path.is_file():
        return output
    for line in path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        stamp = line_timestamp(line)
        if (
            stamp is not None
            and (start_epoch is None or stamp >= float(start_epoch))
            and (end_epoch is None or stamp <= float(end_epoch))
        ):
            output.append({"time": stamp, "fields": parse_fields(line), "line": line})
    return sorted(output, key=lambda row: row["time"])


def phase_events(owner_path: Path, start_epoch=None, end_epoch=None):
    """Use exact person-area transitions when the owner log provides them."""
    output = []
    if not owner_path.is_file():
        return output
    for line in owner_path.read_text(errors="replace").splitlines():
        stamp = line_timestamp(line)
        if (
            stamp is None
            or (start_epoch is not None and stamp < float(start_epoch))
            or (end_epoch is not None and stamp > float(end_epoch))
        ):
            continue
        if "V8 raw frames enabled by person_area arrival" in line:
            output.append((stamp, "search"))
        elif "V8 raw frames stopped at " in line:
            output.append((stamp, "post_search"))
    return sorted(output)


def nearest_phase(timestamp, owner, events=()):
    selected = [phase for event_time, phase in events if event_time <= timestamp]
    if selected:
        return selected[-1]
    # If collection starts during search, the first exact event may be a stop.
    # Fall back to the nearest sampled v8_active value before that event rather
    # than incorrectly labelling the whole prefix as normal navigation.
    previous = [item for item in owner if item["time"] <= timestamp]
    item = previous[-1] if previous else (owner[0] if owner else None)
    if item and item["fields"].get("v8_active") == "true":
        return "search"
    any_prior_search = any(
        row["time"] <= timestamp and row["fields"].get("v8_active") == "true"
        for row in owner
    )
    return "post_search" if any_prior_search else "normal"


def summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return {
        "count": len(values),
        "p50": values[(len(values) - 1) // 2],
        "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)],
        "max": values[-1],
    }


def field_summary(rows, field):
    return summary(number(row["fields"].get(field)) for row in rows)


def compound_value(row, field, quantile="p95"):
    text = row["fields"].get(field, "")
    match = re.search(rf"(?:^|[\s,]){re.escape(quantile)}:([-+0-9.eE]+)", text)
    return number(match.group(1)) if match else None


def compound_summary(rows, field, quantile="p95"):
    return summary(compound_value(row, field, quantile) for row in rows)


def split(rows, owner, events=()):
    result = {name: [] for name in ("normal", "search", "post_search")}
    for row in rows:
        result[nearest_phase(row["time"], owner, events)].append(row)
    return result


def _p50(data):
    return None if data is None else data["p50"]


def latency_assessment(viewer_rows):
    values = [
        (row["time"], compound_value(row, "display_age_ms", "p95"))
        for row in viewer_rows
    ]
    values = [(stamp, value) for stamp, value in values if value is not None]
    if len(values) < 3:
        return "INSUFFICIENT DATA: fewer than 3 display-age samples"
    first = values[0][1]
    last = values[-1][1]
    positive_steps = sum(
        later[1] > earlier[1] + 5.0
        for earlier, later in zip(values, values[1:])
    )
    if last - first > 100.0 and positive_steps >= len(values) // 2:
        return (
            "LATENCY ACCUMULATING: display_age_ms p95 increased "
            f"from {first:.1f} to {last:.1f} ms"
        )
    overall = summary(value for _, value in values)
    if overall and overall["p95"] <= 150.0:
        return "LATENCY BOUNDED: display_age_ms p95 is at most 150 ms"
    if overall and overall["p95"] > 250.0:
        return "HIGH LATENCY/JITTER: display_age_ms p95 exceeds 250 ms"
    return "LATENCY INDETERMINATE: inspect exact samples and pidstat"


def choose_candidate(owner_rows, viewer_rows, v8_rows, events=()):
    owner_by_phase = split(owner_rows, owner_rows, events)
    viewer_by_phase = split(viewer_rows, owner_rows, events)
    v8_by_phase = split(v8_rows, owner_rows, events)
    missing = []
    for label, rows in (("owner", owner_rows), ("viewer", viewer_rows), ("v8", v8_rows)):
        if not rows:
            missing.append(label)
    for phase in ("normal", "search", "post_search"):
        if not owner_by_phase[phase] or not viewer_by_phase[phase]:
            missing.append(f"{phase} phase")
    if not v8_by_phase["search"]:
        missing.append("active-search V8 phase")
    if missing:
        return "INSUFFICIENT DATA: missing " + ", ".join(dict.fromkeys(missing))

    normal_owner = field_summary(owner_by_phase["normal"], "ar_publish_fps")
    normal_callback = field_summary(
        viewer_by_phase["normal"], "ros_callback_fps_est")
    normal_gui = field_summary(viewer_by_phase["normal"], "gui_new_fps_est")
    search_owner_ar = field_summary(owner_by_phase["search"], "ar_publish_fps")
    search_callback = field_summary(
        viewer_by_phase["search"], "ros_callback_fps_est")
    search_gui = field_summary(viewer_by_phase["search"], "gui_new_fps_est")
    search_owner_v8 = field_summary(owner_by_phase["search"], "v8_publish_fps")
    v8_receive = field_summary(v8_by_phase["search"], "receive_fps")
    v8_consume = field_summary(v8_by_phase["search"], "consume_fps")
    yolo = field_summary(v8_by_phase["search"], "yolo_fps")
    required = (normal_owner, normal_callback, normal_gui, search_owner_ar,
                search_callback, search_gui, search_owner_v8, v8_receive,
                v8_consume, yolo)
    if any(item is None for item in required):
        return "INSUFFICIENT DATA: one or more required rate fields are absent"

    if _p50(normal_owner) < 27.0:
        return "FIRST: camera owner capture/serialization/publish stage"
    if _p50(normal_callback) < 0.8 * _p50(normal_owner):
        return "FIRST: owner-to-AR DDS delivery/callback stage"
    if _p50(normal_gui) < 0.8 * _p50(normal_callback):
        return "FIRST: Viewer render/GUI presentation stage"
    if _p50(v8_receive) < 0.8 * _p50(search_owner_v8):
        return "FIRST: owner-to-V8 DDS delivery/callback contention"
    if _p50(v8_consume) < 0.8 * _p50(v8_receive):
        return "FIRST: V8 frame consumption/YOLO/VLM contention"
    if (
        _p50(search_gui) < 0.75 * _p50(normal_gui)
        or _p50(search_callback) < 0.75 * _p50(normal_callback)
        or _p50(search_owner_ar) < 0.75 * _p50(normal_owner)
    ):
        return "FIRST: search-correlated system contention"
    latency = latency_assessment(viewer_rows)
    if latency.startswith(("LATENCY ACCUMULATING", "HIGH LATENCY")):
        return "FIRST: latency/jitter reduction before nominal FPS"
    return "NO SINGLE BOTTLENECK: compare phase CPU/RSS and p95 stage durations"


def build_report(
    owner_path: Path,
    viewer_path: Path,
    v8_path: Path,
    *,
    start_epoch=None,
    end_epoch=None,
):
    owner = records(
        owner_path, "camera_owner_pipeline", start_epoch, end_epoch)
    viewer = records(viewer_path, "camera_pipeline ", start_epoch, end_epoch)
    v8 = records(v8_path, "v8_image_pipeline", start_epoch, end_epoch)
    events = phase_events(owner_path, start_epoch, end_epoch)
    owner_phase = split(owner, owner, events)
    viewer_phase = split(viewer, owner, events)
    v8_phase = split(v8, owner, events)
    report = {
        "samples": {"owner": len(owner), "viewer": len(viewer), "v8": len(v8)},
        "phase_source": "exact owner transitions" if events else "5-second owner samples",
        "phases": {},
    }
    for phase in ("normal", "search", "post_search"):
        report["phases"][phase] = {
            "owner_ar_publish_fps": field_summary(owner_phase[phase], "ar_publish_fps"),
            "owner_v8_publish_fps": field_summary(owner_phase[phase], "v8_publish_fps"),
            "v8_receive_fps": field_summary(v8_phase[phase], "receive_fps"),
            "v8_consume_fps": field_summary(v8_phase[phase], "consume_fps"),
            "v8_yolo_fps": field_summary(v8_phase[phase], "yolo_fps"),
            "owner_cpu_pct": field_summary(owner_phase[phase], "cpu_pct"),
            "viewer_cpu_pct": field_summary(viewer_phase[phase], "process_cpu_pct"),
            "viewer_callback_fps": field_summary(
                viewer_phase[phase], "ros_callback_fps_est"),
            "viewer_gui_new_fps": field_summary(
                viewer_phase[phase], "gui_new_fps_est"),
            "viewer_display_age_ms_p95": compound_summary(
                viewer_phase[phase], "display_age_ms", "p95"),
            "v8_cpu_pct": field_summary(v8_phase[phase], "cpu_pct"),
        }
    report["latency_assessment"] = latency_assessment(viewer)
    report["recommended_first_branch"] = choose_candidate(owner, viewer, v8, events)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-log", type=Path, required=True)
    parser.add_argument("--viewer-log", type=Path, required=True)
    parser.add_argument("--v8-log", type=Path, required=True)
    parser.add_argument("--start-epoch", type=float)
    parser.add_argument("--end-epoch", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.owner_log,
        args.viewer_log,
        args.v8_log,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
