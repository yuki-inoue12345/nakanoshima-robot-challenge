"""Configurable offline, publisher-free route projection and video generation."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .image_source import ImageSource, make_image_source
from .path_adapter import PathStore
from .projection import CameraIntrinsics, CameraMount, project_clipped_segments
from .rendering import render_frame
from .route_io import (
    WaypointRecord,
    load_map_metadata,
    load_waypoint_records,
    read_map_image,
    validate_waypoints_on_map,
    waypoint_coordinates,
)


@dataclass(frozen=True)
class OfflineSettings:
    waypoint_yaml: Path
    map_yaml: Path
    map_image_override: Path | None
    output_dir: Path
    output_video: Path
    image_input_type: str
    image_input_uri: str
    speed_mps: float
    fps: float
    stop_duration_sec: float
    intrinsics: CameraIntrinsics
    mount: CameraMount
    near_clip_m: float = 0.10
    path_timeout_sec: float = 6.5

    def validate(self) -> None:
        self.intrinsics.validate()
        self.mount.validate()
        if self.speed_mps <= 0.0 or not math.isfinite(self.speed_mps):
            raise ValueError("speed_mps must be finite and positive")
        if self.fps <= 0.0 or not math.isfinite(self.fps):
            raise ValueError("fps must be finite and positive")
        if self.stop_duration_sec < 0.0 or not math.isfinite(self.stop_duration_sec):
            raise ValueError("stop_duration_sec must be finite and non-negative")
        if self.near_clip_m <= 0.0 or not math.isfinite(self.near_clip_m):
            raise ValueError("near_clip_m must be finite and positive")
        if self.path_timeout_sec <= 0.0 or not math.isfinite(self.path_timeout_sec):
            raise ValueError("path_timeout_sec must be finite and positive")
        if self.image_input_type not in {"simulation", "image", "video", "camera"}:
            raise ValueError(f"unsupported image input type: {self.image_input_type}")
        if self.image_input_type != "simulation" and not self.image_input_uri:
            raise ValueError(f"input URI is required for {self.image_input_type}")


def interpolate_route(waypoints: np.ndarray, spacing: float = 0.06) -> Tuple[np.ndarray, List[int]]:
    if spacing <= 0.0 or not math.isfinite(spacing):
        raise ValueError("spacing must be finite and positive")
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("at least two planar waypoints are required")
    if not np.isfinite(points).all():
        raise ValueError("waypoint coordinates must be finite")
    samples = [points[0]]
    waypoint_indices = [0]
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / spacing)))
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            samples.append(start + fraction * (end - start))
        waypoint_indices.append(len(samples) - 1)
    return np.asarray(samples), waypoint_indices


def segment_headings(waypoints: np.ndarray) -> np.ndarray:
    differences = np.diff(waypoints, axis=0)
    return np.arctan2(differences[:, 1], differences[:, 0])


def wrapped_differences(angles: np.ndarray) -> np.ndarray:
    raw = np.diff(angles)
    return np.arctan2(np.sin(raw), np.cos(raw))


def build_frame_schedule(
    route_length: int,
    waypoint_indices: Sequence[int],
    records: Sequence[WaypointRecord],
    fps: float,
    stop_duration_sec: float,
) -> Tuple[List[int], float]:
    """Return route indices per frame, including current follower stop/wait semantics."""
    boundary_records = {
        waypoint_indices[index]: record for index, record in enumerate(records)
    }
    schedule: List[int] = []
    dwell_total = 0.0
    for route_index in range(route_length):
        schedule.append(route_index)
        # The first waypoint is the starting pose, not a newly reached destination.
        record = boundary_records.get(route_index) if route_index > 0 else None
        if record is None:
            continue
        dwell = record.wait_time
        if record.stop and dwell <= 0.0:
            dwell = stop_duration_sec
        if dwell > 0.0:
            frame_count = int(round(dwell * fps))
            schedule.extend([route_index] * frame_count)
            dwell_total += frame_count / fps
    return schedule, dwell_total


def _target_route_indices(
    route: np.ndarray,
    waypoint_indices: Sequence[int],
    turn_deltas: np.ndarray,
) -> Dict[str, int]:
    def approach(turn_index: int, distance: float = 0.70) -> int:
        cursor = waypoint_indices[turn_index + 1]
        travelled = 0.0
        while cursor > 0 and travelled < distance:
            travelled += float(np.linalg.norm(route[cursor] - route[cursor - 1]))
            cursor -= 1
        return cursor

    targets = {
        "straight": max(0, waypoint_indices[1] // 2),
        "path_update": int(round((len(route) - 1) * 0.25)),
        "final_waypoint": len(route) - 1,
    }
    left = [index for index, value in enumerate(turn_deltas) if value > math.radians(0.5)]
    right = [index for index, value in enumerate(turn_deltas) if value < math.radians(-0.5)]
    if left:
        targets["left_turn"] = approach(max(left, key=lambda index: turn_deltas[index]))
    if right:
        targets["right_turn"] = approach(min(right, key=lambda index: turn_deltas[index]))
    return targets


def _yaw_for_route_index(
    route: np.ndarray,
    route_index: int,
    waypoint_indices: Sequence[int],
    records: Sequence[WaypointRecord],
) -> float:
    boundary = {value: index for index, value in enumerate(waypoint_indices)}
    if route_index in boundary:
        record = records[boundary[route_index]]
        if record.stop or record.wait_time > 0.0 or route_index == len(route) - 1:
            return record.angle_radians
    for next_index in range(route_index + 1, len(route)):
        delta = route[next_index] - route[route_index]
        if np.linalg.norm(delta) > 1e-9:
            return math.atan2(delta[1], delta[0])
    for previous_index in range(route_index - 1, -1, -1):
        delta = route[route_index] - route[previous_index]
        if np.linalg.norm(delta) > 1e-9:
            return math.atan2(delta[1], delta[0])
    return records[-1].angle_radians


def _write_map_overview(
    output: Path,
    map_image: np.ndarray,
    pixels: np.ndarray,
    records: Sequence[WaypointRecord],
) -> None:
    canvas = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    rounded = np.rint(pixels).astype(np.int32)
    cv2.polylines(canvas, [rounded], False, (0, 255, 255), 2, cv2.LINE_AA)
    for point, record in zip(rounded, records):
        pixel = (int(point[0]), int(point[1]))
        cv2.circle(canvas, pixel, 4, (0, 0, 255), -1)
        cv2.putText(canvas, str(record.number), (pixel[0] + 5, pixel[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"failed to write {output}")


def _background(source: ImageSource, intrinsics: CameraIntrinsics) -> np.ndarray | None:
    image = source.read()
    if image is None:
        return None
    if image.shape[:2] != (intrinsics.height, intrinsics.width):
        image = cv2.resize(image, (intrinsics.width, intrinsics.height))
    return image


def _open_writer(settings: OfflineSettings):
    settings.output_video.parent.mkdir(parents=True, exist_ok=True)
    suffix = settings.output_video.suffix.lower()
    codec = "mp4v" if suffix in {".mp4", ".m4v"} else "MJPG"
    return cv2.VideoWriter(
        str(settings.output_video), cv2.VideoWriter_fourcc(*codec), settings.fps,
        (settings.intrinsics.width, settings.intrinsics.height),
    )


def run_simulation(settings: OfflineSettings) -> Dict[str, object]:
    settings.validate()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_waypoint_records(settings.waypoint_yaml)
    waypoints = waypoint_coordinates(records)
    if len(waypoints) < 2:
        raise ValueError("Offline simulation requires at least two waypoints")
    metadata = load_map_metadata(settings.map_yaml, settings.map_image_override)
    map_image = read_map_image(metadata)
    map_pixels = validate_waypoints_on_map(records, metadata, map_image.shape)

    spacing = settings.speed_mps / settings.fps
    route_xy, waypoint_indices = interpolate_route(waypoints, spacing)
    route_xyz = np.column_stack((route_xy, np.zeros(len(route_xy))))
    headings = segment_headings(waypoints)
    turns = wrapped_differences(headings)
    targets = _target_route_indices(route_xy, waypoint_indices, turns)
    schedule, dwell_total = build_frame_schedule(
        len(route_xy), waypoint_indices, records, settings.fps, settings.stop_duration_sec
    )

    source = make_image_source(
        settings.image_input_type, settings.image_input_uri,
        settings.intrinsics.width, settings.intrinsics.height,
    )
    writer = _open_writer(settings)
    video_available = writer.isOpened()
    fallback_dir = settings.output_dir / "full_route_frames"
    if not video_available:
        writer.release()
        fallback_dir.mkdir(exist_ok=True)

    simulated_now = [0.0]
    store = PathStore(settings.path_timeout_sec, clock=lambda: simulated_now[0])
    saved: Dict[str, str] = {}
    projected_frames = 0
    out_of_view_frames = 0
    camera_unavailable_frames = 0
    exceptions: List[str] = []
    updates = 0
    update_period_frames = max(1, int(round(settings.fps)))
    labels = {value: key.replace("_", " ").upper() for key, value in targets.items()}
    last_background = np.zeros(
        (settings.intrinsics.height, settings.intrinsics.width, 3), dtype=np.uint8
    )

    try:
        for frame_index, route_index in enumerate(schedule):
            simulated_now[0] = frame_index / settings.fps
            if frame_index % update_period_frames == 0 or route_index == targets["path_update"]:
                store.replace(route_xyz[route_index:], "map")
                updates += 1
            _, snapshot = store.state()
            assert snapshot is not None
            background = _background(source, settings.intrinsics)
            if background is None:
                camera_unavailable_frames += 1
                frame = render_frame(last_background, [], "CAMERA UNAVAILABLE")
            else:
                cv2.putText(
                    background,
                    f"OFFLINE {settings.image_input_type.upper()} - NOT LIVE C920/NAV2 PLAN",
                    (12, settings.intrinsics.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                last_background = background
                yaw = _yaw_for_route_index(route_xy, route_index, waypoint_indices, records)
                try:
                    segments = project_clipped_segments(
                        snapshot.points,
                        (route_xy[route_index, 0], route_xy[route_index, 1], yaw),
                        settings.mount,
                        settings.intrinsics,
                        settings.near_clip_m,
                    )
                    state = "ACTIVE" if segments else "PATH OUT OF VIEW"
                    projected_frames += int(bool(segments))
                    out_of_view_frames += int(not segments)
                    prefix = labels.get(route_index, "ROUTE")
                    detail = (
                        f"{prefix} | frame={frame_index} | "
                        f"{100.0*route_index/(len(route_xy)-1):.1f}%"
                    )
                    frame = render_frame(background, segments, state, detail=detail)
                except Exception as exc:
                    exceptions.append(f"frame {frame_index}: {type(exc).__name__}: {exc}")
                    frame = render_frame(background, [], "INVALID CONFIG", detail=str(exc))

            if video_available:
                writer.write(frame)
            elif frame_index % update_period_frames == 0:
                cv2.imwrite(str(fallback_dir / f"frame_{frame_index:05d}.png"), frame)
            for label, target in targets.items():
                if route_index == target and label not in saved:
                    destination = settings.output_dir / f"{label}.png"
                    if not cv2.imwrite(str(destination), frame):
                        raise OSError(f"failed to write {destination}")
                    saved[label] = str(destination)
    finally:
        source.close()
        if video_available:
            writer.release()

    simulated_now[0] += settings.path_timeout_sec + 1e-6
    stale_state, _ = store.state()
    stale_frame = render_frame(
        last_background, [], stale_state,
        detail=f"route hidden after {settings.path_timeout_sec:g} s timeout",
    )
    stale_path = settings.output_dir / "path_stale.png"
    if not cv2.imwrite(str(stale_path), stale_frame):
        raise OSError(f"failed to write {stale_path}")
    saved["path_stale"] = str(stale_path)

    overview_path = settings.output_dir / "route_map_overview.png"
    _write_map_overview(overview_path, map_image, map_pixels, records)
    saved["route_map_overview"] = str(overview_path)
    if video_available:
        saved["video"] = str(settings.output_video)
    else:
        saved["frame_sequence"] = str(fallback_dir)

    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    report: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "waypoint_yaml": str(settings.waypoint_yaml),
        "map": {
            "yaml": str(metadata.yaml_path),
            "image": str(metadata.image_path),
            "resolution": metadata.resolution,
            "origin": list(metadata.origin),
            "image_width": int(map_image.shape[1]),
            "image_height": int(map_image.shape[0]),
        },
        "waypoint_count": int(len(records)),
        "waypoints": [record.report_dict() for record in records],
        "segment_lengths_m": [float(value) for value in segment_lengths],
        "total_interpolated_distance_m": float(segment_lengths.sum()),
        "route_sample_count": int(len(route_xy)),
        "video_frame_count": int(len(schedule)),
        "speed_mps": settings.speed_mps,
        "fps": settings.fps,
        "simulated_dwell_sec": dwell_total,
        "start_end_distance_m": float(np.linalg.norm(waypoints[-1] - waypoints[0])),
        "returns_near_start": bool(np.linalg.norm(waypoints[-1] - waypoints[0]) < 0.5),
        "turn_deltas_deg": [float(math.degrees(value)) for value in turns],
        "left_turn_count": int(np.count_nonzero(turns > math.radians(0.5))),
        "right_turn_count": int(np.count_nonzero(turns < math.radians(-0.5))),
        "projected_frame_count": projected_frames,
        "out_of_view_frame_count": out_of_view_frames,
        "camera_unavailable_frame_count": camera_unavailable_frames,
        "completed_all_waypoints": True,
        "final_waypoint_xy": [float(value) for value in waypoints[-1]],
        "path_update_count": updates,
        "stale_state_after_timeout": stale_state,
        "exceptions": exceptions,
        "camera": {
            "mount": settings.mount.__dict__,
            "intrinsics": settings.intrinsics.__dict__,
            "calibrated": False,
            "input_type": settings.image_input_type,
            "input_uri": settings.image_input_uri,
        },
        "person_area_behavior": (
            "Markers are parsed and reported; Camera person-search behavior is not simulated."
        ),
        "scope_note": (
            "Waypoint-to-waypoint straight interpolation only; synthetic input is not real "
            "camera footage or a Nav2 plan and does not prove obstacle avoidance/traversability."
        ),
        "artifacts": saved,
    }
    log_path = settings.output_dir / "full_route_simulation.json"
    saved["log"] = str(log_path)
    with log_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def _load_config(path: Path | None) -> Tuple[Mapping[str, Any], Path]:
    if path is None:
        return {}, Path.cwd()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Offline config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("Offline config must contain a mapping")
    values = document.get("offline_simulation", document)
    if not isinstance(values, dict):
        raise ValueError("offline_simulation must be a mapping")
    return values, resolved.parent


def _nested(values: Mapping[str, Any], section: str, key: str, default=None):
    nested = values.get(section, {})
    if not isinstance(nested, dict):
        raise ValueError(f"offline_simulation.{section} must be a mapping")
    return nested.get(key, default)


def _path_value(value, base: Path, name: str, required: bool = False) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{name} is required (CLI or config file)")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def parse_settings(args=None) -> OfflineSettings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--waypoints", type=Path)
    parser.add_argument("--map-yaml", type=Path)
    parser.add_argument("--map-image", "--map-pgm", dest="map_image", type=Path)
    parser.add_argument("--input-type", choices=("simulation", "image", "video", "camera"))
    parser.add_argument("--input-uri")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--speed-mps", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--stop-duration-sec", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--camera-x", type=float)
    parser.add_argument("--camera-y", type=float)
    parser.add_argument("--camera-z", type=float)
    parser.add_argument("--camera-yaw-deg", type=float)
    parser.add_argument("--camera-pitch-up-deg", type=float)
    parser.add_argument("--camera-roll-deg", type=float)
    parser.add_argument("--near-clip-m", type=float)
    parser.add_argument("--path-timeout-sec", type=float)
    parsed = parser.parse_args(args)
    values, config_base = _load_config(parsed.config)
    cli_base = Path.cwd()

    def choose(cli_value, config_value, default=None):
        return cli_value if cli_value is not None else (
            config_value if config_value is not None else default
        )

    waypoint_raw = choose(parsed.waypoints, values.get("waypoint_yaml"))
    map_raw = choose(parsed.map_yaml, values.get("map_yaml"))
    map_image_raw = choose(parsed.map_image, values.get("map_image"))
    output_dir_raw = choose(parsed.output_dir, _nested(values, "output", "directory"))
    output_video_raw = choose(parsed.output_video, _nested(values, "output", "video"))
    input_type = str(choose(parsed.input_type, _nested(values, "input", "type"), "simulation"))
    input_uri_raw = choose(parsed.input_uri, _nested(values, "input", "uri"), "")

    def path_base(cli_value):
        return cli_base if cli_value is not None else config_base

    waypoint_yaml = _path_value(waypoint_raw, path_base(parsed.waypoints), "waypoint_yaml", True)
    map_yaml = _path_value(map_raw, path_base(parsed.map_yaml), "map_yaml", True)
    map_image = _path_value(map_image_raw, path_base(parsed.map_image), "map_image")
    output_dir = _path_value(output_dir_raw, path_base(parsed.output_dir), "output.directory")
    output_video = _path_value(output_video_raw, path_base(parsed.output_video), "output.video")
    if output_dir is None and output_video is None:
        output_dir = (cli_base / "outputs").resolve()
    elif output_dir is None:
        output_dir = output_video.parent
    if output_video is None:
        output_video = output_dir / "full_route_ar.mp4"

    input_uri = str(input_uri_raw or "")
    if input_uri and input_type in {"image", "video"}:
        base = cli_base if parsed.input_uri is not None else config_base
        input_uri = str(_path_value(input_uri, base, "input.uri", True))

    camera = lambda name, cli, default: choose(cli, _nested(values, "camera", name), default)
    motion = lambda name, cli, default: choose(cli, _nested(values, "motion", name), default)
    settings = OfflineSettings(
        waypoint_yaml=waypoint_yaml,
        map_yaml=map_yaml,
        map_image_override=map_image,
        output_dir=output_dir,
        output_video=output_video,
        image_input_type=input_type,
        image_input_uri=input_uri,
        speed_mps=float(motion("speed_mps", parsed.speed_mps, 1.2)),
        fps=float(motion("fps", parsed.fps, 20.0)),
        stop_duration_sec=float(motion("stop_duration_sec", parsed.stop_duration_sec, 5.0)),
        intrinsics=CameraIntrinsics(
            width=int(camera("width", parsed.width, 640)),
            height=int(camera("height", parsed.height, 480)),
            fx=float(camera("fx", parsed.fx, 525.0)),
            fy=float(camera("fy", parsed.fy, 525.0)),
            cx=float(camera("cx", parsed.cx, 319.5)),
            cy=float(camera("cy", parsed.cy, 239.5)),
        ),
        mount=CameraMount(
            x=float(camera("x", parsed.camera_x, 0.20)),
            y=float(camera("y", parsed.camera_y, 0.0)),
            z=float(camera("z", parsed.camera_z, 0.6967)),
            yaw_deg=float(camera("yaw_deg", parsed.camera_yaw_deg, 0.0)),
            pitch_up_deg=float(camera("pitch_up_deg", parsed.camera_pitch_up_deg, 7.5)),
            roll_deg=float(camera("roll_deg", parsed.camera_roll_deg, 0.0)),
        ),
        near_clip_m=float(camera("near_clip_m", parsed.near_clip_m, 0.10)),
        path_timeout_sec=float(motion("path_timeout_sec", parsed.path_timeout_sec, 6.5)),
    )
    settings.validate()
    return settings


def main(args=None) -> None:
    try:
        settings = parse_settings(args)
        report = run_simulation(settings)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"offline simulation error: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
