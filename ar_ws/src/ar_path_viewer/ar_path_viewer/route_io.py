"""Generic map and waypoint readers used only by offline simulation.

The waypoint fields mirror the read-only reference implementation in
robotbase_waypoint/waypoint_io.py. Unknown extra fields are retained so a
new harmless attribute does not break AR generation. Active ``change_map``
and ``rotate`` remain explicit errors because one-map straight interpolation
cannot reproduce those operations safely.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class WaypointRecord:
    number: int
    x: float
    y: float
    angle_radians: float
    threshold: float = -1.0
    stop: bool = False
    wait_time: float = 0.0
    person_area: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def report_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "x": self.x,
            "y": self.y,
            "angle_radians": self.angle_radians,
            "threshold": self.threshold,
            "stop": self.stop,
            "wait_time": self.wait_time,
            "person_area": self.person_area,
            "extra_attribute_names": sorted(self.extras),
        }


@dataclass(frozen=True)
class MapMetadata:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin: Tuple[float, float, float]
    negate: int = 0
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    mode: str = "trinary"


_KNOWN_WAYPOINT_FIELDS = {
    "number", "x", "y", "angle_radians", "threshold", "stop",
    "wait_time", "person_area", "change_map", "rotate",
}


def _finite_number(raw: Dict[str, Any], key: str, index: int, default=None) -> float:
    value = raw.get(key, default)
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a numeric waypoint value")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("must be finite")
        return converted
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"waypoints[{index}].{key}: {exc}") from exc


def load_waypoint_records(path: Path) -> List[WaypointRecord]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Waypoint file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("waypoints"), list):
        raise ValueError("Waypoint YAML must contain a 'waypoints' list")
    if not document["waypoints"]:
        raise ValueError("Waypoint list is empty")

    records: List[WaypointRecord] = []
    for index, raw in enumerate(document["waypoints"]):
        if not isinstance(raw, dict):
            raise ValueError(f"waypoints[{index}] must be a mapping")
        missing = [key for key in ("x", "y", "angle_radians") if key not in raw]
        if missing:
            raise ValueError(f"waypoints[{index}] is missing: {', '.join(missing)}")
        unsupported = [
            key for key in ("change_map", "rotate")
            if raw.get(key) not in (None, "", 0, 0.0, False)
        ]
        if unsupported:
            raise ValueError(
                f"waypoints[{index}] uses unsupported fields: {', '.join(unsupported)}"
            )
        for name in ("stop", "person_area"):
            if not isinstance(raw.get(name, False), bool):
                raise ValueError(f"waypoints[{index}].{name} must be true or false")
        number_raw = raw.get("number", index + 1)
        try:
            if isinstance(number_raw, bool) or (
                isinstance(number_raw, float)
                and (not math.isfinite(number_raw) or not number_raw.is_integer())
            ):
                raise ValueError("must be an integer")
            number = int(number_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"waypoints[{index}].number: {exc}") from exc
        wait_time = _finite_number(raw, "wait_time", index, 0.0)
        if wait_time < 0.0:
            raise ValueError(f"waypoints[{index}].wait_time must be non-negative")
        records.append(
            WaypointRecord(
                number=number,
                x=_finite_number(raw, "x", index),
                y=_finite_number(raw, "y", index),
                angle_radians=_finite_number(raw, "angle_radians", index),
                threshold=_finite_number(raw, "threshold", index, -1.0),
                stop=raw.get("stop", False),
                wait_time=wait_time,
                person_area=raw.get("person_area", False),
                extras={key: value for key, value in raw.items() if key not in _KNOWN_WAYPOINT_FIELDS},
            )
        )
    return records


def waypoint_coordinates(records: Sequence[WaypointRecord]) -> np.ndarray:
    return np.asarray([(record.x, record.y) for record in records], dtype=np.float64)


def load_map_metadata(path: Path, image_override: Path | None = None) -> MapMetadata:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Map YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("Map YAML must contain a mapping")
    if image_override is None:
        image_value = document.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError("Map YAML must contain a non-empty 'image' path")
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path
    else:
        image_path = Path(image_override).expanduser()
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Map image not found: {image_path}")
    try:
        resolution = float(document["resolution"])
        origin_values = tuple(float(value) for value in document["origin"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid map resolution/origin: {exc}") from exc
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError("Map resolution must be finite and positive")
    if len(origin_values) < 3 or not np.isfinite(origin_values[:3]).all():
        raise ValueError("Map origin must contain three finite values")
    return MapMetadata(
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=resolution,
        origin=(origin_values[0], origin_values[1], origin_values[2]),
        negate=int(document.get("negate", 0)),
        occupied_thresh=float(document.get("occupied_thresh", 0.65)),
        free_thresh=float(document.get("free_thresh", 0.196)),
        mode=str(document.get("mode", "trinary")),
    )


def read_map_image(metadata: MapMetadata) -> np.ndarray:
    image = cv2.imread(str(metadata.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot decode map image: {metadata.image_path}")
    return image


def world_to_map_pixels(
    points_xy: np.ndarray,
    metadata: MapMetadata,
    image_shape: Sequence[int],
) -> np.ndarray:
    """Convert world coordinates to top-left-origin image pixels, including map yaw."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points_xy must be a finite Nx2 array")
    height, width = int(image_shape[0]), int(image_shape[1])
    ox, oy, yaw = metadata.origin
    c, s = math.cos(yaw), math.sin(yaw)
    relative = points - np.array((ox, oy))
    # World -> map axes is R(-origin_yaw).
    map_x = (c * relative[:, 0] + s * relative[:, 1]) / metadata.resolution
    map_y = (-s * relative[:, 0] + c * relative[:, 1]) / metadata.resolution
    return np.column_stack((map_x, (height - 1.0) - map_y))


def validate_waypoints_on_map(
    records: Sequence[WaypointRecord], metadata: MapMetadata, image_shape: Sequence[int]
) -> np.ndarray:
    points = waypoint_coordinates(records)
    pixels = world_to_map_pixels(points, metadata, image_shape)
    height, width = int(image_shape[0]), int(image_shape[1])
    outside = np.flatnonzero(
        (pixels[:, 0] < 0.0) | (pixels[:, 0] > width - 1.0)
        | (pixels[:, 1] < 0.0) | (pixels[:, 1] > height - 1.0)
    )
    if len(outside):
        labels = ", ".join(str(records[index].number) for index in outside)
        raise ValueError(f"Waypoint(s) outside map image bounds: {labels}")
    return pixels
