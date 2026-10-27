from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from ar_path_viewer.route_io import (
    MapMetadata,
    WaypointRecord,
    load_map_metadata,
    load_waypoint_records,
    read_map_image,
    validate_waypoints_on_map,
    world_to_map_pixels,
)


DATA = Path(__file__).parents[1] / "test_data"


def test_custom_waypoint_fields_and_unknown_attributes_are_safe():
    records = load_waypoint_records(DATA / "test_waypoints_b.yaml")
    assert len(records) == 5
    assert records[1].number == 205
    assert records[1].stop is True
    assert records[1].wait_time == pytest.approx(0.4)
    assert records[1].extras == {"test_note": "harmless-extra-attribute"}
    assert records[3].person_area is True
    assert records[2].threshold == pytest.approx(0.15)


def test_map_image_relative_path_resolution_and_origin_conversion():
    metadata = load_map_metadata(DATA / "test_map_b.yaml")
    assert metadata.image_path == (DATA / "test_map_b.pgm").resolve()
    assert metadata.resolution == pytest.approx(0.5)
    assert metadata.origin == (-2.0, -3.0, 0.0)
    image = read_map_image(metadata)
    pixels = world_to_map_pixels(np.array(((0.0, 0.0), (3.0, 2.0))), metadata, image.shape)
    assert np.allclose(pixels, ((4.0, 13.0), (10.0, 9.0)))


def test_nonzero_map_origin_yaw_is_applied():
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"), image_path=Path("map.pgm"),
        resolution=1.0, origin=(10.0, 20.0, math.pi / 2),
    )
    # World +y is map +x for a map whose x-axis is yawed +90 degrees.
    pixels = world_to_map_pixels(np.array(((10.0, 22.0),)), metadata, (10, 10))
    assert np.allclose(pixels, ((2.0, 9.0)), atol=1e-9)


def test_waypoint_outside_map_is_clear_error():
    metadata = load_map_metadata(DATA / "test_map_b.yaml")
    image = read_map_image(metadata)
    outside = [WaypointRecord(9, 100.0, 100.0, 0.0)]
    with pytest.raises(ValueError, match="outside map image bounds: 9"):
        validate_waypoints_on_map(outside, metadata, image.shape)


def test_invalid_map_resolution_is_clear_error(tmp_path):
    path = tmp_path / "bad_map.yaml"
    path.write_text(
        f"image: {DATA / 'test_map_b.pgm'}\nresolution: 0\norigin: [0, 0, 0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resolution must be finite and positive"):
        load_map_metadata(path)


@pytest.mark.parametrize(
    "body, message",
    (
        ("waypoints: [{x: 0, y: 0}]\n", "angle_radians"),
        ("waypoints: [{x: .nan, y: 0, angle_radians: 0}]\n", "must be finite"),
        ('waypoints: [{x: 0, y: 0, angle_radians: 0, stop: "yes"}]\n', "stop must be true or false"),
        ("waypoints: [{x: 0, y: 0, angle_radians: 0, change_map: next}]\n", "unsupported fields"),
    ),
)
def test_invalid_waypoint_documents_have_clear_errors(tmp_path, body, message):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_waypoint_records(path)
