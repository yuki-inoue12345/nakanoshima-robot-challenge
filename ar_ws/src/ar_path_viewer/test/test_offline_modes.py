from dataclasses import replace
import json
from pathlib import Path

import cv2

import ar_path_viewer.simulation as simulation
import ar_path_viewer.viewer_logic as viewer_logic


PACKAGE = Path(__file__).parents[1]


def test_config_file_switches_route_input_without_source_change():
    route_a = simulation.parse_settings(["--config", str(PACKAGE / "config/offline_route_a.yaml")])
    route_b = simulation.parse_settings(["--config", str(PACKAGE / "config/offline_route_b.yaml")])
    assert route_a.waypoint_yaml.name == "1014-15_True.yaml"
    assert route_b.waypoint_yaml.name == "test_waypoints_b.yaml"
    assert route_a.map_yaml != route_b.map_yaml
    assert route_a.intrinsics.width == 640
    assert route_b.intrinsics.width == 480
    assert route_b.image_input_type == "image"


def test_cli_overrides_speed_fps_camera_and_output(tmp_path):
    settings = simulation.parse_settings([
        "--waypoints", str(PACKAGE / "test_data/test_waypoints_b.yaml"),
        "--map-yaml", str(PACKAGE / "test_data/test_map_b.yaml"),
        "--output-video", str(tmp_path / "chosen.mp4"),
        "--speed-mps", "0.55", "--fps", "9", "--fx", "333",
        "--camera-x", "0.42", "--input-type", "simulation",
    ])
    assert settings.speed_mps == 0.55
    assert settings.fps == 9.0
    assert settings.intrinsics.fx == 333.0
    assert settings.mount.x == 0.42
    assert settings.output_video == tmp_path / "chosen.mp4"


def test_live_and_offline_import_the_same_projection_engine():
    assert simulation.project_clipped_segments is viewer_logic.project_clipped_segments


def test_route_b_end_to_end_video_generation(tmp_path):
    original = simulation.parse_settings([
        "--config", str(PACKAGE / "config/offline_route_b.yaml")
    ])
    settings = replace(
        original,
        output_dir=tmp_path,
        output_video=tmp_path / "route_b_test.mp4",
    )
    report = simulation.run_simulation(settings)
    video = tmp_path / "route_b_test.mp4"
    assert video.is_file() and video.stat().st_size > 0
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert decoded == report["video_frame_count"]
    assert report["waypoint_count"] == 5
    assert report["map"]["resolution"] == 0.5
    assert report["simulated_dwell_sec"] > 0.0
    assert report["exceptions"] == []
    saved = json.loads((tmp_path / "full_route_simulation.json").read_text())
    assert saved["camera"]["input_type"] == "image"
