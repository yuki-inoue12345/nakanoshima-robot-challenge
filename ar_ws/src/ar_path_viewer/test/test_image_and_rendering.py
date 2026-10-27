from pathlib import Path

import cv2
import numpy as np
import pytest

from ar_path_viewer.image_source import SimulationSource, StaticImageSource, make_image_source
from ar_path_viewer.rendering import VALID_STATES, render_frame


def test_simulation_background_dimensions_and_copy():
    source = SimulationSource(320, 240)
    first = source.read()
    second = source.read()
    assert first.shape == (240, 320, 3)
    first[:] = 0
    assert np.any(second)


def test_static_image_input(tmp_path):
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), np.full((10, 20, 3), 127, np.uint8))
    source = StaticImageSource(str(path), 40, 30)
    assert source.read().shape == (30, 40, 3)


def test_video_file_input(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), (10, 20, 30), np.uint8))
    writer.write(np.full((24, 32, 3), (40, 50, 60), np.uint8))
    writer.release()
    source = make_image_source("video", str(path), 32, 24)
    try:
        assert source.read().shape == (24, 32, 3)
        assert source.read().shape == (24, 32, 3)
        # File input loops rather than becoming a false live-camera failure.
        assert source.read().shape == (24, 32, 3)
    finally:
        source.close()


def test_camera_disconnect_or_missing_device_is_reported():
    with pytest.raises(OSError):
        make_image_source("camera", "/dev/ar_path_viewer-device-does-not-exist", 64, 48)


def test_all_required_states_render_and_only_active_draws_route():
    image = np.zeros((100, 160, 3), np.uint8)
    segment = [((20, 80), (140, 50))]
    for state in VALID_STATES:
        frame = render_frame(image, segment, state, thickness=3)
        assert frame.shape == image.shape
        # Route midpoint is away from the status label.
        midpoint = frame[65, 80]
        if state == "ACTIVE":
            assert midpoint[1] > 100
        else:
            assert np.array_equal(midpoint, (0, 0, 0))
