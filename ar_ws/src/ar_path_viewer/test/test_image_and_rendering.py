from pathlib import Path

import cv2
import numpy as np
import pytest

from ar_path_viewer.image_source import SimulationSource, StaticImageSource, make_image_source
from ar_path_viewer.display_window import (
    DisplayWindow,
    fit_frame_to_window,
    is_fullscreen_key,
)
from ar_path_viewer.rendering import (
    VALID_STATES,
    direction_arrow_segments,
    render_frame,
)


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


def test_direction_arrow_follows_projected_path_direction():
    forward = direction_arrow_segments([((10, 50), (130, 50))])
    reverse = direction_arrow_segments([((130, 50), (10, 50))])
    assert forward and reverse
    assert forward[0][0][0] < forward[0][1][0]
    assert reverse[0][0][0] > reverse[0][1][0]


def test_display_letterboxes_without_changing_source_frame():
    frame = np.full((480, 640, 3), 127, np.uint8)
    original = frame.copy()
    fitted = fit_frame_to_window(frame, 1000, 600)
    assert fitted.shape == (600, 1000, 3)
    assert np.array_equal(frame, original)
    assert np.all(fitted[:, :100] == 0)
    assert np.all(fitted[:, 900:] == 0)
    assert np.all(fitted[:, 100:900] == 127)


class FakeHighGui:
    WINDOW_NORMAL = 0
    WINDOW_KEEPRATIO = 0
    WINDOW_FULLSCREEN = 1
    WND_PROP_FULLSCREEN = 0

    def __init__(self, keys):
        self.keys = iter(keys)
        self.named = []
        self.resized = []
        self.properties = []
        self.shown = []
        self.destroyed = []

    def namedWindow(self, name, flags):
        self.named.append((name, flags))

    def resizeWindow(self, name, width, height):
        self.resized.append((name, width, height))

    def getWindowImageRect(self, name):
        return (0, 0, 1000, 600)

    def imshow(self, name, frame):
        self.shown.append((name, frame.copy()))

    def waitKeyEx(self, delay):
        return next(self.keys)

    def waitKey(self, delay):
        return next(self.keys)

    def setWindowProperty(self, name, prop, value):
        self.properties.append((name, prop, value))

    def destroyWindow(self, name):
        self.destroyed.append(name)


def test_resizable_window_f11_toggle_and_q_passthrough():
    gui = FakeHighGui((0x0100003A, 0x0100003A, ord("q")))
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.full((480, 640, 3), 80, np.uint8)
    assert window.show_and_poll(frame) == -1
    assert window.fullscreen
    assert window.show_and_poll(frame) == -1
    assert not window.fullscreen
    assert window.show_and_poll(frame) == ord("q")
    assert gui.shown[-1][1].shape == (600, 1000, 3)
    assert gui.resized[0][1:] == (960, 720)
    assert gui.resized[-1][1:] == (960, 720)
    window.close()
    window.close()
    assert len(gui.destroyed) == 1


def test_f11_backend_codes_are_recognized():
    for key_code in (0x007A0000, 0x0000FFC8, 0x0100003A):
        assert is_fullscreen_key(key_code)
    assert not is_fullscreen_key(ord("q"))


def test_fullscreen_api_failure_keeps_normal_window_running():
    class NoFullscreenHighGui(FakeHighGui):
        def setWindowProperty(self, name, prop, value):
            raise RuntimeError("fullscreen unsupported")

    gui = NoFullscreenHighGui((0x0000FFC8, ord("q")))
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.zeros((480, 640, 3), np.uint8)
    assert window.show_and_poll(frame) == -1
    assert not window.fullscreen
    assert window.show_and_poll(frame) == ord("q")
