import threading

import cv2
import numpy as np
import pytest

from ar_path_viewer import display_window
from ar_path_viewer.costmap_alerts import CostmapAlert, CostmapAlertHistory
from ar_path_viewer.costmap_overlay import ProjectedCostmap
from ar_path_viewer.costmap_minimap import CostmapMinimap
from ar_path_viewer.costmap_minimap import robot_view_tab_rectangle
from ar_path_viewer.image_source import SimulationSource, StaticImageSource, make_image_source
from ar_path_viewer.localization_monitor import LocalizationWarning
from ar_path_viewer.offscreen_guidance import OffscreenGuidance
from ar_path_viewer.paced_video_writer import PacedVideoWriter
from ar_path_viewer.display_window import (
    ACTION_COPY_LOG,
    ACTION_TOGGLE_ROBOT_VIEW,
    ACTION_VIEW_LOG,
    CONTROL_BAR_HEIGHT,
    DisplayWindow,
    compose_display_frame,
    fit_frame_to_window,
    is_fullscreen_key,
)
from ar_path_viewer.rendering import (
    VALID_STATES,
    dashed_line_segments,
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


def test_paced_video_writer_preserves_elapsed_duration_with_real_frames():
    class Writer:
        def __init__(self):
            self.frames = []
            self.released = False

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    now = [0.0]
    output = Writer()
    writer = PacedVideoWriter(output, 20.0, clock=lambda: now[0])
    writer.write(np.full((2, 2, 3), 10, np.uint8))
    # A faster callback is retained but does not create an extra 20 fps slot.
    now[0] = 0.02
    writer.write(np.full((2, 2, 3), 20, np.uint8))
    # At 0.11 s the file needs three frames. Fill one missed slot with the
    # last actually rendered frame, then write the current rendering.
    now[0] = 0.11
    writer.write(np.full((2, 2, 3), 30, np.uint8))
    writer.release()

    assert [int(frame[0, 0, 0]) for frame in output.frames] == [10, 20, 30]
    assert writer.frames_written == 3
    assert output.released


def test_paced_video_writer_rejects_invalid_fps():
    with pytest.raises(ValueError, match="FPS"):
        PacedVideoWriter(object(), 0.0)


def test_thirty_fps_writer_counts_real_and_duplicate_output_separately():
    class Writer:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            pass

    now = [0.0]
    output = Writer()
    writer = PacedVideoWriter(output, 30.0, clock=lambda: now[0])
    writer.write(np.full((2, 2, 3), 10, np.uint8))
    now[0] = 0.1
    writer.write(np.full((2, 2, 3), 20, np.uint8))

    assert writer.frames_written == 4
    assert writer.real_frames_written == 2
    assert writer.duplicate_frames_written == 2
    assert [int(frame[0, 0, 0]) for frame in output.frames] == [10, 10, 10, 20]


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



def test_frame_sync_wait_is_supported_and_hides_route():
    assert "FRAME SYNC WAIT" in VALID_STATES
    image = np.zeros((100, 160, 3), np.uint8)
    segment = [((20, 80), (140, 50))]
    frame = render_frame(image, segment, "FRAME SYNC WAIT", thickness=3)
    assert frame.shape == image.shape
    assert np.array_equal(frame[65, 80], (0, 0, 0))


def test_direction_arrow_follows_projected_path_direction():
    forward = direction_arrow_segments([((10, 50), (130, 50))])
    reverse = direction_arrow_segments([((130, 50), (10, 50))])
    assert forward and reverse
    assert forward[0][0][0] < forward[0][1][0]
    assert reverse[0][0][0] > reverse[0][1][0]


def test_reference_path_is_cyan_dashed_and_green_plan_remains_on_top():
    image = np.zeros((180, 320, 3), np.uint8)
    overlap = [((20, 120), (300, 120))]
    reference_only = [((20, 150), (300, 150))]
    frame = render_frame(
        image,
        overlap,
        "ACTIVE",
        thickness=5,
        reference_segments=overlap + reference_only,
        reference_thickness=3,
        reference_dash_px=14.0,
        reference_gap_px=9.0,
    )

    # The solid control path is painted after the reference path and remains green.
    assert frame[120, 160, 1] > 180
    assert frame[120, 160, 0] < 100
    cyan = (
        (frame[145:156, :, 0] > 150)
        & (frame[145:156, :, 1] > 150)
        & (frame[145:156, :, 2] < 100)
    )
    assert np.count_nonzero(cyan) > 50
    # Cyan gaps remain visible even though the dark readability halo may span
    # part of a short gap on an otherwise black image.
    cyan_row = (
        (frame[150, 20:301, 0] > 150)
        & (frame[150, 20:301, 1] > 150)
        & (frame[150, 20:301, 2] < 100)
    )
    assert np.count_nonzero(~cyan_row) > 20


def test_display_mode_label_is_rendered_without_changing_route_geometry():
    image = np.zeros((180, 640, 3), np.uint8)
    route = [((20, 120), (300, 120))]
    without_label = render_frame(image, route, "ACTIVE")
    with_label = render_frame(
        image, route, "ACTIVE", display_mode_label="AR FLOAT 0.50m"
    )
    assert np.array_equal(without_label[100:140], with_label[100:140])
    assert not np.array_equal(
        with_label[12:38, 430:630], without_label[12:38, 430:630]
    )


def test_dashed_reference_preserves_real_segment_endpoints_and_empty_is_noop():
    dashes = dashed_line_segments([((5, 40), (95, 40))], 10.0, 5.0)
    assert dashes
    assert dashes[0][0] == (5, 40)
    assert all(5 <= point[0] <= 95 for dash in dashes for point in dash)
    assert dashed_line_segments([]) == []


def test_offscreen_guidance_is_orange_ui_not_green_projected_path():
    image = np.zeros((480, 640, 3), np.uint8)
    guidance = OffscreenGuidance("RIGHT", -35.0, 2.0, 1.8)
    frame = render_frame(
        image, [], "PATH OUT OF VIEW", guidance=guidance
    )
    orange = (
        (frame[:, :, 2] > 180)
        & (frame[:, :, 1] > 80)
        & (frame[:, :, 0] < 80)
    )
    green = (
        (frame[:, :, 1] > 180)
        & (frame[:, :, 2] < 80)
        & (frame[:, :, 0] < 80)
    )
    assert np.count_nonzero(orange) > 100
    assert np.count_nonzero(green) == 0

    no_path = render_frame(image, [], "NO PATH", guidance=guidance)
    no_path_orange = (
        (no_path[:, :, 2] > 180)
        & (no_path[:, :, 1] > 80)
        & (no_path[:, :, 0] < 80)
    )
    assert np.count_nonzero(no_path_orange[120:]) == 0


def test_active_route_keeps_green_line_and_adds_direction_and_distance_badges():
    image = np.zeros((480, 640, 3), np.uint8)
    route = [((100, 350), (540, 240))]
    baseline = render_frame(image, route, "ACTIVE")
    frame = render_frame(
        image,
        route,
        "ACTIVE",
        guidance=OffscreenGuidance("RIGHT", -22.0, 2.4, 2.0),
        waypoint_status="PLAN LEFT: 2.4 m",
    )

    # Existing projected line remains green and unchanged away from the badges.
    assert np.array_equal(frame[292:304, 300:340], baseline[292:304, 300:340])
    assert np.count_nonzero(frame[:, :, 1] > 180) > 100
    # The distance remains at the bottom; direction is moved above the floor.
    assert not np.array_equal(frame[430:475, :310], baseline[430:475, :310])
    assert not np.array_equal(frame[48:130, 400:], baseline[48:130, 400:])


def test_normal_overlay_hides_technical_detail_but_debug_mode_keeps_it():
    image = np.zeros((480, 640, 3), np.uint8)
    detail = "gen=42 frame=map pts=110 len=2.80m seg=75 px=154"
    normal = render_frame(
        image, [], "PATH OUT OF VIEW", detail=detail, debug_overlay=False
    )
    debug = render_frame(
        image, [], "PATH OUT OF VIEW", detail=detail, debug_overlay=True
    )
    assert np.array_equal(normal[43:86], image[43:86])
    assert not np.array_equal(debug[43:86], image[43:86])


def test_active_direction_panel_is_prominent_and_keeps_green_path():
    image = np.zeros((480, 640, 3), np.uint8)
    route = [((80, 330), (450, 250))]
    frame = render_frame(
        image,
        route,
        "ACTIVE",
        guidance=OffscreenGuidance("LEFT", 32.0, 2.4, 2.0),
        debug_overlay=False,
    )
    orange = (
        (frame[48:130, :, 2] > 180)
        & (frame[48:130, :, 1] > 80)
        & (frame[48:130, :, 0] < 80)
    )
    green = (
        (frame[:, :, 1] > 180)
        & (frame[:, :, 2] < 80)
        & (frame[:, :, 0] < 80)
    )
    assert np.count_nonzero(orange) > 250
    assert np.count_nonzero(green) > 100
    # The cue stays above the floor route rather than covering the lower view.
    assert np.array_equal(frame[390:470, 430:630], image[390:470, 430:630])


@pytest.mark.parametrize('direction', ('LEFT', 'STRAIGHT', 'RIGHT'))
def test_turn_shaped_direction_card_is_clear_for_every_safe_decision(direction):
    image = np.zeros((480, 640, 3), np.uint8)
    frame = render_frame(
        image,
        [],
        'PATH OUT OF VIEW',
        guidance=OffscreenGuidance(direction, 0.0, 2.4, 2.0),
        debug_overlay=False,
    )
    orange = (
        (frame[48:130, :, 2] > 180)
        & (frame[48:130, :, 1] > 80)
        & (frame[48:130, :, 0] < 80)
    )
    assert np.count_nonzero(orange) > 250
    assert np.count_nonzero(frame[150:390]) == 0


def test_reference_dashes_have_dark_halo_without_hiding_green_route():
    image = np.full((180, 320, 3), 245, np.uint8)
    reference = [((20, 140), (300, 140))]
    overlap = [((20, 110), (300, 110))]
    frame = render_frame(
        image,
        overlap,
        'ACTIVE',
        reference_segments=reference + overlap,
        reference_thickness=3,
    )
    # Cyan remains readable on a bright background and has a dark edge.
    assert np.any(np.all(frame[140] == (255, 255, 0), axis=1))
    assert np.any(np.all(frame[136:145] == (18, 18, 18), axis=2))
    # Green real Path is still rendered after the reference layer.
    assert frame[110, 160, 1] > 180
    assert frame[110, 160, 0] < 100


def test_waypoint_badge_is_suppressed_by_non_path_viewer_states():
    image = np.zeros((480, 640, 3), np.uint8)
    label = "PLAN LEFT: 82 cm"
    no_path = render_frame(image, [], "NO PATH", waypoint_status=label)
    baseline = render_frame(image, [], "NO PATH")
    assert np.array_equal(no_path, baseline)


def test_green_path_alpha_keeps_camera_context_visible():
    image = np.full((180, 320, 3), (60, 80, 100), np.uint8)
    route = [((20, 120), (300, 120))]
    opaque = render_frame(image, route, "ACTIVE", path_alpha=1.0)
    translucent = render_frame(image, route, "ACTIVE", path_alpha=0.5)

    assert opaque[120, 160, 1] > translucent[120, 160, 1]
    assert translucent[120, 160, 0] > opaque[120, 160, 0]
    assert np.array_equal(translucent[100, 160], image[100, 160])
    with pytest.raises(ValueError):
        render_frame(image, route, "ACTIVE", path_alpha=0.0)


def test_costmap_alert_and_robot_view_label_class_and_metric_evidence():
    image = np.full((480, 640, 3), 70, np.uint8)
    minimap = CostmapMinimap(
        points_in_base=np.asarray(((1.0, 0.0, 0.0),), dtype=float),
        costs=np.asarray((100,), dtype=np.int16),
        path_points_base=np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=float
        ),
        high_cost_min=60,
        lethal_cost_min=100,
    )
    frame = render_frame(
        image,
        [],
        "ACTIVE",
        costmap_alert=CostmapAlert(
            "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.234"
        ),
        costmap_minimap=minimap,
        robot_view_enabled=True,
        robot_view_expanded=True,
    )
    no_metric_detail = render_frame(
        image,
        [],
        "ACTIVE",
        costmap_alert=CostmapAlert(
            "LETHAL COST NEAR PATH", "lethal_path", ""
        ),
        costmap_minimap=minimap,
        robot_view_enabled=True,
        robot_view_expanded=True,
    )

    # Metric evidence remains a distinct line inside the prominent alert.
    assert np.any(frame[178:197, 20:288] != no_metric_detail[178:197, 20:288])
    # The expanded ROBOT VIEW legend contains distinct high/lethal classes
    # without resizing or reprojecting the camera.
    high = (frame[:, :, 0] == 20) & (frame[:, :, 1] == 190) & (frame[:, :, 2] == 230)
    lethal = (frame[:, :, 0] == 205) & (frame[:, :, 1] == 45) & (frame[:, :, 2] == 235)
    assert np.count_nonzero(high[330:355, 440:]) > 0
    assert np.count_nonzero(lethal[330:355, 440:]) > 0
    assert np.array_equal(frame[280:310, :410], image[280:310, :410])


def test_costmap_sync_hold_is_dimmed_labelled_and_does_not_create_alert():
    image = np.full((480, 640, 3), 80, np.uint8)
    projection = ProjectedCostmap(
        unknown_pixels=np.empty((0, 2), dtype=np.int32),
        high_cost_pixels=np.asarray(((320, 300),), dtype=np.int32),
        lethal_cost_pixels=np.asarray(((380, 300),), dtype=np.int32),
        input_points=2,
        visible_points=2,
    )
    live = render_frame(
        image, [], "ACTIVE", costmap_projection=projection,
        costmap_alpha=0.8, costmap_radius_px=2,
    )
    held = render_frame(
        image, [], "ACTIVE", costmap_projection=projection,
        costmap_alpha=0.8, costmap_radius_px=2,
        costmap_sync_hold_age_sec=0.18,
    )
    baseline = render_frame(image, [], "ACTIVE")

    # Held pixels are visibly weaker than a live synchronized projection.
    assert np.linalg.norm(
        held[300, 320].astype(float) - image[300, 320].astype(float)
    ) < np.linalg.norm(
        live[300, 320].astype(float) - image[300, 320].astype(float)
    )
    # The top status differs because the held layer is explicitly labelled.
    assert not np.array_equal(held[8:42], live[8:42])
    # No warning card is synthesized from a visual-only hold.
    assert np.array_equal(held[109:147, 10:380], baseline[109:147, 10:380])


def test_robot_view_is_collapsed_by_default_and_expands_without_reprojection():
    image = np.full((480, 640, 3), (45, 65, 85), np.uint8)
    route = [((80, 390), (560, 280))]
    minimap = CostmapMinimap(
        np.asarray(((0.8, -0.3, 0.0), (1.5, 0.2, 0.0))),
        np.asarray((70, 100), dtype=np.int16),
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        60, 100,
    )
    collapsed = render_frame(
        image, route, "ACTIVE", costmap_minimap=minimap,
        robot_view_enabled=True, robot_view_expanded=False,
    )
    expanded = render_frame(
        image, route, "ACTIVE", costmap_minimap=minimap,
        robot_view_enabled=True, robot_view_expanded=True,
    )
    tab_left, tab_top, _, tab_bottom = robot_view_tab_rectangle(640, 480)
    assert np.any(collapsed[tab_top:tab_bottom + 1, tab_left:] != image[
        tab_top:tab_bottom + 1, tab_left:
    ])
    # Closed mode leaves the former right-column minimap area as live camera.
    assert np.array_equal(collapsed[140:210, 450:600], image[140:210, 450:600])
    assert np.any(expanded[140:350, 425:615] != image[140:350, 425:615])
    # The panel overlays rather than rescaling; the left-side real Path is exact.
    assert np.array_equal(collapsed[360:400, 70:220], expanded[360:400, 70:220])


@pytest.mark.parametrize(
    "alert,color_channel",
    (
        (CostmapAlert("HIGH COST NEAR PATH", "high_path", "path_distance_m=0.2"), 1),
        (CostmapAlert("LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"), 2),
    ),
)
def test_large_costmap_warning_is_prominent_and_clears(alert, color_channel):
    image = np.full((480, 640, 3), 55, np.uint8)
    warned = render_frame(image, [], "ACTIVE", costmap_alert=alert)
    cleared = render_frame(image, [], "ACTIVE", costmap_alert=None)
    region = warned[109:202, 10:411]
    assert np.count_nonzero(region[:, :, color_channel] > 120) > 1000
    assert not np.array_equal(region, cleared[109:202, 10:411])
    assert np.array_equal(cleared[109:202, 10:411], image[109:202, 10:411])


def test_large_warning_has_ascii_fallback_without_optional_japanese_font(monkeypatch):
    monkeypatch.setattr(display_window, "_japanese_font", lambda: None)
    image = np.zeros((480, 640, 3), np.uint8)
    warned = render_frame(
        image, [], "ACTIVE",
        costmap_alert=CostmapAlert(
            "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"
        ),
    )
    assert np.count_nonzero(warned[112:198, 15:405]) > 1000


def test_recent_alert_history_is_visually_distinct_from_live_warning():
    image = np.full((480, 640, 3), 55, np.uint8)
    alert = CostmapAlert(
        "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"
    )
    live = render_frame(image, [], "ACTIVE", costmap_alert=alert)
    history = render_frame(
        image, [], "ACTIVE",
        costmap_alert_history=CostmapAlertHistory(
            alert, 1.2, "UNCONFIRMED"
        ),
    )
    region = np.s_[109:202, 10:411]
    assert not np.array_equal(history[region], image[region])
    assert not np.array_equal(history[region], live[region])
    # Historical state is muted and must not reproduce the live red alert.
    assert np.count_nonzero(history[region][:, :, 2] > 230) < np.count_nonzero(
        live[region][:, :, 2] > 230
    )


def test_localization_warning_active_recent_and_costmap_can_coexist():
    image = np.full((480, 640, 3), 55, np.uint8)
    costmap_alert = CostmapAlert(
        "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"
    )
    active = render_frame(
        image, [], "LOCALIZATION UNSTABLE",
        costmap_alert=costmap_alert,
        localization_warning=LocalizationWarning(
            "active", "AMCL POSITION JUMP", "distance=2.1m", 0.0
        ),
    )
    recent = render_frame(
        image, [], "ACTIVE",
        localization_warning=LocalizationWarning(
            "recent", "AMCL POSITION JUMP", "distance=2.1m", 4.0
        ),
    )
    costmap_region = np.s_[109:202, 10:411]
    localization_region = np.s_[207:280, 10:411]
    assert not np.array_equal(active[costmap_region], image[costmap_region])
    assert not np.array_equal(active[localization_region], image[localization_region])
    assert not np.array_equal(recent[207:266, 10:411], image[207:266, 10:411])
    # Current and recovered history are deliberately different visual states.
    assert not np.array_equal(
        active[207:266, 10:411], recent[207:266, 10:411]
    )


def test_recorded_composite_keeps_fixed_size_panel_and_warning(tmp_path):
    output = tmp_path / "composite.avi"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (640, 480)
    )
    assert writer.isOpened()
    image = np.full((480, 640, 3), 64, np.uint8)
    minimap = CostmapMinimap(
        np.asarray(((1.0, 0.0, 0.0),)), np.asarray((100,), np.int16),
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))), 60, 100,
    )
    collapsed = render_frame(
        image, [], "ACTIVE", robot_view_enabled=True,
        robot_view_expanded=False,
    )
    expanded = render_frame(
        image, [], "ACTIVE", robot_view_enabled=True,
        robot_view_expanded=True, costmap_minimap=minimap,
        costmap_alert=CostmapAlert(
            "LETHAL COST NEAR PATH", "lethal_path", "path_distance_m=0.1"
        ),
    )
    writer.write(collapsed)
    writer.write(expanded)
    writer.release()
    capture = cv2.VideoCapture(str(output))
    ok1, decoded_collapsed = capture.read()
    ok2, decoded_expanded = capture.read()
    capture.release()
    assert ok1 and ok2
    assert decoded_collapsed.shape == decoded_expanded.shape == (480, 640, 3)
    assert np.mean(cv2.absdiff(decoded_collapsed, decoded_expanded)) > 4.0


def test_replanning_and_navigation_hold_show_status_without_route():
    image = np.zeros((480, 640, 3), np.uint8)
    route = [((100, 350), (540, 240))]
    for state, label in (
        ("REPLANNING", "PATH CHANGE CHECK"),
        ("NAVIGATION HOLD", "PERSON SEARCH"),
    ):
        frame = render_frame(
            image, route, state, navigation_status=label, debug_overlay=False
        )
        assert np.array_equal(frame[292:304, 300:340], image[292:304, 300:340])
        assert not np.array_equal(frame[430:475, 430:], image[430:475, 430:])


def test_display_letterboxes_without_changing_source_frame():
    frame = np.full((480, 640, 3), 127, np.uint8)
    original = frame.copy()
    fitted = fit_frame_to_window(frame, 1000, 600)
    assert fitted.shape == (600, 1000, 3)
    assert np.array_equal(frame, original)
    assert np.all(fitted[:, :100] == 0)
    assert np.all(fitted[:, 900:] == 0)
    assert np.all(fitted[:, 100:900] == 127)


def test_display_control_bar_is_below_video_and_does_not_change_source():
    frame = np.full((480, 640, 3), 127, np.uint8)
    original = frame.copy()
    display, rectangles = compose_display_frame(frame, 1000, 700)
    assert display.shape == (700, 1000, 3)
    assert np.array_equal(frame, original)
    assert set(rectangles) == {
        ACTION_VIEW_LOG, ACTION_COPY_LOG, ACTION_TOGGLE_ROBOT_VIEW
    }
    source_mask = np.all(display[:636] == 127, axis=2)
    ys, xs = np.nonzero(source_mask)
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (848, 636)
    for action in (ACTION_VIEW_LOG, ACTION_COPY_LOG):
        x1, y1, x2, y2 = rectangles[action]
        assert 0 <= x1 < x2 < display.shape[1]
        assert 480 < y1 < y2 < display.shape[0]
    assert np.any(display[650:] != 0)


@pytest.mark.parametrize(
    "window_size",
    ((480, 360), (960, 720), (1280, 1024), (1920, 1080)),
)
def test_display_fills_small_large_and_fullscreen_viewports(window_size):
    width, height = window_size
    color = np.asarray((17, 83, 141), dtype=np.uint8)
    frame = np.empty((480, 640, 3), dtype=np.uint8)
    frame[:] = color
    display, _ = compose_display_frame(frame, width, height)
    video_height = height - 64
    mask = np.all(display[:video_height] == color, axis=2)
    rows, columns = np.nonzero(mask)
    expected_scale = min(width / 640.0, video_height / 480.0)
    expected_width = int(round(640 * expected_scale))
    expected_height = int(round(480 * expected_scale))
    assert columns.max() - columns.min() + 1 == expected_width
    assert rows.max() - rows.min() + 1 == expected_height
    # Exactly one axis must consume the available viewport: there is no
    # additional HighGUI-style scaling or a second letterbox in this canvas.
    assert expected_width == width or expected_height == video_height


@pytest.mark.parametrize("window_size", ((480, 360), (960, 720), (1920, 1080)))
def test_log_button_text_is_centered_and_not_clipped(window_size):
    display, rectangles = compose_display_frame(
        np.zeros((480, 640, 3), np.uint8), *window_size
    )
    for action in (ACTION_VIEW_LOG, ACTION_COPY_LOG):
        x1, y1, x2, y2 = rectangles[action]
        assert y2 - y1 == 52
        # The button fill and border remain below 200; this isolates label ink.
        region = display[y1 + 1:y2, x1 + 1:x2]
        rows, columns = np.nonzero(np.min(region, axis=2) > 200)
        assert rows.size > 0
        text_center_x = x1 + 1 + (columns.min() + columns.max()) / 2.0
        text_center_y = y1 + 1 + (rows.min() + rows.max()) / 2.0
        assert abs(text_center_x - (x1 + x2) / 2.0) <= 2.0
        assert abs(text_center_y - (y1 + y2) / 2.0) <= 1.0
        assert rows.min() >= 1
        assert rows.max() <= region.shape[0] - 2


def test_ascii_fallback_text_is_also_centered(monkeypatch):
    monkeypatch.setattr(display_window, "_japanese_font", lambda: None)
    display, rectangles = compose_display_frame(
        np.zeros((480, 640, 3), np.uint8), 960, 720
    )
    for action in (ACTION_VIEW_LOG, ACTION_COPY_LOG):
        x1, y1, x2, y2 = rectangles[action]
        region = display[y1 + 1:y2, x1 + 1:x2]
        rows, columns = np.nonzero(np.min(region, axis=2) > 200)
        assert rows.size > 0
        text_center_x = x1 + 1 + (columns.min() + columns.max()) / 2.0
        text_center_y = y1 + 1 + (rows.min() + rows.max()) / 2.0
        assert abs(text_center_x - (x1 + x2) / 2.0) <= 2.0
        assert abs(text_center_y - (y1 + y2) / 2.0) <= 1.0


class FakeHighGui:
    WINDOW_NORMAL = 0
    WINDOW_KEEPRATIO = 0
    WINDOW_FREERATIO = 0x100
    WINDOW_GUI_NORMAL = 0x10
    WINDOW_FULLSCREEN = 1
    WND_PROP_FULLSCREEN = 0
    EVENT_LBUTTONUP = 4

    def __init__(self, keys):
        self.keys = iter(keys)
        self.named = []
        self.resized = []
        self.properties = []
        self.shown = []
        self.destroyed = []
        self.mouse_callback = None
        self.image_rect = (0, 0, 1000, 600)

    def namedWindow(self, name, flags):
        self.named.append((name, flags))

    def resizeWindow(self, name, width, height):
        self.resized.append((name, width, height))

    def getWindowImageRect(self, name):
        return self.image_rect

    def imshow(self, name, frame):
        self.shown.append((name, frame.copy()))

    def waitKeyEx(self, delay):
        return next(self.keys)

    def waitKey(self, delay):
        return next(self.keys)

    def setWindowProperty(self, name, prop, value):
        self.properties.append((name, prop, value))

    def setMouseCallback(self, name, callback):
        self.mouse_callback = callback

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
    assert gui.named[0][1] == gui.WINDOW_FREERATIO | gui.WINDOW_GUI_NORMAL
    assert gui.resized[0][1:] == (960, 720)
    assert gui.resized[-1][1:] == (960, 720)
    window.close()
    window.close()
    assert len(gui.destroyed) == 1


def test_highgui_operations_reject_a_different_thread():
    gui = FakeHighGui((-1,))
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.zeros((480, 640, 3), np.uint8)
    failures = []

    def use_from_worker():
        for operation in (
            lambda: window.show_and_poll(frame),
            window.consume_actions,
            lambda: window.set_status("worker", True),
            window.toggle_fullscreen,
            window.close,
        ):
            try:
                operation()
            except RuntimeError as exc:
                failures.append(str(exc))

    worker = threading.Thread(target=use_from_worker)
    worker.start()
    worker.join()

    assert len(failures) == 5
    assert all("owner thread" in message for message in failures)
    assert gui.shown == []
    assert gui.destroyed == []
    window.close()
    assert len(gui.destroyed) == 1


def test_resized_highgui_viewport_drives_entire_display_canvas():
    gui = FakeHighGui((-1, -1))
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.zeros((480, 640, 3), np.uint8)
    window.show_and_poll(frame)
    assert gui.shown[-1][1].shape == (600, 1000, 3)
    gui.image_rect = (40, 50, 1600, 900)
    window.show_and_poll(frame)
    assert gui.shown[-1][1].shape == (900, 1600, 3)


def test_fullscreen_restore_retries_normal_size_for_asynchronous_qt():
    gui = FakeHighGui((-1, -1, -1))
    window = DisplayWindow(960, 720, cv_module=gui)
    assert window.toggle_fullscreen()
    assert window.toggle_fullscreen()
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):
        window.show_and_poll(frame)
    assert gui.resized == [("AR Path Viewer", 960, 720)] * 4


def test_log_buttons_emit_mouse_actions_without_affecting_keys():
    gui = FakeHighGui((-1, ord("q")))
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.zeros((480, 640, 3), np.uint8)
    assert window.show_and_poll(frame) == -1
    assert gui.mouse_callback is not None
    for expected in (ACTION_VIEW_LOG, ACTION_COPY_LOG):
        x1, y1, x2, y2 = window._button_rectangles[expected]
        gui.mouse_callback(gui.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0, None)
        assert window.consume_actions() == [expected]
    assert window.show_and_poll(frame) == ord("q")


@pytest.mark.parametrize("image_rect", ((0, 0, 1000, 600), (0, 0, 1920, 1080)))
def test_robot_view_tab_mouse_hitbox_tracks_resized_and_fullscreen(image_rect):
    gui = FakeHighGui((-1,))
    gui.image_rect = image_rect
    window = DisplayWindow(960, 720, cv_module=gui)
    frame = np.zeros((480, 640, 3), np.uint8)
    window.show_and_poll(frame)
    x1, y1, x2, y2 = window._button_rectangles[ACTION_TOGGLE_ROBOT_VIEW]
    gui.mouse_callback(
        gui.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0, None
    )
    assert window.consume_actions() == [ACTION_TOGGLE_ROBOT_VIEW]
    assert 0 <= x1 < x2 < image_rect[2]
    assert 0 <= y1 < y2 < image_rect[3] - CONTROL_BAR_HEIGHT
    window.close()


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
