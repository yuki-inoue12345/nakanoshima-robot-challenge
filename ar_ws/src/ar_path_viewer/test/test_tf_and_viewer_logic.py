import math

import numpy as np

from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.tf_adapter import quaternion_matrix_xyzw, transform_points
from ar_path_viewer.viewer_logic import project_path_state, project_path_state_detailed


def test_quaternion_and_translation_transform_points():
    quaternion = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    transformed = transform_points(np.array(((1.0, 0.0, 0.0),)), (2.0, 3.0, 0.0), quaternion)
    assert np.allclose(transformed, ((2.0, 4.0, 0.0)), atol=1e-9)


def test_target_from_source_map_to_base_keeps_long_forward_path_visible():
    # Robot is at map (10, 5), facing map +y.  The tf2 lookup target=base,
    # source=map therefore rotates by -90 deg and translates by (-5, +10).
    map_path = np.column_stack((
        np.full(119, 10.0), np.linspace(7.5, 25.0, 119), np.zeros(119)
    ))
    quaternion_base_from_map = (
        0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4)
    )

    def map_to_base(points, frame):
        assert frame == "map"
        return transform_points(points, (-5.0, 10.0, 0.0), quaternion_base_from_map)

    state, segments, stats = project_path_state_detailed(
        map_path, "map", "robot/base_footprint", map_to_base,
        CameraMount(), CameraIntrinsics(), 0.1,
    )
    assert state == "ACTIVE"
    assert len(segments) == 118
    assert stats is not None
    assert math.isclose(stats.path_length_m, 17.5, abs_tol=1e-9)
    assert math.isclose(stats.visible_path_length_m, 17.5, abs_tol=1e-9)
    assert stats.visible_depth_max_m > 19.0


def test_tf_unavailable_has_explicit_state():
    errors = []

    def unavailable(points, frame):
        raise RuntimeError("no transform")

    state, segments = project_path_state(
        np.array(((1.0, 0.0, 0.0), (4.0, 0.0, 0.0))),
        "map", "robot/base_footprint", unavailable,
        CameraMount(), CameraIntrinsics(), 0.1, on_tf_error=errors.append,
    )
    assert state == "TF UNAVAILABLE"
    assert segments == []
    assert len(errors) == 1
    assert str(errors[0]) == "no transform"


def test_path_out_of_view_has_explicit_state():
    state, segments = project_path_state(
        np.array(((-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        "robot/base_footprint", "robot/base_footprint", lambda points, frame: points,
        CameraMount(), CameraIntrinsics(), 0.1,
    )
    assert state == "PATH OUT OF VIEW"
    assert segments == []


def test_detailed_path_state_explains_out_of_view_reason():
    state, segments, stats = project_path_state_detailed(
        np.array(((-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        "robot/base_footprint", "robot/base_footprint", lambda points, frame: points,
        CameraMount(), CameraIntrinsics(), 0.1,
    )
    assert state == "PATH OUT OF VIEW"
    assert segments == []
    assert stats is not None
    assert stats.behind_points == 2
    assert stats.out_of_view_reason == "all Path points are behind the camera"
