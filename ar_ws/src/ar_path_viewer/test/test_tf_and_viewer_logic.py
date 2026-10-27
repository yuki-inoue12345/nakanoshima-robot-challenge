import math

import numpy as np

from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.tf_adapter import quaternion_matrix_xyzw, transform_points
from ar_path_viewer.viewer_logic import project_path_state


def test_quaternion_and_translation_transform_points():
    quaternion = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    transformed = transform_points(np.array(((1.0, 0.0, 0.0),)), (2.0, 3.0, 0.0), quaternion)
    assert np.allclose(transformed, ((2.0, 4.0, 0.0)), atol=1e-9)


def test_tf_unavailable_has_explicit_state():
    def unavailable(points, frame):
        raise RuntimeError("no transform")

    state, segments = project_path_state(
        np.array(((1.0, 0.0, 0.0), (4.0, 0.0, 0.0))),
        "map", "robot/base_footprint", unavailable,
        CameraMount(), CameraIntrinsics(), 0.1,
    )
    assert state == "TF UNAVAILABLE"
    assert segments == []


def test_path_out_of_view_has_explicit_state():
    state, segments = project_path_state(
        np.array(((-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        "robot/base_footprint", "robot/base_footprint", lambda points, frame: points,
        CameraMount(), CameraIntrinsics(), 0.1,
    )
    assert state == "PATH OUT OF VIEW"
    assert segments == []
