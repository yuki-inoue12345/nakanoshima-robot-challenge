"""Static safety contracts for the planning-only package."""

from pathlib import Path


PACKAGE = Path(__file__).parents[1]


def test_launch_remaps_internal_plan_and_uses_reference_topic():
    """Keep every generated plan away from the control plan topic."""
    launch = (PACKAGE / 'launch' / 'reference_planner.launch.py').read_text()

    assert "('plan', 'planner_internal_plan')" in launch
    assert "'reference_path_topic': '/ar/reference_path'" in launch
    assert "'reference_path_state_topic': '/ar/reference_path_state'" in launch
    assert "package='nav2_controller'" not in launch
    assert "package='nav2_bt_navigator'" not in launch


def test_request_node_uses_only_compute_path_action():
    """Prevent control actions and velocity outputs entering this package."""
    source = (PACKAGE / 'ar_reference_planner' / 'request_node.py').read_text()

    assert 'ComputePathThroughPoses' in source
    assert 'NavigateToPose' not in source
    assert 'NavigateThroughPoses' not in source
    assert 'cmd_vel' not in source
    assert "self._invalidate('startup_no_reference_path')" in source
    assert 'GetState' in source
    assert 'State.PRIMARY_STATE_ACTIVE' in source


def test_production_topics_are_absolute_for_shared_inputs():
    """Use root shared sensor topics even below the reference namespace."""
    config = (PACKAGE / 'config' / 'reference_planner.yaml').read_text()

    assert 'map_topic: /map' in config
    assert 'topic: /scan3' in config
    assert 'topic: /hokuyo_scan' in config
    assert 'filters:' not in config
