"""Test Launcher2/Follower automatic selection without robot hardware."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from ar_reference_planner.auto_supervisor import (
    make_reference_process_command,
    ReferencePlannerAutoSupervisor,
)
from ar_reference_planner.follower_state import FollowerSnapshot
from ar_reference_planner.runtime_selection import (
    default_launcher_map_state_path,
    LauncherMapStateReader,
    resolve_follower_waypoint,
    runtime_transition,
    RuntimeSelection,
    SelectionStatus,
)
from PIL import Image
import pytest
import yaml


def _map(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    Image.new('L', (12, 10), color=255).save(directory / f'{name}.pgm')
    path = directory / f'{name}.yaml'
    path.write_text(yaml.safe_dump({
        'image': f'{name}.pgm',
        'mode': 'trinary',
        'resolution': 0.05,
        'origin': [-1.0, -2.0, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }), encoding='utf-8')
    return path


def _route(root: Path, name: str, offset: float = 0.0) -> Path:
    path = root / f'{name}.yaml'
    path.write_text(yaml.safe_dump({'waypoints': [
        {
            'number': index + 1,
            'x': offset + float(index),
            'y': 0.0,
            'angle_radians': 0.0,
        }
        for index in range(4)
    ]}), encoding='utf-8')
    return path


def _state_file(path: Path, selected_map: Path) -> None:
    path.write_text(yaml.safe_dump({
        'current_map': str(selected_map),
        'current_map_name': selected_map.stem,
        'current_map_yaml': selected_map.name,
    }), encoding='utf-8')


def _follower(route, index: int = 1) -> FollowerSnapshot:
    return FollowerSnapshot(
        stamp_ns=20_000_000_000,
        frame_id='map',
        route_sha256=route.sha256,
        next_index=index,
        next_number=route.waypoints[index].number,
        navigation_mode='to_pose',
        goal_generation=3,
        activity=1,
        reference_allowed=True,
        safe_end_index=3,
        waypoint_count=len(route.waypoints),
    )


def _selection(state_path: Path, map_yaml: Path, route_path: Path):
    _state_file(state_path, map_yaml)
    map_status = LauncherMapStateReader(
        state_path, wall_clock=lambda: state_path.stat().st_mtime
    ).read()
    from ar_reference_planner.route import load_route
    route = load_route(str(route_path))
    waypoint_status = resolve_follower_waypoint(
        str(route_path), _follower(route)
    )
    assert map_status.valid and waypoint_status.valid
    return RuntimeSelection(map_status.value, waypoint_status.value)


def test_default_state_path_follows_launcher_environment_without_fixed_map():
    environment = {
        'HOME': '/home/tester',
        'ROBOTBASE_ID': 'other_robot',
        'ROBOTBASE_LAUNCHER_ID': 'other_launcher2',
    }

    result = default_launcher_map_state_path(environment)

    assert result == Path(
        '/home/tester/.other_launcher2_nav2_current_map.yaml'
    )


def test_map_a_to_map_b_change_is_detected_by_content_identity(tmp_path):
    map_a = _map(tmp_path, 'map_a')
    map_b = _map(tmp_path, 'map_b')
    state_path = tmp_path / '.launcher_state.yaml'
    _state_file(state_path, map_a)
    now = state_path.stat().st_mtime
    reader = LauncherMapStateReader(state_path, wall_clock=lambda: now)

    first = reader.read()
    _state_file(state_path, map_b)
    os.utime(state_path, ns=(state_path.stat().st_atime_ns,
                             state_path.stat().st_mtime_ns + 1_000_000))
    second = reader.read()

    assert first.valid and first.value.map_yaml == map_a.resolve()
    assert second.valid and second.value.map_yaml == map_b.resolve()
    assert first.value.identity != second.value.identity


def test_missing_invalid_and_initially_stale_map_state_fail_closed(tmp_path):
    state_path = tmp_path / '.launcher_state.yaml'
    reader = LauncherMapStateReader(state_path, wall_clock=lambda: 1000.0)
    assert reader.read().reason == 'launcher_map_state_missing'

    state_path.write_text('current_map: relative/map.yaml\n', encoding='utf-8')
    assert reader.read().reason == 'launcher_current_map_not_absolute'

    selected = _map(tmp_path, 'map_a')
    _state_file(state_path, selected)
    os.utime(state_path, (100.0, 100.0))
    stale = LauncherMapStateReader(
        state_path,
        maximum_initial_age_sec=10.0,
        wall_clock=lambda: 1000.0,
    ).read()
    assert not stale.valid
    assert stale.reason.startswith('launcher_map_state_stale')


def test_keepout_mismatch_in_selected_map_fails_before_process_start(tmp_path):
    selected = _map(tmp_path, 'map_a')
    state_path = tmp_path / '.launcher_state.yaml'
    _state_file(state_path, selected)
    keepout = selected.parent / '.keepout' / selected.stem
    keepout.mkdir(parents=True)
    (keepout / 'settings.json').write_text(json.dumps({
        'enabled': True,
        'source_map': '/wrong/map.yaml',
        'map_metadata': {
            'resolution': 0.05,
            'origin': [-1.0, -2.0, 0.0],
            'size': [12, 10],
        },
    }), encoding='utf-8')
    reader = LauncherMapStateReader(
        state_path, wall_clock=lambda: state_path.stat().st_mtime
    )

    status = reader.read()

    assert not status.valid
    assert 'launcher_map_validation_failed' in status.reason
    assert 'source_map' in status.reason


def test_follower_parameter_resolves_route_and_rejects_every_mismatch(tmp_path):
    from ar_reference_planner.route import load_route
    route_a_path = _route(tmp_path, 'route_a')
    route_b_path = _route(tmp_path, 'route_b', offset=10.0)
    route_a = load_route(str(route_a_path))
    follower = _follower(route_a)

    assert resolve_follower_waypoint(str(route_a_path), follower).valid
    assert resolve_follower_waypoint('', follower).reason == (
        'follower_waypoint_parameter_empty')
    assert resolve_follower_waypoint('relative.yaml', follower).reason == (
        'follower_waypoint_path_not_absolute')
    assert resolve_follower_waypoint(
        str(tmp_path / 'missing.yaml'), follower
    ).reason.startswith('follower_waypoint_missing')
    assert resolve_follower_waypoint(str(route_b_path), follower).reason == (
        'follower_waypoint_route_hash_mismatch')


def test_runtime_transitions_cover_wait_start_switch_stop_and_restart(tmp_path):
    state_path = tmp_path / '.launcher_state.yaml'
    first = _selection(
        state_path, _map(tmp_path, 'map_a'), _route(tmp_path, 'route_a')
    )
    second = _selection(
        state_path, _map(tmp_path, 'map_b'), _route(tmp_path, 'route_b', 10.0)
    )

    assert runtime_transition(None, None, False) == 'wait'
    assert runtime_transition(None, first, False) == 'start'
    assert runtime_transition(first, first, True) == 'keep'
    assert runtime_transition(first, second, True) == 'restart'
    assert runtime_transition(first, None, True) == 'stop'
    assert runtime_transition(first, first, False) == 'restart'


def test_supervisor_queries_jazzy_parameter_services_api(tmp_path):
    from ar_reference_planner.route import load_route
    route = load_route(str(_route(tmp_path, 'route_a')))
    requested = []

    class Future:
        def add_done_callback(self, callback):
            self.callback = callback

    class ParameterClient:
        def services_are_ready(self):
            return True

        def get_parameters(self, names):
            requested.append(names)
            return Future()

    fake = SimpleNamespace(
        _follower=_follower(route),
        _parameter_future=None,
        _next_parameter_query_at=0.0,
        _parameter_retry=1.0,
        _parameter_client=ParameterClient(),
        _waypoint_status=None,
        _parameter_identity=None,
        _parameter_result=lambda future, expected: None,
    )

    ReferencePlannerAutoSupervisor._request_waypoint_parameter(
        fake, now=10.0
    )

    assert requested == [['waypoint_file']]
    assert fake._parameter_identity == (route.sha256, len(route.waypoints))
    assert fake._parameter_future is not None


def test_waiting_for_follower_state_recovers_after_verified_snapshot(tmp_path):
    """The startup wait must not remain sticky after Follower discovery."""
    state_path = tmp_path / '.launcher_state.yaml'
    selection = _selection(
        state_path, _map(tmp_path, 'map_a'), _route(tmp_path, 'route_a')
    )
    fake = SimpleNamespace(
        _map_reader=SimpleNamespace(
            read=lambda: SelectionStatus(
                selection.map_selection, 'launcher_map_verified'
            )
        ),
        _follower=None,
        _follower_received_at=None,
        _follower_timeout=2.5,
        _waypoint_status=SelectionStatus(
            selection.waypoint_selection, 'follower_waypoint_verified'
        ),
        _request_waypoint_parameter=lambda _now: None,
        _guard_map_route_transition=lambda candidate: (
            candidate, 'selection_verified'
        ),
    )

    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=10.0
    )
    assert desired is None
    assert reason == 'waiting_for_follower_state'

    fake._follower = _follower(selection.waypoint_selection.route)
    fake._follower_received_at = 10.1
    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=10.2
    )

    assert desired == selection
    assert reason == 'selection_verified'


def test_map_switch_waits_for_new_follower_route_before_restart(tmp_path):
    state_path = tmp_path / '.launcher_state.yaml'
    map_a = _map(tmp_path, 'map_a')
    map_b = _map(tmp_path, 'map_b')
    route_a_path = _route(tmp_path, 'route_a')
    route_b_path = _route(tmp_path, 'route_b', 10.0)
    active = _selection(state_path, map_a, route_a_path)
    wrong_pair = _selection(state_path, map_b, route_a_path)
    correct_pair = _selection(state_path, map_b, route_b_path)
    fake = SimpleNamespace(
        _active_selection=active,
        _blocked_map_identity=None,
        _blocked_route_sha256=None,
    )

    selected, reason = (
        ReferencePlannerAutoSupervisor._guard_map_route_transition(
            fake, wrong_pair
        )
    )
    assert selected is None
    assert reason == 'waiting_for_follower_route_after_map_change'

    selected, reason = (
        ReferencePlannerAutoSupervisor._guard_map_route_transition(
            fake, correct_pair
        )
    )
    assert selected == correct_pair
    assert reason == 'selection_verified'
    assert fake._blocked_map_identity is None
    assert fake._blocked_route_sha256 is None


def test_supervisor_invalidates_before_start_switch_and_follower_stop(tmp_path):
    state_path = tmp_path / '.launcher_state.yaml'
    first = _selection(
        state_path, _map(tmp_path, 'map_a'), _route(tmp_path, 'route_a')
    )
    second = _selection(
        state_path, _map(tmp_path, 'map_b'), _route(tmp_path, 'route_b', 10.0)
    )
    events = []

    class Process:
        running = False

        def stop(self):
            events.append('stop')
            self.running = False

        def start(self, command):
            events.append(('start', command))
            self.running = True

    fake = SimpleNamespace(
        _active_selection=None,
        _process=Process(),
        _next_process_retry_at=0.0,
        _process_retry=3.0,
        _invalidate=lambda reason, force=False: events.append(
            ('invalidate', reason, force)
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            'reference_namespace': 'ar_reference',
            'robot_base_frame': 'robot/base_footprint',
            'use_sim_time': False,
            'reference_params_file': '',
        }[name]),
        get_logger=lambda: SimpleNamespace(
            info=lambda message: events.append(('info', message))
        ),
    )

    ReferencePlannerAutoSupervisor._reconcile(
        fake, first, 'selection_verified', now=10.0
    )
    assert events[0] == (
        'invalidate', 'selection_change_in_progress', False
    )
    assert events[1] == 'stop'
    assert events[2] == (
        'invalidate', 'selection_change_in_progress', True
    )
    assert events[3][0] == 'start'
    fake._active_selection = first
    events.clear()

    ReferencePlannerAutoSupervisor._reconcile(
        fake, second, 'selection_verified', now=11.0
    )
    assert events[0] == (
        'invalidate', 'selection_change_in_progress', False
    )
    assert events[1] == 'stop'
    assert events[2] == (
        'invalidate', 'selection_change_in_progress', True
    )
    assert events[3][0] == 'start'
    fake._active_selection = second
    events.clear()

    ReferencePlannerAutoSupervisor._reconcile(
        fake, None, 'follower_state_stale', now=12.0
    )
    assert events == [
        ('invalidate', 'follower_state_stale', False),
        'stop',
        ('invalidate', 'follower_state_stale', True),
    ]


def test_process_command_uses_verified_dynamic_paths_and_no_control_action(tmp_path):
    state_path = tmp_path / '.launcher_state.yaml'
    selection = _selection(
        state_path, _map(tmp_path, 'map_a'), _route(tmp_path, 'route_a')
    )

    command = make_reference_process_command(
        selection,
        namespace='ar_reference',
        robot_base_frame='robot/base_footprint',
        use_sim_time=False,
    )
    joined = ' '.join(command)

    assert f'map:={selection.map_selection.map_yaml}' in command
    assert (
        f'waypoint_file:={selection.waypoint_selection.waypoint_yaml}'
        in command
    )
    assert '/plan' not in joined
    assert 'cmd_vel' not in joined
    assert 'NavigateToPose' not in joined
    assert 'NavigateThroughPoses' not in joined


def test_dedicated_launch_has_no_manual_map_or_waypoint_argument():
    package = Path(__file__).parents[1]
    launch = (
        package / 'launch' / 'launcher2_reference_ar.launch.py'
    ).read_text(encoding='utf-8')

    assert "DeclareLaunchArgument('map'" not in launch
    assert "DeclareLaunchArgument('waypoint_file'" not in launch
    assert "'reference_path_enabled': True" in launch
    assert (
        "DeclareLaunchArgument('start_viewer', default_value='true')"
        in launch
    )
    assert "'viewer_calibration_overlay_file', default_value=''" in launch
    assert 'if calibration_overlay:' in launch
    assert "'viewer_runtime_overlay_file', default_value=''" in launch
    assert 'if runtime_overlay:' in launch
    assert launch.index('if calibration_overlay:') < launch.index(
        'if perception_overlay:'
    )


def test_installed_desktop_runner_is_read_only_and_session_scoped():
    package = Path(__file__).parents[1]
    runner = (
        package / 'scripts' / 'run_kokochan_navi_launcher2'
    ).read_text(encoding='utf-8')
    setup = (package / 'setup.py').read_text(encoding='utf-8')
    desktop = (
        package / 'desktop' / 'kokochan_navi_launcher2.desktop'
    ).read_text(encoding='utf-8')
    launch = (
        package / 'launch' / 'launcher2_reference_ar.launch.py'
    ).read_text(encoding='utf-8')

    assert "'scripts/run_kokochan_navi_launcher2'" in setup
    assert "'scripts/install_kokochan_navi_desktop'" in setup
    assert "'scripts/run_kokochan_navi_offline_review'" in setup
    assert 'launcher2_reference_ar.launch.py' in runner
    assert 'viewer_runtime_overlay_file:=' in runner
    assert 'ar_path_viewer_launcher2.yaml' in runner
    assert 'ROS_DOMAIN_ID:-54' in runner
    assert 'kokochan_navi_${stamp}' in runner
    assert '.mp4' in runner
    assert '.log' in runner
    assert '.launch.log' in runner
    assert 'OnProcessExit' in launch
    assert "Shutdown(reason='AR Viewer exited')" in launch
    supervisor = (
        package / 'ar_reference_planner' / 'auto_supervisor.py'
    ).read_text(encoding='utf-8')
    bridge = (
        package / 'ar_reference_planner' / 'legacy_follower_bridge.py'
    ).read_text(encoding='utf-8')
    assert 'ExternalShutdownException' in supervisor
    assert 'ExternalShutdownException' in bridge
    assert 'except KeyboardInterrupt' in supervisor
    assert 'except KeyboardInterrupt' in bridge
    assert 'flock -n' in runner
    assert 'ros2 node list' in runner
    assert 'fuser' in runner
    source_start = runner.index('set +u')
    source_end = runner.index('set -u', source_start)
    assert source_start < runner.index('source /opt/ros/jazzy/setup.bash') < source_end
    assert 'AMENT_TRACE_SETUP_FILES' not in runner
    assert 'AMENT_PYTHON_EXECUTABLE' not in runner
    assert 'cmd_vel' not in runner
    assert 'NavigateToPose' not in runner
    assert 'send_goal' not in runner
    assert 'run_kokochan_navi_launcher2' in desktop
    assert 'waypoint_follow' not in desktop
    assert 'nav2' not in desktop.lower()
    assert '/home/koko-chan-nuc' not in desktop
    assert 'viewer_perception_overlay_file:=' in runner
    assert 'image_input_uri:' in runner
    assert 'C920 video device was not found' in runner
    assert 'KOKO_NAVI_FPS' in runner
    assert 'KOKO_NAVI_FPS must be 20 or 30' in runner
    assert 'fps: ${comparison_fps}.0' in runner
    assert 'image_input_type: camera_timestamped' in runner
    assert '${comparison_fps}fps' in runner
    assert "'viewer_perception_overlay_file', default_value=''" in launch


def test_autonomous_shared_camera_entry_enables_review_profile_only_for_autonomy():
    repository = Path(__file__).parents[4]
    wrapper = (
        repository / 'camera_v8' / 'development_20261123'
        / 'run_ar_with_v8_share.sh'
    ).read_text(encoding='utf-8')
    manual = (
        repository / 'camera_v8' / 'development_20261123'
        / 'run_ar_manual_mapping_viewer.sh'
    ).read_text(encoding='utf-8')
    runner = (
        Path(__file__).parents[1] / 'scripts'
        / 'run_kokochan_navi_launcher2'
    ).read_text(encoding='utf-8')
    viewer_source = (
        repository / 'ar_ws/src/ar_path_viewer/ar_path_viewer/viewer_node.py'
    ).read_text(encoding='utf-8')

    autonomous_case = wrapper.split(
        'manual_mapping) RUNNER=', 1
    )[0]
    assert 'KOKO_NAVI_PERCEPTION_REVIEW:-1' in autonomous_case
    assert 'KOKO_NAVI_PERCEPTION_REVIEW=${KOKO_NAVI_PERCEPTION_REVIEW}' in wrapper
    for key in (
        'perception_costmap_enabled',
        'perception_minimap_enabled',
        'perception_alert_enabled',
    ):
        assert f"'{key}':False" in manual
    assert 'Perception profile: enabled overlay=' in runner
    assert 'KOKO_NAVI_PERCEPTION_REVIEW:-1' in runner
    assert 'costmap_subscription=true minimap=true alerts=true' in runner
    assert 'camera_projection=false' in runner
    assert '"perception_costmap_configuration "' in viewer_source
    assert 'f"enabled={self._perception_costmap_enabled} "' in viewer_source
    assert '"perception_review_ui_configuration "' in viewer_source
    assert 'f"minimap_enabled={self._perception_minimap_enabled} "' in viewer_source
    assert 'f"alert_enabled={self._perception_alert_enabled} "' in viewer_source


def test_desktop_installer_backs_up_and_restores(tmp_path):
    package = Path(__file__).parents[1]
    installer = package / 'scripts' / 'install_kokochan_navi_desktop'
    desktop = tmp_path / 'Desktop'
    desktop.mkdir()
    destination = desktop / 'ココちゃんナビ.desktop'
    destination.write_text('old entry\n', encoding='utf-8')
    environment = dict(os.environ)
    environment['HOME'] = str(tmp_path)
    environment['XDG_DESKTOP_DIR'] = str(desktop)
    environment['KOKO_NAVI_REPOSITORY'] = str(Path(__file__).parents[4])

    subprocess.run(
        [str(installer)], check=True, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    installed = destination.read_text(encoding='utf-8')
    assert 'select_kokochan_navi_mode.sh' in installed
    assert 'Path=' in installed
    assert '使用するモードを選んで' in installed
    assert 'Launcher2とは独立' not in installed
    assert list(desktop.glob('ココちゃんナビ.desktop.backup.*'))

    subprocess.run(
        [str(installer), '--restore'], check=True, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert destination.read_text(encoding='utf-8') == 'old entry\n'


def test_desktop_reinstall_keeps_original_as_restore_target(tmp_path):
    package = Path(__file__).parents[1]
    installer = package / 'scripts' / 'install_kokochan_navi_desktop'
    desktop = tmp_path / 'Desktop'
    desktop.mkdir()
    destination = desktop / 'ココちゃんナビ.desktop'
    destination.write_text('user owned entry\n', encoding='utf-8')
    environment = dict(os.environ)
    environment.update({
        'HOME': str(tmp_path),
        'XDG_DESKTOP_DIR': str(desktop),
        'KOKO_NAVI_REPOSITORY': str(Path(__file__).parents[4]),
    })

    for _ in range(2):
        subprocess.run(
            [str(installer)], check=True, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    backups = list(desktop.glob('ココちゃんナビ.desktop.backup.*'))
    assert len(backups) == 1
    assert backups[0].read_text(encoding='utf-8') == 'user owned entry\n'

    subprocess.run(
        [str(installer), '--restore'], check=True, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert destination.read_text(encoding='utf-8') == 'user owned entry\n'


def test_viewer_parameter_sources_apply_perception_profile_last():
    launch_path = (
        Path(__file__).parents[1] / 'launch'
        / 'launcher2_reference_ar.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'launcher2_reference_ar_audit', launch_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._ordered_viewer_parameters(
        'old-machine.yaml', 'runtime.yaml', 'calibration.yaml',
        'perception.yaml',
    ) == [
        'old-machine.yaml', 'runtime.yaml', 'calibration.yaml',
        'perception.yaml',
    ]
    assert module._ordered_viewer_parameters(
        'base.yaml', 'runtime.yaml', '', ''
    ) == ['base.yaml', 'runtime.yaml']


def test_old_machine_yaml_cannot_override_final_perception_profile(tmp_path):
    launch_path = (
        Path(__file__).parents[1] / 'launch'
        / 'launcher2_reference_ar.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'launcher2_reference_ar_merge_audit', launch_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = tmp_path / 'old-machine.yaml'
    runtime = tmp_path / 'runtime.yaml'
    calibration = tmp_path / 'calibration.yaml'
    old.write_text(yaml.safe_dump({'ar_path_viewer': {'ros__parameters': {
        'perception_costmap_enabled': False,
        'perception_minimap_enabled': False,
        'perception_alert_enabled': False,
    }}}), encoding='utf-8')
    runtime.write_text(yaml.safe_dump({'ar_path_viewer': {'ros__parameters': {
        'image_input_type': 'ros_image',
        'perception_costmap_enabled': False,
    }}}), encoding='utf-8')
    calibration.write_text(yaml.safe_dump({
        'ar_path_viewer': {'ros__parameters': {'fx': 658.0}}
    }), encoding='utf-8')
    perception = (
        Path(__file__).parents[2] / 'ar_path_viewer' / 'config'
        / 'perception_costmap_review_overlay.yaml'
    )
    sources = module._ordered_viewer_parameters(
        str(old), str(runtime), str(calibration), str(perception)
    )
    effective = {}
    for source in sources:
        document = yaml.safe_load(Path(source).read_text(encoding='utf-8'))
        effective.update(document['ar_path_viewer']['ros__parameters'])

    assert effective['perception_costmap_enabled'] is True
    assert effective['perception_minimap_enabled'] is True
    assert effective['perception_alert_enabled'] is True
    assert effective['perception_costmap_camera_overlay_enabled'] is False
    assert effective['image_input_type'] == 'ros_image'
    assert effective['fx'] == 658.0


def test_offline_review_filters_control_topics_and_rejects_production_domain():
    package = Path(__file__).parents[1]
    review = (
        package / 'scripts' / 'run_kokochan_navi_offline_review'
    ).read_text(encoding='utf-8')
    assert 'KOKO_NAVI_REVIEW_DOMAIN_ID:-154' in review
    assert 'production ROS_DOMAIN_ID=54' in review
    assert '/local_costmap/costmap' in review
    assert '/plan' in review
    assert '/cmd_vel' not in review
    assert 'NavigateToPose' not in review
    assert 'send_goal' not in review
    assert 'kill -TERM "${bag_pid}"' in review
    assert 'kill -KILL "${bag_pid}"' in review


@pytest.mark.parametrize('bad_value', (-1.0, float('nan')))
def test_negative_or_nonfinite_initial_state_age_is_rejected(
    tmp_path, bad_value
):
    with pytest.raises(ValueError):
        LauncherMapStateReader(
            tmp_path / 'state.yaml', maximum_initial_age_sec=bad_value
        )


def test_supervisor_uses_scoped_follower_get_service():
    from types import SimpleNamespace
    requested = []

    class Future:
        def add_done_callback(self, callback):
            self.callback = callback

    fake = SimpleNamespace(
        _follower=SimpleNamespace(
            route_sha256='a' * 64, waypoint_count=30,
        ),
        _parameter_future=None,
        _next_parameter_query_at=0.0,
        _parameter_retry=1.0,
        _parameter_get_service=SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=lambda req: requested.append(list(req.names)) or Future(),
        ),
        _parameter_client=SimpleNamespace(
            services_are_ready=lambda: False,
        ),
        _parameter_result=lambda result, expected: None,
    )
    ReferencePlannerAutoSupervisor._request_waypoint_parameter(fake, now=10.0)
    assert requested == [['waypoint_file']]
    assert fake._parameter_identity == ('a' * 64, 30)
