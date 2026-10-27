#!/usr/bin/env python3

import os
import json
import tempfile
from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def configure_keepout(context):
    """Load only the saved keepout belonging to the selected map."""
    map_name = LaunchConfiguration('map').perform(context)
    if not map_name or LaunchConfiguration('localization').perform(context) == 'slam':
        return []
    map_path = Path(map_name).resolve()
    state_path = map_path.parent / '.keepout' / map_path.stem / 'settings.json'
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text())
    if not state.get('enabled', False):
        return []
    source_map = Path(state.get('source_map', ''))
    same_file = source_map == map_path
    # A copied workspace keeps the map and mask together, but changes HOME.
    map_location = ('maps_waypoints', 'maps', map_path.name)
    relocated_file = (tuple(source_map.parts[-3:]) == map_location and
                      tuple(map_path.parts[-3:]) == map_location)
    if not (same_file or relocated_file):
        raise RuntimeError('Keepout mask belongs to a different map.')
    meta = yaml.safe_load(map_path.read_text())
    # Match geometry before activating a previously saved mask.
    from PIL import Image
    with Image.open(map_path.parent / meta['image']) as image:
        geometry = dict(resolution=meta['resolution'], origin=meta['origin'], size=list(image.size))
    if state.get('map_metadata') != geometry:
        raise RuntimeError('Keepout geometry does not match the selected map. Re-save it in the editor.')
    mask = state_path.parent / 'mask.yaml'
    if not mask.exists():
        raise RuntimeError('Enabled keepout mask is missing: ' + str(mask))
    params = yaml.safe_load(Path(LaunchConfiguration('params_file').perform(context)).read_text())
    for name in ('global_costmap', 'local_costmap'):
        block = params[name][name]['ros__parameters']
        filters = block.setdefault('filters', [])
        if 'keepout_filter' not in filters:
            filters.append('keepout_filter')
        block['keepout_filter'] = dict(plugin='nav2_costmap_2d::KeepoutFilter', enabled=True,
                                     filter_info_topic='/costmap_filter_info')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', prefix='robotbase_keepout_', delete=False) as f:
        yaml.safe_dump(params, f)
        params_path = f.name
    import atexit
    atexit.register(lambda: os.path.exists(params_path) and os.unlink(params_path))
    sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    return [
        SetLaunchConfiguration('params_file', params_path),
        Node(package='nav2_map_server', executable='map_server', name='filter_mask_server',
             output='screen', parameters=[dict(use_sim_time=sim_time, frame_id='map',
                 topic_name='/keepout_filter_mask', yaml_filename=str(mask))]),
        Node(package='nav2_map_server', executable='costmap_filter_info_server',
             name='costmap_filter_info_server', output='screen',
             parameters=[dict(use_sim_time=sim_time, type=0, filter_info_topic='/costmap_filter_info',
                              mask_topic='/keepout_filter_mask', base=0.0, multiplier=1.0)]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_keepout', output='screen',
             parameters=[dict(use_sim_time=sim_time, autostart=True,
                              node_names=['filter_mask_server', 'costmap_filter_info_server'])]),
    ]


def generate_launch_description():
    bringup_share = get_package_share_directory('robotbase_bringup')
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    default_params = os.path.join(params_root, 'real', 'nav2.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    nav_to_pose_bt = PathJoinSubstitution([
        bringup_share, 'behavior_trees',
        'navigate_to_pose_w_replanning_goal_patience_and_recovery_no_spin.xml',
    ])
    nav_through_poses_bt = PathJoinSubstitution([
        bringup_share, 'behavior_trees',
        'navigate_through_poses_w_replanning_goal_patience_and_recovery_no_spin.xml',
    ])

    localization = LaunchConfiguration('localization')
    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    tf_prefix = LaunchConfiguration('tf_prefix')
    base_frame = PathJoinSubstitution([tf_prefix, 'base_footprint'])
    odom_frame = PathJoinSubstitution([tf_prefix, 'odom'])
    odom_topic = LaunchConfiguration('odom_topic')

    configured = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={
            'use_sim_time': use_sim_time,
            'base_frame_id': base_frame,
            'robot_base_frame': base_frame,
            'odom_frame_id': odom_frame,
            'odom_topic': odom_topic,
            'local_costmap.local_costmap.ros__parameters.global_frame': odom_frame,
            'behavior_server.ros__parameters.local_frame': odom_frame,
            'default_nav_to_pose_bt_xml': nav_to_pose_bt,
            'default_nav_through_poses_bt_xml': nav_through_poses_bt,
        },
        convert_types=True,
    )
    parameters = ParameterFile(configured, allow_substs=True)
    amcl_mode = IfCondition(PythonExpression(["'", localization, "' == 'amcl'"]))
    static_mode = IfCondition(PythonExpression(["'", localization, "' == 'static'"]))
    map_mode = IfCondition(PythonExpression(["'", localization, "' != 'slam'"]))
    tf_remaps = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen', parameters=[parameters, {'yaml_filename': map_file}],
        remappings=tf_remaps, condition=map_mode,
    )
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[parameters], remappings=tf_remaps, condition=amcl_mode,
    )
    static_localization = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_odom_ground_truth', output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', odom_frame,
        ],
        condition=static_mode,
    )
    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
        condition=amcl_mode,
    )
    static_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_static_map', output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server'],
        }],
        condition=static_mode,
    )

    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen', parameters=[parameters],
        remappings=tf_remaps + [('cmd_vel', 'cmd_vel_nav')],
    )
    smoother = Node(
        package='nav2_smoother', executable='smoother_server',
        name='smoother_server', output='screen', parameters=[parameters],
        remappings=tf_remaps,
    )
    planner = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen', parameters=[parameters],
        remappings=tf_remaps,
    )
    behaviors = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen', parameters=[parameters],
        remappings=tf_remaps + [('cmd_vel', 'cmd_vel_nav')],
    )
    navigator = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen', parameters=[parameters],
        remappings=tf_remaps,
    )
    velocity_smoother = Node(
        package='nav2_velocity_smoother', executable='velocity_smoother',
        name='velocity_smoother', output='screen', parameters=[parameters],
        remappings=tf_remaps + [('cmd_vel', 'cmd_vel_nav')],
    )
    waypoint_follower = Node(
        package='nav2_waypoint_follower', executable='waypoint_follower',
        name='waypoint_follower', output='screen', parameters=[parameters],
        remappings=tf_remaps,
    )
    navigation_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': [
                'controller_server', 'smoother_server', 'planner_server',
                'behavior_server', 'bt_navigator', 'velocity_smoother',
                'waypoint_follower',
            ],
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'localization', default_value='amcl',
            description='amcl, static, or slam (map and localization supplied by slam_toolbox).'),
        DeclareLaunchArgument(
            'map', default_value='',
            description='Absolute map YAML path; unused when localization:=slam.'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        DeclareLaunchArgument('odom_topic', default_value='/odom/filtered'),
        OpaqueFunction(function=configure_keepout),
        map_server,
        amcl,
        static_localization,
        localization_lifecycle,
        static_lifecycle,
        controller,
        smoother,
        planner,
        behaviors,
        navigator,
        velocity_smoother,
        waypoint_follower,
        navigation_lifecycle,
    ])
