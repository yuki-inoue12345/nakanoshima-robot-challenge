"""Reproducible planning integration test without robot hardware."""

from ament_index_python.packages import get_package_share_directory
from ar_reference_planner.test_fixture import create_test_map_bundle
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _prepare_fixture(context):
    enabled = LaunchConfiguration('keepout_enabled').perform(context).lower()
    if enabled not in ('true', 'false'):
        raise RuntimeError('keepout_enabled must be true or false')
    selected, mask = create_test_map_bundle(enabled == 'true')
    return [
        SetLaunchConfiguration('test_map_yaml', str(selected)),
        SetLaunchConfiguration('test_mask_yaml', str(mask or '')),
    ]


def generate_launch_description():
    """Return a hardware-free reference-planning integration environment."""
    share = get_package_share_directory('ar_reference_planner')
    reference_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{share}/launch/reference_planner.launch.py'
        ),
        launch_arguments={
            'namespace': 'ar_reference_test',
            'params_file': f'{share}/config/reference_planner_test.yaml',
            'map': LaunchConfiguration('test_map_yaml'),
            'map_topic': '/ar_reference_test/map',
            'keepout_filter_info_topic':
                '/ar_reference_test/costmap_filter_info',
            'keepout_mask_topic': '/ar_reference_test/keepout_filter_mask',
            'keepout_mask_state_service': '/filter_mask_server/get_state',
            'keepout_info_state_service':
                '/costmap_filter_info_server/get_state',
            'robot_base_frame': 'robot/base_footprint',
            'waypoint_file': LaunchConfiguration('waypoint_file'),
            'target_length_m': '11.0',
            'minimum_length_m': '8.0',
            'maximum_length_m': '15.0',
            'refresh_period_sec': '3.0',
            'result_timeout_sec': '5.0',
            'validity_sec': '30.0',
            'max_follower_state_age_sec': '2.5',
        }.items(),
    )
    mock_state = Node(
        package='ar_reference_planner',
        executable='mock_follower_state',
        name='mock_waypoint_follower_state',
        output='screen',
        parameters=[{
            'waypoint_file': LaunchConfiguration('waypoint_file'),
            'next_waypoint_index': LaunchConfiguration('next_waypoint_index'),
            'safe_section_end_index': LaunchConfiguration(
                'safe_section_end_index'),
            'goal_generation': LaunchConfiguration('goal_generation'),
            'navigation_mode': LaunchConfiguration('navigation_mode'),
            'activity': LaunchConfiguration('activity'),
        }],
    )
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='reference_test_map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'frame_id': 'map',
            'topic_name': '/ar_reference_test/map',
            'yaml_filename': LaunchConfiguration('test_map_yaml'),
        }],
    )
    map_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_reference_test_map',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['reference_test_map_server'],
        }],
    )
    mask_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='filter_mask_server',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_keepout_servers')),
        parameters=[{
            'use_sim_time': False,
            'frame_id': 'map',
            'topic_name': '/ar_reference_test/keepout_filter_mask',
            'yaml_filename': LaunchConfiguration('test_mask_yaml'),
        }],
    )
    info_server = Node(
        package='nav2_map_server',
        executable='costmap_filter_info_server',
        name='costmap_filter_info_server',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_keepout_servers')),
        parameters=[{
            'use_sim_time': False,
            'type': 0,
            'filter_info_topic': '/ar_reference_test/costmap_filter_info',
            'mask_topic': '/ar_reference_test/keepout_filter_mask',
            'base': 0.0,
            'multiplier': 1.0,
        }],
    )
    keepout_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_reference_test_keepout',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_keepout_servers')),
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['filter_mask_server', 'costmap_filter_info_server'],
        }],
    )
    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='reference_test_robot_pose',
        output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', 'robot/base_footprint',
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument('waypoint_file', default_value=''),
        DeclareLaunchArgument('next_waypoint_index', default_value='1'),
        DeclareLaunchArgument('safe_section_end_index', default_value='7'),
        DeclareLaunchArgument('goal_generation', default_value='1'),
        DeclareLaunchArgument('navigation_mode', default_value='through_poses'),
        DeclareLaunchArgument('activity', default_value='navigating'),
        DeclareLaunchArgument('keepout_enabled', default_value='true'),
        DeclareLaunchArgument(
            'start_keepout_servers',
            default_value=LaunchConfiguration('keepout_enabled'),
        ),
        OpaqueFunction(function=_prepare_fixture),
        map_server,
        map_lifecycle,
        mask_server,
        info_server,
        keepout_lifecycle,
        transform,
        mock_state,
        reference_launch,
    ])
