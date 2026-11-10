"""Reproducible planning integration test without robot hardware."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
            'map_topic': '/ar_reference_test/map',
            'robot_base_frame': 'robot/base_footprint',
            'waypoint_file': LaunchConfiguration('waypoint_file'),
            'target_length_m': '11.0',
            'minimum_length_m': '8.0',
            'maximum_length_m': '15.0',
            'refresh_period_sec': '0.0',
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
    map_publisher = Node(
        package='ar_reference_planner',
        executable='synthetic_map_publisher',
        name='synthetic_reference_map',
        output='screen',
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
        map_publisher,
        transform,
        mock_state,
        reference_launch,
    ])
