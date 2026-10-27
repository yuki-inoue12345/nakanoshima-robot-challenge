#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    slam_share = get_package_share_directory('slam_toolbox')
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    default_params = os.path.join(params_root, 'real', 'slam_toolbox.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_config_file = LaunchConfiguration('slam_config_file')
    tf_prefix = LaunchConfiguration('tf_prefix')
    base_frame = PathJoinSubstitution([tf_prefix, 'base_footprint'])
    odom_frame = PathJoinSubstitution([tf_prefix, 'odom'])

    configured = RewrittenYaml(
        source_file=slam_config_file,
        root_key='',
        param_rewrites={
            'use_sim_time': use_sim_time,
            'base_frame': base_frame,
            'odom_frame': odom_frame,
        },
        convert_types=True,
    )
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'slam_params_file': configured,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('slam_config_file', default_value=default_params),
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        slam,
    ])
