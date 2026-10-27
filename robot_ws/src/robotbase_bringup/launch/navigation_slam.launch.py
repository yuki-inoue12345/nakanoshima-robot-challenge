#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('robotbase_bringup')
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))

    tf_prefix = LaunchConfiguration('tf_prefix')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    slam_config_file = LaunchConfiguration('slam_config_file')

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'slam.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'tf_prefix': tf_prefix,
            'slam_config_file': slam_config_file,
        }.items(),
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'localization': 'slam',
            'tf_prefix': tf_prefix,
            'odom_topic': '/odom/filtered',
            'params_file': nav2_params_file,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=os.path.join(params_root, 'real', 'nav2.yaml')),
        DeclareLaunchArgument(
            'slam_config_file',
            default_value=os.path.join(params_root, 'real', 'slam_toolbox.yaml')),
        slam,
        nav2,
    ])
