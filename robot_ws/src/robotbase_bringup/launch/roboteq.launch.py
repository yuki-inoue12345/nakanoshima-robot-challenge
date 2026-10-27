#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory('robotbase_bringup')
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    default_params = os.path.join(params_root, 'real', 'roboteq.yaml')
    urdf_path = os.path.join(share, 'urdf', 'robotbase.urdf')
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    tf_prefix = LaunchConfiguration('tf_prefix')
    pub_odom_tf = LaunchConfiguration('pub_odom_tf')
    params_file = LaunchConfiguration('params_file')
    odom_frame = PathJoinSubstitution([tf_prefix, 'odom'])
    base_frame = PathJoinSubstitution([tf_prefix, 'base_footprint'])
    frame_prefix = PythonExpression(["'", tf_prefix, "/'"])
    configured = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'pub_odom_tf': pub_odom_tf,
        },
        convert_types=True,
    )
    parameters = ParameterFile(configured, allow_substs=True)

    return LaunchDescription([
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        DeclareLaunchArgument('pub_odom_tf', default_value='false'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robotbase_state_publisher', output='screen',
            parameters=[{
                'use_sim_time': False,
                'robot_description': robot_description,
                'frame_prefix': frame_prefix,
            }],
        ),
        Node(
            package='roboteq_ros2_driver', executable='roboteq_ros2_driver',
            name='roboteq_ros2_driver', output='screen', respawn=True,
            parameters=[parameters],
        ),
    ])
