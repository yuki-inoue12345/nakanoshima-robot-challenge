#!/usr/bin/env python3

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _persistent_rviz_node(context, template_path):
    tf_prefix = LaunchConfiguration('tf_prefix').perform(context).strip('/')
    reset_config = LaunchConfiguration('reset_config').perform(context).lower() in {
        '1', 'true', 'yes', 'on'}
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    workspace_root = os.path.dirname(params_root)
    config_dir = os.environ.get(
        'ROBOTBASE_RVIZ_CONFIG_DIR', os.path.join(workspace_root, 'rviz'))
    os.makedirs(config_dir, exist_ok=True)

    config_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', tf_prefix) or 'no_prefix'
    persistent_path = os.path.join(config_dir, f'robotbase_{config_id}.rviz')
    if reset_config or not os.path.exists(persistent_path):
        with open(template_path, encoding='utf-8') as template_file:
            config = template_file.read().replace('TF_PREFIX', tf_prefix)
        with open(persistent_path, 'w', encoding='utf-8') as config_file:
            config_file.write(config)

    return [Node(
        package='rviz2', executable='rviz2', name='koko_rviz',
        output='screen', arguments=['-d', persistent_path],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )]


def generate_launch_description():
    share = get_package_share_directory('robotbase_bringup')
    urdf_path = os.path.join(share, 'urdf', 'robotbase.urdf')
    rviz_template = os.path.join(share, 'rviz', 'robotbase.rviz')
    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_description = LaunchConfiguration('publish_description')
    tf_prefix = LaunchConfiguration('tf_prefix')
    frame_prefix = PythonExpression(["'", tf_prefix, "/'"])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'publish_description', default_value='true',
            description='Publish the robot URDF. Set false when the simulator already does it.'),
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        DeclareLaunchArgument(
            'reset_config', default_value='false',
            description='Overwrite the persistent RViz config from the project template.'),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robotbase_state_publisher', output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description,
                'frame_prefix': frame_prefix,
            }],
            condition=IfCondition(publish_description),
        ),
        OpaqueFunction(
            function=_persistent_rviz_node,
            kwargs={'template_path': rviz_template}),
    ])
