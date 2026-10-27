#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    default_address = os.environ.get('ROBOTBASE_FOXGLOVE_ADDRESS', '0.0.0.0')
    default_port = os.environ.get('ROBOTBASE_FOXGLOVE_PORT', '8766')

    address = LaunchConfiguration('address')
    port = LaunchConfiguration('port')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'address', default_value=default_address,
            description='WebSocket listen address. Use 127.0.0.1 for local-only access.'),
        DeclareLaunchArgument(
            'port', default_value=default_port,
            description='WebSocket port. Koko defaults to 8766; Sirius normally uses 8765.'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(params_root, 'common', 'foxglove.yaml')),
        LogInfo(msg=['Foxglove bridge listening on ws://', address, ':', port]),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='koko_foxglove_bridge',
            output='screen',
            parameters=[
                params_file,
                {
                    'address': address,
                    'port': ParameterValue(port, value_type=int),
                },
            ],
        ),
    ])
