#!/usr/bin/env python3

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _launch_setup(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    calibration_file = LaunchConfiguration('calibration_file').perform(context)
    tf_prefix = LaunchConfiguration('tf_prefix').perform(context).strip('/')

    with open(params_file, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    driver_params = dict(config['velodyne_driver_node']['ros__parameters'])
    pointcloud_params = dict(config['velodyne_transform_node']['ros__parameters'])
    laserscan_params = dict(config['velodyne_laserscan_node']['ros__parameters'])
    driver_params['frame_id'] = (
        f'{tf_prefix}/lidar_link' if tf_prefix else 'lidar_link')
    pointcloud_params['calibration'] = calibration_file

    container = ComposableNodeContainer(
        name='velodyne_container', namespace='', package='rclcpp_components',
        executable='component_container', output='both',
        composable_node_descriptions=[
            ComposableNode(
                package='velodyne_driver', plugin='velodyne_driver::VelodyneDriver',
                name='velodyne_driver_node', parameters=[driver_params]),
            ComposableNode(
                package='velodyne_pointcloud', plugin='velodyne_pointcloud::Transform',
                name='velodyne_transform_node', parameters=[pointcloud_params]),
            ComposableNode(
                package='velodyne_laserscan', plugin='velodyne_laserscan::VelodyneLaserScan',
                name='velodyne_laserscan_node', parameters=[laserscan_params]),
        ],
    )

    return [container]


def generate_launch_description():
    params_root = os.environ.get(
        'ROBOTBASE_PARAMS_DIR',
        os.path.join(os.path.expanduser('~'), 'robotbase_ws', 'params'))
    return LaunchDescription([
        DeclareLaunchArgument('tf_prefix', default_value='robot'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(params_root, 'real', 'velodyne.yaml')),
        DeclareLaunchArgument(
            'calibration_file',
            default_value=os.path.join(params_root, 'real', 'VLP16db.yaml')),
        OpaqueFunction(function=_launch_setup),
    ])
