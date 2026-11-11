"""
Start the Viewer and auto-selected display-only reference Planner.

No map or waypoint launch argument is exposed: Launcher2 and the running
Follower remain their respective sources of truth.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _launch_viewer(context):
    """Build Viewer parameters, adding a review overlay only when requested."""
    parameters = [LaunchConfiguration('viewer_config_file').perform(context)]
    calibration_overlay = LaunchConfiguration(
        'viewer_calibration_overlay_file').perform(context).strip()
    if calibration_overlay:
        parameters.append(calibration_overlay)
    # The checked-in base YAML remains false. Only this explicit test launch
    # enables reference rendering after all Viewer gates pass.
    parameters.append({'reference_path_enabled': True})
    return [Node(
        package='ar_path_viewer',
        executable='ar_path_viewer',
        name='ar_path_viewer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_viewer')),
        parameters=parameters,
    )]


def generate_launch_description():
    """Return the explicit reference-enabled Launcher2 integration launch."""
    viewer_share = get_package_share_directory('ar_path_viewer')
    default_viewer_config = (
        f'{viewer_share}/config/ar_path_viewer_c920.yaml'
    )
    supervisor = Node(
        package='ar_reference_planner',
        executable='reference_planner_auto_supervisor',
        name='reference_planner_auto_supervisor',
        output='screen',
        parameters=[{
            'launcher_map_state_file': LaunchConfiguration(
                'launcher_map_state_file'),
            'map_state_max_initial_age_sec': LaunchConfiguration(
                'map_state_max_initial_age_sec'),
            'follower_node_name': LaunchConfiguration('follower_node_name'),
            'follower_state_topic': '/waypoint_follower/reference_state',
            'reference_path_topic': '/ar/reference_path',
            'reference_path_state_topic': '/ar/reference_path_state',
            'reference_namespace': LaunchConfiguration('reference_namespace'),
            'robot_base_frame': LaunchConfiguration('robot_base_frame'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument('launcher_map_state_file', default_value=''),
        DeclareLaunchArgument(
            'map_state_max_initial_age_sec', default_value='86400.0'),
        DeclareLaunchArgument(
            'follower_node_name', default_value='/koko_waypoint_follower'),
        DeclareLaunchArgument('reference_namespace', default_value='ar_reference'),
        DeclareLaunchArgument(
            'robot_base_frame',
            default_value=[
                EnvironmentVariable(
                    'ROBOTBASE_TF_PREFIX', default_value='robot'),
                '/base_footprint',
            ],
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_viewer', default_value='true'),
        DeclareLaunchArgument(
            'viewer_config_file', default_value=default_viewer_config),
        DeclareLaunchArgument(
            'viewer_calibration_overlay_file', default_value=''),
        supervisor,
        OpaqueFunction(function=_launch_viewer),
    ])
