"""Launch a planning-only Nav2 server isolated from robot control."""

import atexit
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from ar_reference_planner.keepout_config import (
    params_with_keepout,
    resolve_keepout_selection,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile, ParameterValue
from nav2_common.launch import RewrittenYaml
import yaml


def _configure_selected_map(context):
    """Validate one selected map and derive Keepout parameters fail-closed."""
    map_yaml = LaunchConfiguration('map').perform(context).strip()
    if not map_yaml:
        raise RuntimeError(
            'map:=<selected map YAML> is required for reference planning'
        )
    selection = resolve_keepout_selection(map_yaml)
    params_file = LaunchConfiguration('params_file').perform(context)
    filter_info_topic = LaunchConfiguration(
        'keepout_filter_info_topic').perform(context)
    if selection.enabled:
        with open(params_file, encoding='utf-8') as source:
            params = yaml.safe_load(source)
        params = params_with_keepout(params, filter_info_topic)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', prefix='ar_reference_keepout_',
            delete=False, encoding='utf-8'
        ) as derived:
            yaml.safe_dump(params, derived, sort_keys=False)
            params_file = derived.name
        atexit.register(
            lambda path=params_file: os.path.exists(path) and os.unlink(path)
        )

    map_signature = selection.map_signature
    mask_signature = selection.mask_signature
    values = {
        'params_file': params_file,
        'keepout_required': str(selection.enabled).lower(),
        'expected_map_width': str(map_signature.width),
        'expected_map_height': str(map_signature.height),
        'expected_map_resolution': str(map_signature.resolution),
        'expected_map_origin_x': str(map_signature.origin_x),
        'expected_map_origin_y': str(map_signature.origin_y),
        'expected_map_origin_yaw': str(map_signature.origin_yaw),
        'expected_map_data_sha256': map_signature.data_sha256,
        'expected_mask_width': str(mask_signature.width if mask_signature else 0),
        'expected_mask_height': str(mask_signature.height if mask_signature else 0),
        'expected_mask_resolution': str(
            mask_signature.resolution if mask_signature else 0.0),
        'expected_mask_origin_x': str(
            mask_signature.origin_x if mask_signature else 0.0),
        'expected_mask_origin_y': str(
            mask_signature.origin_y if mask_signature else 0.0),
        'expected_mask_origin_yaw': str(
            mask_signature.origin_yaw if mask_signature else 0.0),
        'expected_mask_data_sha256': (
            mask_signature.data_sha256 if mask_signature else ''),
    }
    actions = [SetLaunchConfiguration(name, value) for name, value in values.items()]
    actions.append(LogInfo(msg=(
        'reference_selected_map_verified '
        f'map={map_signature.yaml_path} keepout={selection.enabled} '
        f'reason={selection.reason}'
    )))
    return actions


def generate_launch_description():
    """Return the namespaced planning-only launch description."""
    share = get_package_share_directory('ar_reference_planner')
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_topic = LaunchConfiguration('map_topic')
    robot_base_frame = LaunchConfiguration('robot_base_frame')

    configured = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites={
                'use_sim_time': use_sim_time,
                'map_topic': map_topic,
                'robot_base_frame': robot_base_frame,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        namespace=namespace,
        name='planner_server',
        output='screen',
        parameters=[configured],
        # Never allow the planning-only server to resolve its debug path as /plan.
        remappings=[
            ('plan', 'planner_internal_plan'),
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static'),
        ],
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        namespace=namespace,
        name='lifecycle_manager_reference_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['planner_server'],
        }],
    )
    requester = Node(
        package='ar_reference_planner',
        executable='reference_path_request',
        namespace=namespace,
        name='reference_path_request',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'waypoint_file': LaunchConfiguration('waypoint_file'),
            'target_length_m': LaunchConfiguration('target_length_m'),
            'minimum_length_m': LaunchConfiguration('minimum_length_m'),
            'maximum_length_m': LaunchConfiguration('maximum_length_m'),
            'refresh_period_sec': LaunchConfiguration('refresh_period_sec'),
            'result_timeout_sec': LaunchConfiguration('result_timeout_sec'),
            'validity_sec': LaunchConfiguration('validity_sec'),
            'max_follower_state_age_sec': LaunchConfiguration(
                'max_follower_state_age_sec'),
            'action_name': 'compute_path_through_poses',
            'reference_path_topic': '/ar/reference_path',
            'reference_path_state_topic': '/ar/reference_path_state',
            'follower_state_topic': '/waypoint_follower/reference_state',
            'map_topic': map_topic,
            'keepout_required': ParameterValue(
                LaunchConfiguration('keepout_required'), value_type=bool),
            'keepout_filter_info_topic': LaunchConfiguration(
                'keepout_filter_info_topic'),
            'keepout_mask_topic': LaunchConfiguration('keepout_mask_topic'),
            'keepout_mask_state_service': LaunchConfiguration(
                'keepout_mask_state_service'),
            'keepout_info_state_service': LaunchConfiguration(
                'keepout_info_state_service'),
            'expected_map_width': ParameterValue(
                LaunchConfiguration('expected_map_width'), value_type=int),
            'expected_map_height': ParameterValue(
                LaunchConfiguration('expected_map_height'), value_type=int),
            'expected_map_resolution': ParameterValue(
                LaunchConfiguration('expected_map_resolution'), value_type=float),
            'expected_map_origin_x': ParameterValue(
                LaunchConfiguration('expected_map_origin_x'), value_type=float),
            'expected_map_origin_y': ParameterValue(
                LaunchConfiguration('expected_map_origin_y'), value_type=float),
            'expected_map_origin_yaw': ParameterValue(
                LaunchConfiguration('expected_map_origin_yaw'), value_type=float),
            'expected_map_data_sha256': LaunchConfiguration(
                'expected_map_data_sha256'),
            'expected_mask_width': ParameterValue(
                LaunchConfiguration('expected_mask_width'), value_type=int),
            'expected_mask_height': ParameterValue(
                LaunchConfiguration('expected_mask_height'), value_type=int),
            'expected_mask_resolution': ParameterValue(
                LaunchConfiguration('expected_mask_resolution'), value_type=float),
            'expected_mask_origin_x': ParameterValue(
                LaunchConfiguration('expected_mask_origin_x'), value_type=float),
            'expected_mask_origin_y': ParameterValue(
                LaunchConfiguration('expected_mask_origin_y'), value_type=float),
            'expected_mask_origin_yaw': ParameterValue(
                LaunchConfiguration('expected_mask_origin_yaw'), value_type=float),
            'expected_mask_data_sha256': LaunchConfiguration(
                'expected_mask_data_sha256'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='ar_reference'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file', default_value=f'{share}/config/reference_planner.yaml'
        ),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument(
            'keepout_filter_info_topic', default_value='/costmap_filter_info'),
        DeclareLaunchArgument(
            'keepout_mask_topic', default_value='/keepout_filter_mask'),
        DeclareLaunchArgument(
            'keepout_mask_state_service',
            default_value='/filter_mask_server/get_state'),
        DeclareLaunchArgument(
            'keepout_info_state_service',
            default_value='/costmap_filter_info_server/get_state'),
        DeclareLaunchArgument(
            'robot_base_frame', default_value='robot/base_footprint'
        ),
        DeclareLaunchArgument('waypoint_file', default_value=''),
        DeclareLaunchArgument('target_length_m', default_value='11.0'),
        DeclareLaunchArgument('minimum_length_m', default_value='8.0'),
        DeclareLaunchArgument('maximum_length_m', default_value='15.0'),
        DeclareLaunchArgument('refresh_period_sec', default_value='3.0'),
        DeclareLaunchArgument('result_timeout_sec', default_value='3.0'),
        DeclareLaunchArgument('validity_sec', default_value='6.5'),
        DeclareLaunchArgument('max_follower_state_age_sec', default_value='2.5'),
        OpaqueFunction(function=_configure_selected_map),
        planner,
        lifecycle,
        requester,
    ])
