"""Launch a planning-only Nav2 server isolated from robot control."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


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
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='ar_reference'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file', default_value=f'{share}/config/reference_planner.yaml'
        ),
        DeclareLaunchArgument('map_topic', default_value='/map'),
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
        planner,
        lifecycle,
        requester,
    ])
