from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    default_config = Path(get_package_share_directory("ar_path_viewer")) / "config" / "ar_path_viewer.yaml"
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description="Viewer ROS parameter YAML; no source edit is required",
            ),
            Node(
                package="ar_path_viewer",
                executable="ar_path_viewer",
                name="ar_path_viewer",
                output="screen",
                parameters=[LaunchConfiguration("config_file")],
            )
        ]
    )
