from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
)
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    # Overwrites config search based on robot name
    config_path = LaunchConfiguration("config").perform(context)

    robot = LaunchConfiguration("robot").perform(context)

    if config_path:  # non-empty string means it was provided
        param_file = config_path
    else:
        pkg_share = get_package_share_directory("hector_detection_aggregator")
        param_file = PathJoinSubstitution([pkg_share, "config", [robot, ".yaml"]])

    node_args = {
        "package": "hector_detection_aggregator",
        "executable": "detection_aggregator_node",
        "name": "detection_aggregator_node",
        "output": "screen",
        "parameters": [param_file],
    }

    return [Node(**node_args)]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="athena"),
            DeclareLaunchArgument("config", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
