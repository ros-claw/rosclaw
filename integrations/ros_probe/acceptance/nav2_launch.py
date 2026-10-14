"""Existing Nav2 servers with the final velocity output behind a leased sink."""

import json
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = "/evidence/nav2.yaml"
    descriptor_path = Path("/evidence/bringup.json")
    descriptor = json.loads(descriptor_path.read_text()) if descriptor_path.exists() else None
    namespace = descriptor["namespace"] if descriptor else ""
    servers = [
        ("nav2_map_server", "map_server", "map_server", []),
        ("nav2_amcl", "amcl", "amcl", []),
        ("nav2_controller", "controller_server", "controller_server", [("cmd_vel", "cmd_vel_nav")]),
        ("nav2_smoother", "smoother_server", "smoother_server", []),
        ("nav2_planner", "planner_server", "planner_server", []),
        ("nav2_behaviors", "behavior_server", "behavior_server", [("cmd_vel", "cmd_vel_nav")]),
        ("nav2_bt_navigator", "bt_navigator", "bt_navigator", []),
        (
            "nav2_velocity_smoother",
            "velocity_smoother",
            "velocity_smoother",
            [("cmd_vel", "cmd_vel_nav")],
        ),
        (
            "nav2_collision_monitor",
            "collision_monitor",
            "collision_monitor",
            [("cmd_vel", "/nav_cmd_vel")],
        ),
    ]
    nodes = [
        Node(
            package=package,
            executable=executable,
            name=name,
            namespace=namespace,
            output="screen",
            parameters=[params, {"use_sim_time": True, "yaml_filename": "/evidence/map.yaml"}],
            remappings=(
                [("/tf", "tf"), ("/tf_static", "tf_static")]
                if descriptor and descriptor["tf_topics_relative"]
                else []
            )
            + [
                (source, namespace + target if namespace and target.startswith("/") else target)
                for source, target in remaps
            ],
        )
        for package, executable, name, remaps in servers
    ]
    nodes.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "autostart": descriptor["autostart"] if descriptor else True,
                    "node_names": [row[2] for row in servers],
                }
            ],
        )
    )
    return LaunchDescription(nodes)
