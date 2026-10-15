"""SDK-owned extra command endpoint for engineering diagnosis; publishes nothing."""

import json
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def main():
    settings = json.loads(Path("/settings.json").read_text())
    policy = settings.get("engineering_capture", {})
    sources = json.loads(Path("/assets/observer-sources.json").read_text())
    descriptor = json.loads(Path("/evidence/bringup.json").read_text())
    endpoint = descriptor["additional_command_endpoint"]
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or policy.get("case") != "duplicate_velocity_route"
        or policy.get("phase") not in {"before_repair", "rollback"}
        or policy.get("run_id") != sources["run_id"]
        or set(endpoint) != {"enabled", "topic", "msg_type"}
        or endpoint["enabled"] is not True
        or endpoint["topic"] != sources["topics"]["velocity_input"]
        or endpoint["msg_type"] != "geometry_msgs/msg/Twist"
    ):
        raise ValueError("closed SDK declared duplicate protected endpoint required")
    rclpy.init()
    namespace = sources["observer_node"].rsplit("/", 1)[0]
    node = Node("rosclaw_engineering_extra_endpoint", namespace=namespace)
    publisher = None
    try:
        deadline = time.monotonic() + 15
        while True:
            observed = node.get_publishers_info_by_topic(endpoint["topic"])
            if len(observed) == 1 and observed[0].topic_type == endpoint["msg_type"]:
                break
            if time.monotonic() >= deadline:
                raise ValueError("one original typed protected velocity publisher required")
            rclpy.spin_once(node, timeout_sec=0.05)
        publisher = node.create_publisher(Twist, endpoint["topic"], 10)
        Path("/evidence/engineering-extra-endpoint-original.json").write_text(
            json.dumps(
                {
                    "run_id": sources["run_id"],
                    "namespace": namespace,
                    "endpoint": endpoint,
                    "owner_pid": os.getpid(),
                    "monotonic_ns": time.monotonic_ns(),
                    "velocity_messages_published": 0,
                    "navigation_goals_sent": 0,
                },
                indent=2,
            )
            + "\n"
        )
        rclpy.spin(node)
    finally:
        if publisher is not None:
            node.destroy_publisher(publisher)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
