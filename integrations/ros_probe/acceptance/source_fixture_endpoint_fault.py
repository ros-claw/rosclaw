"""Owned SIM fault: create one extra command endpoint, publish zero messages."""

import argparse
import json
import os
import time
from pathlib import Path

import rclpy
from gazebo_source_witness.prepare import check_prepared_assets
from geometry_msgs.msg import Twist
from rclpy.node import Node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated operator-owned disposable SIM required")
    check_prepared_assets("/assets", args.inventory_sha256, "/tmp/source-endpoint-fault-assets")
    initialization = json.loads(Path("/assets/source-initialization.json").read_text())
    if (
        initialization["run_id"] != args.run_id
        or initialization["operator_owned_disposable_sim_only"] is not True
    ):
        raise ValueError("explicit source fixture identity required")
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_explicit_duplicate_endpoint_fault")
    publisher = node.create_publisher(Twist, "/nav_cmd_vel", 10)
    try:
        print(
            json.dumps({"fault": "extra_publisher_endpoint_only", "velocity_messages_sent": 0}),
            flush=True,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_publisher(publisher)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
