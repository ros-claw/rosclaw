"""One-shot operator SIM setup, never an Agent tool or source observer.

Consumes the independently pinned read-only known-fixture initialization input.
No action client, velocity publisher, Runtime or daemon is instantiated.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import rclpy
from gazebo_source_witness.prepare import check_prepared_assets
from nav2_msgs.srv import SetInitialPose
from rclpy.node import Node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-owned-source-fixture", action="store_true", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    # Container-only fixture paths and explicit pins prevent implicit targeting.
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated disposable SIM fixture required")
    check_prepared_assets("/assets", args.inventory_sha256, "/tmp/source-initializer-assets")
    config = json.loads(Path("/assets/source-initialization.json").read_text())
    if (
        config["schema_version"] != "rosclaw.owned_sim_source_initialization.v1"
        or config["run_id"] != args.run_id
        or config["operator_owned_disposable_sim_only"] is not True
        or config["service"] != "/set_initial_pose"
        or config["frame_id"] != "map"
    ):
        raise ValueError("explicit known SIM initialization identity required")
    pose, covariance = config["pose"], config["covariance_diagonal"]
    if (
        len(covariance) != 3
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 0.01
            for x in covariance
        )
        or any(
            type(pose[k]) not in (int, float) or not math.isfinite(pose[k])
            for k in ("x", "y", "z", "yaw")
        )
    ):
        raise ValueError("finite bounded operator pose and nondegenerate covariance required")
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_source_fixture_initializer")
    client = node.create_client(SetInitialPose, config["service"])
    try:
        if not client.wait_for_service(timeout_sec=8):
            raise RuntimeError("operator initialization service unavailable")
        deadline = time.monotonic() + 5
        while node.get_clock().now().nanoseconds <= 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("actual simulator clock missing for initialization")
            rclpy.spin_once(node, timeout_sec=0.05)
        request = SetInitialPose.Request()
        request.pose.header.frame_id = config["frame_id"]
        request.pose.header.stamp = node.get_clock().now().to_msg()
        request.pose.pose.pose.position.x = float(pose["x"])
        request.pose.pose.pose.position.y = float(pose["y"])
        request.pose.pose.pose.position.z = float(pose["z"])
        request.pose.pose.pose.orientation.z = math.sin(pose["yaw"] / 2)
        request.pose.pose.pose.orientation.w = math.cos(pose["yaw"] / 2)
        for index, value in zip((0, 7, 35), covariance, strict=True):
            request.pose.pose.covariance[index] = float(value)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=8)
        if not future.done() or future.result() is None:
            raise RuntimeError("operator initialization response missing")
        print(
            json.dumps(
                {
                    "status": "OPERATOR_SIM_INITIALIZATION_RESPONSE_ONLY_NOT_ACCEPTANCE",
                    "input": config,
                    "action_goals_sent": 0,
                    "velocity_commands_sent": 0,
                }
            ),
            flush=True,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
