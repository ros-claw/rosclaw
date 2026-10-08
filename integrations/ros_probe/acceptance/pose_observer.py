"""Independent passive Gazebo pose stream for observer-process failure tests."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import rclpy
from profiles import PROFILES
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=90)
    args = parser.parse_args()
    if not 60 <= args.duration <= 1920:
        parser.error("owned passive observation duration must be between 60 and 1920 seconds")
    saved = json.loads(Path("/evidence/fixture_profile.json").read_text())
    profile = PROFILES[saved["name"]]
    if saved != profile.to_dict():
        raise ValueError("independent observer profile differs from supported geometry")
    rclpy.init()
    node = Node("independent_stop_observer")
    with open(args.output, "w", buffering=1) as trace:

        def observe(message):
            for transform in message.transforms:
                if transform.child_frame_id != profile.simulation_model:
                    continue
                q = transform.transform.rotation
                trace.write(
                    json.dumps(
                        {
                            "x": transform.transform.translation.x,
                            "y": transform.transform.translation.y,
                            "yaw": math.atan2(
                                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
                            ),
                            "captured_at": datetime.now(UTC).isoformat(),
                            "time_sec": transform.header.stamp.sec
                            + transform.header.stamp.nanosec / 1e9,
                            "source": "independent_gazebo_ground_truth_subscription",
                        }
                    )
                    + "\n"
                )

        node.create_subscription(
            TFMessage,
            "/rosclaw_sim/ground_truth",
            observe,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        until = time.monotonic() + args.duration
        try:
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
