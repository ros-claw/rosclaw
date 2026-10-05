"""Independent passive Gazebo pose stream for observer-process failure tests."""

import argparse
import json
import math
import time
from datetime import UTC, datetime

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Node("independent_stop_observer")
    with open(args.output, "w", buffering=1) as trace:

        def observe(message):
            for transform in message.transforms:
                if transform.child_frame_id != "turtlebot3_waffle":
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

        node.create_subscription(TFMessage, "/rosclaw_sim/ground_truth", observe, 20)
        until = time.monotonic() + 90
        try:
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
