"""ROS-host acceptance against synthetic DDS observations, never motion commands.

Run directly using ROS's Python in an isolated ROS_DOMAIN_ID.
This is native-probe evidence, not Nav2/Gazebo task acceptance.
"""

import argparse
import json
import time
from pathlib import Path

import rclpy
from example_interfaces.action import Fibonacci
from geometry_msgs.msg import TransformStamped
from probe import ReadOnlyProbe
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    sensor = Node("synthetic_sensor", namespace="/fixture")
    planner = LifecycleNode("planner_server", namespace="/fixture")
    planner.trigger_configure()
    probe = ReadOnlyProbe()
    executor = SingleThreadedExecutor()
    for node in (sensor, planner, probe):
        executor.add_node(node)
    scan_pub = sensor.create_publisher(
        LaserScan, "/fixture/scan", QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    )
    incompatible_sub = sensor.create_subscription(
        LaserScan,
        "/fixture/scan",
        lambda _msg: None,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
    )
    tf_pub = sensor.create_publisher(TFMessage, "/tf", 10)
    action_server = ActionServer(
        sensor, Fibonacci, "/fixture/fibonacci", execute_callback=lambda _goal: Fibonacci.Result()
    )

    def publish_fixture():
        scan = LaserScan()
        scan.header.stamp = sensor.get_clock().now().to_msg()
        scan.header.frame_id = "laser"
        scan.range_min, scan.range_max = 0.1, 10.0
        scan.ranges = [1.0] * 10
        scan_pub.publish(scan)
        transform = TransformStamped()
        transform.header.stamp = scan.header.stamp
        transform.header.frame_id, transform.child_frame_id = "base_link", "laser"
        transform.transform.rotation.w = 1.0
        tf_pub.publish(TFMessage(transforms=[transform]))

    timer = sensor.create_timer(0.05, publish_fixture)
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            probe.discover_reads()
            executor.spin_once(timeout_sec=0.05)
        snapshot = probe.snapshot()
        signal = next(s for s in snapshot["signals"] if s["topic"] == "/fixture/scan")
        assert 10 <= signal["rate_hz"] <= 30, signal
        assert signal["last_message_age_ms"] < 200, signal
        assert any(e["child"] == "laser" for e in snapshot["transforms"])
        assert any(e["state"] == "INACTIVE" for e in snapshot["lifecycle"]), snapshot["lifecycle"]
        assert any(
            e["name"] == "/fixture/fibonacci"
            and e["action_type"] == "example_interfaces/action/Fibonacci"
            for e in snapshot["graph"]["actions"]
        )
        assert any(e["topic"] == "/fixture/scan" for e in snapshot["qos"]["incompatible_pairs"])
        assert not snapshot["errors"], snapshot["errors"]
        # Explicit synthetic clock inputs test interpretation only. Keep these
        # separate from the independently captured live DDS snapshot above.
        saved_parameters, saved_edges = probe.parameters, probe.edges
        saved_clock = list(probe.clock_values)
        try:
            probe.parameters = {
                "/fixture/sim_node": {"use_sim_time": True},
                probe.get_fully_qualified_name(): {"use_sim_time": False},
            }
            probe.clock_values.clear()
            probe.clock_values.append(0.5)
            probe.edges = {
                ("map", "odom"): {
                    "parent": "map",
                    "child": "odom",
                    "static": False,
                    "source": "synthetic_clock_input",
                    "stamp_sec": 0.4,
                }
            }
            assert abs(probe.snapshot()["transforms"][0]["age_ms"] - 100) < 0.001
            probe.parameters["/fixture/wall_node"] = {"use_sim_time": False}
            assert probe.snapshot()["transforms"][0]["age_ms"] is None
        finally:
            probe.parameters, probe.edges = saved_parameters, saved_edges
            probe.clock_values.clear()
            probe.clock_values.extend(saved_clock)
        args.output.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "evidence_domain": "synthetic_ros_graph",
                    "nav2_acceptance": "NOT_RUN",
                    "snapshot": snapshot,
                    "clock_selection_checks": {
                        "status": "PASS",
                        "evidence_domain": "synthetic_clock_inputs",
                        "simulated_tf_age_ms": 100,
                        "mixed_clock_tf_age": None,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print("Native read-only probe: measured rate, TF, lifecycle and QoS PASS")
    finally:
        action_server.destroy()
        sensor.destroy_timer(timer)
        sensor.destroy_subscription(incompatible_sub)
        executor.shutdown()
        for node in (probe, planner, sensor):
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
