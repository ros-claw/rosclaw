"""Isolated SDK synthetic discovery contract; no World or motion endpoint.

Run only in an owned, network-isolated SDK container. All sensor/TF/description
sources are synthetic. A typed navigation action server rejects every goal;
this contract never sends one. Results cannot establish physical L1/L2 binding.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--base-frame", required=True)
    parser.add_argument("--sensor-frame", required=True)
    parser.add_argument("--body-width", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid, Odometry
    from probe import ReadOnlyProbe
    from rclpy.action import ActionServer, GoalResponse
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage

    from rosclaw.connectors.ros.context.discovery import discover_body_candidate
    from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
    from rosclaw.connectors.ros.intelligence import build_system_model

    namespace = args.namespace
    urdf = (
        f'<robot name="synthetic_discovery_contract"><link name="{args.base_frame}">'
        f'<collision><geometry><box size="{args.body_width} 0.2 0.1"/></geometry>'
        f'</collision></link><link name="{args.sensor_frame}"/>'
        f'<joint name="mount" type="fixed"><parent link="{args.base_frame}"/>'
        f'<child link="{args.sensor_frame}"/></joint></robot>'
    ).encode()
    rclpy.init()
    source = Node("description_source", namespace=namespace)
    source.declare_parameter("robot_description", urdf.decode())
    streaming = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    scan_pub = source.create_publisher(LaserScan, namespace + "/scan2", streaming)
    odom_pub = source.create_publisher(Odometry, namespace + "/position_feedback", streaming)
    map_pub = source.create_publisher(OccupancyGrid, namespace + "/floor", latched)
    tf_pub = source.create_publisher(TFMessage, "/tf", streaming)

    def forbidden_execute(_goal):
        raise AssertionError("synthetic discovery contract must never execute navigation")

    action = ActionServer(
        source,
        NavigateToPose,
        namespace + "/go",
        forbidden_execute,
        goal_callback=lambda _: GoalResponse.REJECT,
    )

    def publish():
        stamp = source.get_clock().now().to_msg()
        scan = LaserScan()
        scan.header.frame_id = args.sensor_frame
        scan.header.stamp = stamp
        scan.angle_min, scan.angle_max, scan.angle_increment = -1.0, 1.0, 1.0
        scan.range_min, scan.range_max, scan.ranges = 0.1, 10.0, [2.0, 2.0, 2.0]
        scan_pub.publish(scan)
        if odom_pub is not None:
            odom = Odometry()
            odom.header.frame_id, odom.child_frame_id = "local", args.base_frame
            odom.header.stamp = stamp
            odom.pose.pose.orientation.w = 1.0
            odom_pub.publish(odom)
        grid = OccupancyGrid()
        grid.header.frame_id, grid.header.stamp = "world", stamp
        grid.info.resolution, grid.info.width, grid.info.height = 0.1, 4, 4
        grid.info.origin.orientation.w, grid.data = 1.0, [0] * 16
        map_pub.publish(grid)
        edges = []
        for parent, child in [
            ("world", "local"),
            ("local", args.base_frame),
            (args.base_frame, args.sensor_frame),
        ]:
            edge = TransformStamped()
            edge.header.frame_id, edge.child_frame_id, edge.header.stamp = parent, child, stamp
            edge.transform.rotation.w = 1.0
            edges.append(edge)
        tf_pub.publish(TFMessage(transforms=edges))

    source.create_timer(0.05, publish)
    probe = ReadOnlyProbe()
    executor = SingleThreadedExecutor()
    executor.add_node(source)
    executor.add_node(probe)
    cases = []

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=min(0.02, max(0.0, until - time.monotonic())))

    def capture(name, *, raw_urdf=urdf, expected="UNKNOWN", gap=None):
        native = probe.snapshot()
        graph = RosGraphSnapshot.from_dict(
            {
                "ros_version": "ros2",
                "distro": "jazzy",
                "endpoint": "fixture://isolated-sdk",
                "captured_at": native["captured_at"],
                **native["graph"],
            }
        )
        model = build_system_model(graph, native=native)
        result = discover_body_candidate(model, raw_urdf)
        (args.output / f"{name}.original-native.json").write_text(
            json.dumps(native, indent=2) + "\n"
        )
        (args.output / f"{name}.sealed-system.json").write_text(
            json.dumps(model.to_dict(), indent=2) + "\n"
        )
        (args.output / f"{name}.candidate.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["status"] == expected, (name, result)
        if gap:
            assert gap in result["unknown_fields"], (name, result)
        assert not result["authorization"] and not result["binding_verified"]
        assert (
            result["capabilities_granted"] == []
            and result["physical_acceptance_level"] == "NOT_RUN"
        )
        cases.append(
            {"case": name, "status": result["status"], "unknown_fields": result["unknown_fields"]}
        )
        return model

    try:
        spin(5)
        positive = capture("unique_fresh_streams", expected="PROPOSED")
        (args.output / "robot.urdf").write_bytes(urdf)
        capture(
            "mismatched_urdf", raw_urdf=urdf + b" ", gap="urdf.unique_matching_live_description"
        )
        extra = source.create_publisher(LaserScan, namespace + "/scan2", streaming)
        spin(1)
        capture("multiple_lidar_publishers", gap="sensing.lidar.unique_fresh_typed_stream")
        source.destroy_publisher(extra)
        source.destroy_publisher(odom_pub)
        odom_pub = None
        spin(1.5)
        capture("odometry_stopped", gap="state.odometry.unique_fresh_typed_stream")
        report = {
            "status": "PASS_ACTUAL_ROS_SDK_SYNTHETIC_DISCOVERY_NOT_UNSEEN_BODY_OR_PHYSICS",
            "namespace": namespace,
            "base_frame": args.base_frame,
            "sensor_frame": args.sensor_frame,
            "body_width": args.body_width,
            "cases": cases,
            "actual_positive_snapshot_hash": positive.snapshot_hash,
            "World_started": False,
            "navigation_goals_sent": 0,
            "Body_admitted": False,
            "physical_acceptance_level": "NOT_RUN",
        }
        (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        executor.shutdown()
        action.destroy()
        probe.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
