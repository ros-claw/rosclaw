"""Synthetic ROS SDK sources for observation review, never a Nav2 executor."""

import json
import time


def prepare_navigation_fixture(source, executor, namespace):
    from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
    from lifecycle_msgs.srv import GetState
    from nav_msgs.msg import OccupancyGrid
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock as ClockMessage

    source.set_parameters([Parameter("use_sim_time", value=True)])
    state = {"lifecycle": 3, "variance": 0.01}
    nodes = {}
    for role in (
        "localization",
        "planner",
        "controller",
        "collision_monitor",
        "navigator",
        "drive_adapter",
    ):
        node = Node(role, namespace=namespace)
        node.set_parameters([Parameter("use_sim_time", value=True)])
        executor.add_node(node)
        nodes[role] = node
        if role != "drive_adapter":

            def get_state(_request, response):
                response.current_state.id = state["lifecycle"]
                return response

            node.create_service(GetState, node.get_fully_qualified_name() + "/get_state", get_state)
    requirements = {
        "interface_names": {"mapping.occupancy_map": namespace + "/floor"},
        "lifecycle_nodes": {
            r: n.get_fully_qualified_name() for r, n in nodes.items() if r != "drive_adapter"
        },
        "costmap_topics": [namespace + "/near_grid", namespace + "/far_grid"],
        "localization_topic": namespace + "/estimated_pose",
        "velocity_input_topic": namespace + "/unfiltered",
        "velocity_output_topic": namespace + "/filtered",
        "velocity_consumer_node": nodes["drive_adapter"].get_fully_qualified_name(),
    }
    monitor = nodes["collision_monitor"]
    for key, value in {
        "observation_sources": ["range_source"],
        "range_source.topic": namespace + "/scan2",
        "polygons": ["stop_zone"],
        "cmd_vel_in_topic": requirements["velocity_input_topic"],
        "cmd_vel_out_topic": requirements["velocity_output_topic"],
    }.items():
        monitor.declare_parameter(key, value)
    # Both publishers exist, but this fixture never publishes a velocity.
    nodes["controller"].create_publisher(Twist, requirements["velocity_input_topic"], 10)
    monitor.create_publisher(Twist, requirements["velocity_output_topic"], 10)

    def forbidden_velocity(_message):
        raise AssertionError("observation fixture must never receive a velocity command")

    monitor.create_subscription(Twist, requirements["velocity_input_topic"], forbidden_velocity, 10)
    nodes["drive_adapter"].create_subscription(
        Twist, requirements["velocity_output_topic"], forbidden_velocity, 10
    )
    pose_pub = nodes["localization"].create_publisher(
        PoseWithCovarianceStamped, requirements["localization_topic"], 10
    )
    grids = [
        nodes["planner"].create_publisher(OccupancyGrid, n, 10)
        for n in requirements["costmap_topics"]
    ]
    clock_pub = source.create_publisher(ClockMessage, "/clock", 10)
    started = time.monotonic()

    def publish_observations():
        seconds = 10 + time.monotonic() - started
        clock = ClockMessage()
        clock.clock.sec = int(seconds)
        clock.clock.nanosec = int((seconds % 1) * 1e9)
        clock_pub.publish(clock)
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id, pose.header.stamp = "world", clock.clock
        pose.pose.pose.orientation.w = 1.0
        for i in (0, 7, 35):
            pose.pose.covariance[i] = state["variance"]
        pose_pub.publish(pose)
        grid = OccupancyGrid()
        grid.header.frame_id, grid.header.stamp = "world", clock.clock
        grid.info.resolution, grid.info.width, grid.info.height = 0.1, 4, 4
        grid.info.origin.orientation.w, grid.data = 1.0, [0] * 16
        for publisher in grids:
            publisher.publish(grid)

    source.create_timer(0.05, publish_observations, clock=Clock(clock_type=ClockType.STEADY_TIME))
    return nodes, requirements, state


def run_navigation_reviews(probe, output, urdf, requirements, state, spin):
    from rosclaw.connectors.ros.context.discovery import discover_body_candidate
    from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
    from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
    from rosclaw.connectors.ros.intelligence import build_system_model

    cases = []
    (output / "requirements.json").write_text(json.dumps(requirements, indent=2) + "\n")
    (output / "robot.urdf").write_bytes(urdf)
    for name, expected in (
        ("fresh_review", "PROPOSED"),
        ("inactive_lifecycle", "UNKNOWN"),
        ("bad_covariance", "UNKNOWN"),
    ):
        if name == "inactive_lifecycle":
            state["lifecycle"] = 2
        if name == "bad_covariance":
            state.update(lifecycle=3, variance=-0.01)
        spin(6)
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
        result = review_navigation_candidate(model, urdf, requirements)
        for suffix, data in (
            ("original-native", native),
            ("sealed-system", model.to_dict()),
            ("review", result),
        ):
            (output / f"{name}.{suffix}.json").write_text(json.dumps(data, indent=2) + "\n")
        assert result["status"] == expected, (name, result)
        if name == "fresh_review":
            assert discover_body_candidate(model, urdf)["status"] == "UNKNOWN", (
                "unselected map ambiguity must remain unknown"
            )
        assert not result["authorization"] and not result["binding_verified"]
        cases.append(
            {"case": name, "status": result["status"], "unknown_fields": result["unknown_fields"]}
        )
    report = {
        "status": "PASS_SYNTHETIC_SDK_NAVIGATION_OBSERVATIONS_ONLY",
        "cases": cases,
        "Worlds": 0,
        "navigation_goals_sent": 0,
        "velocity_messages_sent": 0,
        "Body_admitted": False,
        "physical_acceptance_level": "NOT_RUN",
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
