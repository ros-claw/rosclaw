"""Read-only SIM navigation evidence review; never admits a Body or action."""

import math
from datetime import UTC, datetime

from rosclaw.connectors.ros.context.discovery import discover_body_candidate
from rosclaw.connectors.ros.resolver.semantics import is_initial_pose_command
from rosclaw.contracts.common import content_hash


def review_navigation_candidate(model, urdf_bytes, requirements, *, now=None):
    """Review explicitly selected interfaces against sealed, original observations.

    Requirements describe an operator's intended topology, not trusted evidence,
    a fixture approval, a permit, limits or a runtime configuration. PROPOSED
    only means the observations below are sufficient for later binding review.
    """
    now = now or datetime.now(UTC)
    keys = {
        "interface_names",
        "lifecycle_nodes",
        "costmap_topics",
        "localization_topic",
        "velocity_input_topic",
        "velocity_output_topic",
        "velocity_consumer_node",
    }
    roles = {"localization", "planner", "controller", "collision_monitor", "navigator"}
    if type(requirements) is not dict or set(requirements) != keys:
        raise ValueError("navigation requirements must contain exactly the documented fields")
    nodes = requirements["lifecycle_nodes"]
    costmaps = requirements["costmap_topics"]
    if (
        type(nodes) is not dict
        or set(nodes) != roles
        or type(costmaps) is not list
        or len(costmaps) != 2
        or any(
            type(n) is not str or not n.startswith("/")
            for n in [
                *nodes.values(),
                *costmaps,
                requirements["localization_topic"],
                requirements["velocity_input_topic"],
                requirements["velocity_output_topic"],
                requirements["velocity_consumer_node"],
            ]
        )
        or len(set(nodes.values())) != 5
        or len(set(costmaps)) != 2
        or requirements["velocity_input_topic"] == requirements["velocity_output_topic"]
        or requirements["velocity_consumer_node"] in nodes.values()
    ):
        raise ValueError(
            "explicit distinct fully qualified navigation roles and interfaces required"
        )
    candidate = discover_body_candidate(
        model, urdf_bytes, now=now, interface_names=requirements["interface_names"]
    )
    result = {
        "schema_version": "rosclaw.navigation_observation_review.v1",
        "status": "UNKNOWN",
        "candidate": candidate,
        "source_snapshot_hash": model.snapshot_hash,
        "requirements_hash": content_hash("navrequirements", requirements),
        "reviewed_at": now.isoformat(),
        "unknown_fields": list(candidate["unknown_fields"]),
        "authorization": False,
        "capabilities_granted": [],
        "binding_verified": False,
        "usable_for_real_execution": False,
        "physical_acceptance_level": "NOT_RUN",
        "next_gate": "operator-reviewed immutable Body binding, daemon admission and independent guarded SIM navigation receipt",
        "unverified_checks": [
            "collision_polygon_geometry",
            "obstacle_response",
            "speed_and_acceleration_limits",
            "immutable_Body_binding",
            "daemon_admission",
            "physical_navigation",
        ],
    }
    unknown = result["unknown_fields"]
    if model.completeness.get("signals") is not True:
        unknown.append("complete_signal_observation")

    def fresh(stamp):
        try:
            return -100 <= (now - datetime.fromisoformat(stamp)).total_seconds() * 1000 <= 5000
        except (TypeError, ValueError):
            return False

    def finite_between(value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high

    def stream(name, allowed_types, *, messages=True):
        topics = [t for t in model.graph.get("topics", []) if t.get("name") == name]
        signals = [s for s in model.signals if s.topic == name]
        if len(topics) != 1 or topics[0].get("msg_type") not in allowed_types:
            unknown.append("unique_typed_stream:" + name)
            return None
        if not messages:
            # Graph capture already passed discovery's age/completeness gate.
            # Do not subscribe to commands and alter the topology under review.
            return topics[0]
        if len(signals) != 1:
            unknown.append("unique_typed_stream:" + name)
            return None
        signal = signals[0]
        if (
            signal.publisher_count != 1
            or len(topics[0].get("publishers", [])) != 1
            or not fresh(signal.captured_at.isoformat())
            or (
                messages
                and (
                    signal.last_message_age_ms is None
                    or signal.last_message_age_ms > min(signal.max_age_ms, 1000)
                )
            )
        ):
            unknown.append("fresh_unique_stream:" + name)
        if messages:
            header = model.observations.get("message_frames", {}).get(name, {})
            frames = candidate["frames"]
            if (
                header.get("source") != name
                or not fresh(header.get("captured_at"))
                or not finite_between(header.get("stamp_age_ms"), -100, 1000)
                or header.get("frame_id") not in {frames.get("map"), frames.get("odom")}
                or not header.get("frame_id")
            ):
                unknown.append("fresh_navigation_frame:" + name)
        return topics[0]

    clock_age = model.observations.get("clock_last_receive_age_ms")
    if (
        model.environment.get("ros_generation") != "ros2"
        or model.environment.get("use_sim_time") is not True
        or model.observations.get("clock_advancing") is not True
        or not finite_between(clock_age, 0, 1000)
        or model.completeness.get("time") is not True
    ):
        unknown.append("observed_advancing_ROS2_SIM_clock")
    if model.completeness.get("lifecycle") is not True:
        unknown.append("complete_lifecycle_observation")
    for role, node in nodes.items():
        rows = [r for r in model.lifecycle if r.name == node]
        if (
            len(rows) != 1
            or rows[0].state != "ACTIVE"
            or rows[0].source != node + "/get_state"
            or not fresh(rows[0].captured_at.isoformat())
        ):
            unknown.append("lifecycle." + role)
    localization = requirements["localization_topic"]
    topic = stream(localization, {"geometry_msgs/msg/PoseWithCovarianceStamped"})
    quality = [
        r
        for r in model.observations.get("localization_quality", [])
        if r.get("source") == localization
    ]
    if (
        topic is None
        or is_initial_pose_command(topic)
        or topic.get("publishers") != [nodes["localization"]]
        or len(quality) != 1
        or not fresh(quality[0].get("captured_at"))
        or any(
            not finite_between(quality[0].get(k), 0, 0.25)
            for k in ("x_variance", "y_variance", "yaw_variance")
        )
        or model.observations.get("message_frames", {}).get(localization, {}).get("frame_id")
        != candidate["frames"].get("map")
    ):
        unknown.append("fresh_localization_estimate_and_covariance")
    for name in costmaps:
        stream(name, {"nav_msgs/msg/OccupancyGrid"})
    monitor = nodes["collision_monitor"]
    params = model.observations.get("node_parameters", {}).get(monitor, {})
    stamp = model.observations.get("parameter_captured_at", {}).get(monitor)
    sources = params.get("observation_sources")
    polygons = params.get("polygons")
    lidar = candidate["interfaces"].get("sensing.lidar", [])
    # Relative parameter values require ROS-host expansion/remapping evidence.
    # No namespace concatenation or conventional sensor instance is inferred.
    if (
        not fresh(stamp)
        or params.get("use_sim_time") is not True
        or params.get("enabled") is False
        or type(sources) is not list
        or not sources
        or any(type(s) is not str or not s for s in sources)
        or len(lidar) != 1
        or not any(params.get(s + ".topic") == lidar[0]["name"] for s in sources)
        or params.get("cmd_vel_in_topic") != requirements["velocity_input_topic"]
        or params.get("cmd_vel_out_topic") != requirements["velocity_output_topic"]
        or type(polygons) is not list
        or not polygons
        or any(type(p) is not str or not p for p in polygons)
        or any(params.get(p + ".enabled") is False for p in polygons)
        or any(params.get(s + ".enabled") is False for s in sources)
    ):
        unknown.append("fresh_explicit_collision_monitor_configuration")
    controls = []
    for name, publisher, subscriber in (
        (requirements["velocity_input_topic"], nodes["controller"], monitor),
        (requirements["velocity_output_topic"], monitor, requirements["velocity_consumer_node"]),
    ):
        topic = stream(
            name, {"geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped"}, messages=False
        )
        if (
            topic is None
            or topic.get("publishers") != [publisher]
            or topic.get("subscribers") != [subscriber]
        ):
            unknown.append("observed_exclusive_velocity_chain:" + name)
        if topic:
            controls.append(topic["msg_type"])
    if len(controls) != 2 or len(set(controls)) != 1:
        unknown.append("consistent_velocity_message_type")
    if model.qos.get("incompatible_pairs"):
        unknown.append("observed_QoS_incompatibility")
    if model.completeness.get("qos") is not True:
        unknown.append("complete_QoS_observation")
    result["unknown_fields"] = sorted(set(unknown))
    if not result["unknown_fields"]:
        result["status"] = "PROPOSED"
    return result
