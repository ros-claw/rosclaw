"""Evidence-bound simulator fixture proposals; no host or robot writes.

Fixture policy is an explicit simulator-operator input, not an Agent permit.
This computes a reviewed configuration candidate; it never installs, dispatches
or grants REAL authority. Actual compiled binding and L2/L3 require later gates.
"""

import math
from copy import deepcopy
from datetime import UTC, datetime

from rosclaw.connectors.ros.context.discovery import discover_body_candidate
from rosclaw.connectors.ros.context.sim_attachment import validate_sim_attachment
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.intelligence.topic_parameters import observed_topic_parameter
from rosclaw.connectors.ros.resolver.semantics import is_initial_pose_command


def propose_sim_fixture_binding(model, urdf_bytes, *, attachment, policy, now=None):
    now = now or datetime.now(UTC)
    candidate = discover_body_candidate(model, urdf_bytes, now=now)
    cleaner = validate_sim_attachment(attachment)
    result = {
        "schema_version": "rosclaw.sim_fixture_binding_proposal.v1",
        "status": "UNKNOWN",
        "candidate": candidate,
        "attachment": cleaner,
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": candidate["source_urdf_sha256"],
        "fixture_policy_hash": digest(policy),
        "unknown_fields": [],
        "capabilities_granted": [],
        "binding_verified": False,
        "usable_for_real_execution": False,
        "physical_acceptance_level": "NOT_RUN",
    }
    if candidate["status"] != "PROPOSED":
        result["unknown_fields"] = list(candidate["unknown_fields"])
        return result
    if (
        policy.get("schema_version") != "rosclaw.sim_fixture_body_policy.v1"
        or policy.get("source") != "simulator_operator_fixture_policy"
        or policy.get("evidence_domain") != "SIMULATION"
        or policy.get("approved") is not True
        or type(policy.get("policy_id")) is not str
        or not policy["policy_id"]
        or type(policy.get("body_id")) is not str
        or not policy["body_id"]
        or policy.get("source_snapshot_hash") != model.snapshot_hash
        or policy.get("source_urdf_sha256") != candidate["source_urdf_sha256"]
        or policy.get("attachment_hash") != cleaner["attachment_hash"]
    ):
        result["unknown_fields"].append("explicit_matching_SIM_fixture_policy")
        return result

    def fresh(stamp):
        try:
            return -100 <= (now - datetime.fromisoformat(stamp)).total_seconds() * 1000 <= 5000
        except (TypeError, ValueError):
            return False

    clock_age = model.observations.get("clock_last_receive_age_ms")
    if (
        type(clock_age) not in (int, float)
        or not math.isfinite(clock_age)
        or not 0 <= clock_age < 1000
        or model.environment.get("ros_generation") != "ros2"
        or model.environment.get("use_sim_time") is not True
        or model.observations.get("clock_advancing") is not True
    ):
        result["unknown_fields"].append("observed_ROS2_SIM_clock")
    for key in ("localization_ready", "costmaps_fresh", "obstacle_source_configured"):
        if model.navigation.get(key) is not True:
            result["unknown_fields"].append("navigation." + key)
    roles = policy.get("lifecycle_nodes", {})
    if type(roles) is not dict or set(roles) != {
        "localization",
        "planner",
        "controller",
        "collision_monitor",
        "navigator",
    }:
        result["unknown_fields"].append("explicit_lifecycle_roles")
        return result
    if any(type(n) is not str or not n for n in roles.values()) or len(set(roles.values())) != 5:
        result["unknown_fields"].append("unique_lifecycle_roles")
        return result
    if model.completeness.get("lifecycle") is not True:
        result["unknown_fields"].append("complete_lifecycle_observation")
    for role, name in roles.items():
        matches = [n for n in model.lifecycle if n.name == name]
        if (
            len(matches) != 1
            or matches[0].state != "ACTIVE"
            or matches[0].source != name + "/get_state"
            or not fresh(matches[0].captured_at.isoformat())
        ):
            result["unknown_fields"].append("lifecycle." + role)
    monitor = roles["collision_monitor"]
    parameters = model.observations.get("node_parameters", {}).get(monitor, {})
    parameter_time = model.observations.get("parameter_captured_at", {}).get(monitor)
    sensors = parameters.get("observation_sources")
    lidar = candidate["interfaces"]["sensing.lidar"][0]["name"]
    input_topic = observed_topic_parameter(model, monitor, "cmd_vel_in_topic", now=now)
    output_topic = observed_topic_parameter(model, monitor, "cmd_vel_out_topic", now=now)
    if (
        not fresh(parameter_time)
        or parameters.get("use_sim_time") is not True
        or not parameters.get("polygons")
        or type(sensors) is not list
        or not sensors
        or any(type(n) is not str or not n for n in sensors)
        or not any(
            observed_topic_parameter(model, monitor, n + ".topic", now=now) == lidar
            for n in sensors
        )
        or type(input_topic) is not str
        or not input_topic
        or type(output_topic) is not str
        or not output_topic
        or output_topic == input_topic
    ):
        result["unknown_fields"].append("fresh_collision_monitor_sensor_and_velocity_chain")
    localization = []
    for topic in model.graph.get("topics", []):
        name = topic.get("name")
        if topic.get(
            "msg_type"
        ) != "geometry_msgs/msg/PoseWithCovarianceStamped" or is_initial_pose_command(topic):
            continue
        matches = [s for s in model.signals if s.topic == name]
        header = model.observations.get("message_frames", {}).get(name, {})
        if (
            len(matches) == 1
            and matches[0].publisher_count == 1
            and matches[0].last_message_age_ms is not None
            and matches[0].last_message_age_ms <= matches[0].max_age_ms
            and fresh(matches[0].captured_at.isoformat())
            and header.get("source") == name
            and fresh(header.get("captured_at"))
            and header.get("frame_id") == candidate["frames"]["map"]
        ):
            localization.append(
                {"name": name, "ros_type": topic["msg_type"], "frames": deepcopy(header)}
            )
    if len(localization) != 1:
        result["unknown_fields"].append("unique_fresh_localization_pose_estimate_in_map_frame")

    cleaner_service = policy.get("cleaning_service")
    cleaner_state = policy.get("cleaning_state_topic")
    services = model.graph.get("services", [])
    matches = [
        s
        for s in services
        if s.get("name") == cleaner_service and s.get("srv_type") == "std_srvs/srv/SetBool"
    ]
    if type(cleaner_service) is not str or not cleaner_service or len(matches) != 1:
        result["unknown_fields"].append("explicit_SIM_cleaning_service")
    topics = model.graph.get("topics", [])
    if (
        type(cleaner_state) is not str
        or not cleaner_state
        or len(
            [
                t
                for t in topics
                if t.get("name") == cleaner_state and t.get("msg_type") == "std_msgs/msg/Bool"
            ]
        )
        != 1
    ):
        result["unknown_fields"].append("explicit_SIM_cleaning_state_topic")
    for topic, msg_type in [
        (input_topic, "geometry_msgs/msg/Twist"),
        (output_topic, "geometry_msgs/msg/Twist"),
        (cleaner_state, "std_msgs/msg/Bool"),
    ]:
        signals = [s for s in model.signals if s.topic == topic]
        typed = [t for t in topics if t.get("name") == topic and t.get("msg_type") == msg_type]
        if (
            len(typed) != 1
            or len(signals) != 1
            or signals[0].publisher_count != 1
            or (
                topic == cleaner_state
                and (
                    signals[0].last_message_age_ms is None
                    or signals[0].last_message_age_ms > signals[0].max_age_ms
                )
            )
            or not fresh(signals[0].captured_at.isoformat())
        ):
            result["unknown_fields"].append("fresh_unique_typed_control_stream:" + str(topic))
    for topic, direction in [(input_topic, "subscribers"), (output_topic, "publishers")]:
        observed_topics = [t for t in topics if t.get("name") == topic]
        if len(observed_topics) != 1 or monitor not in observed_topics[0].get(direction, []):
            result["unknown_fields"].append("observed_collision_monitor_connection:" + str(topic))
    if result["unknown_fields"]:
        return result
    result["specification"] = {
        "body_id": policy["body_id"],
        "model_identity": candidate["geometry"]["model_identity"],
        "frames": deepcopy(candidate["frames"]),
        "physical_radius_m": candidate["geometry"]["physical_radius_m"],
        "cleaning_polygon": deepcopy(cleaner["cleaning_polygon"]),
        "ros_capability_bindings": {
            "navigation.navigate_to_pose": deepcopy(
                {
                    **candidate["interfaces"]["navigation.navigate_to_pose"][0],
                    "lifecycle_nodes": {
                        "planner_server": roles["planner"],
                        "controller_server": roles["controller"],
                        "bt_navigator": roles["navigator"],
                    },
                }
            ),
            "localization.pose_estimate": localization[0],
            "state.odometry": deepcopy(candidate["interfaces"]["state.odometry"][0]),
            "mapping.occupancy_map": deepcopy(candidate["interfaces"]["mapping.occupancy_map"][0]),
            "sensing.lidar": deepcopy(candidate["interfaces"]["sensing.lidar"][0]),
            "safety.collision_monitor": {
                "name": monitor,
                "input_topic": input_topic,
                "output_topic": output_topic,
                "sensor_topic": lidar,
            },
            **{
                semantic: {
                    "name": cleaner_service,
                    "srv_type": "std_srvs/srv/SetBool",
                    "data": enabled,
                    "state_topic": cleaner_state,
                    "state_type": "std_msgs/msg/Bool",
                }
                for semantic, enabled in [("cleaning.enable", True), ("cleaning.disable", False)]
            },
        },
        "lifecycle_nodes": deepcopy(roles),
        "fixture_policy_id": policy["policy_id"],
        "evidence_domain": "SIMULATION",
        "cleaner_kind": "SIMULATED_CLEANING",
        "usable_for_real_execution": False,
    }
    result["status"] = "READY_FOR_SIM_WORKSPACE_COMPILATION"
    result["proposal_hash"] = digest(result)
    return result
