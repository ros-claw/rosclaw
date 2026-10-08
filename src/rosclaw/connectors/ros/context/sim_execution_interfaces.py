"""Observed SIM interface proposal; no executor registration or motion authority."""

from datetime import UTC, datetime

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint, freeze_sim_endpoints


def propose_sim_execution_interfaces(model, policy, specification, *, now=None):
    now = now or datetime.now(UTC)
    if model.snapshot_hash != model.compute_snapshot_hash():
        raise ValueError("ROS snapshot hash mismatch")
    result = {
        "schema_version": "rosclaw.sim_execution_interfaces.v1",
        "status": "UNKNOWN",
        "unknown_fields": [],
        "source_snapshot_hash": model.snapshot_hash,
        "physical_acceptance_level": "NOT_RUN",
        "capabilities_granted": [],
        "binding_verified": False,
        "usable_for_real_execution": False,
    }
    declaration = policy.get("execution_interfaces")
    if type(declaration) is not dict or set(declaration) != {
        "endpoints",
        "observation_topic",
        "coverage_lifecycle_node",
    }:
        result["unknown_fields"].append("explicit_complete_execution_interface_declaration")
        return result
    try:
        endpoints = freeze_sim_endpoints(declaration["endpoints"])
        observation = absolute_endpoint(declaration["observation_topic"])
        node = absolute_endpoint(declaration["coverage_lifecycle_node"])
    except ValueError as exc:
        result["unknown_fields"].append(str(exc))
        return result
    if (
        policy.get("approved") is not True
        or policy.get("evidence_domain") != "SIMULATION"
        or policy.get("source") != "simulator_operator_fixture_policy"
        or policy.get("source_snapshot_hash") != model.snapshot_hash
        or not -100 <= model.age_ms(now) <= 5000
        or model.completeness.get("graph") is not True
    ):
        result["unknown_fields"].append("fresh_explicit_matching_SIM_fixture_policy")
    bindings = specification["ros_capability_bindings"]
    if (
        endpoints["navigate_to_pose"] != bindings["navigation.navigate_to_pose"]["name"]
        or endpoints["cleaning"] != bindings["cleaning.enable"]["name"]
    ):
        result["unknown_fields"].append("execution_endpoints_match_observed_body_bindings")
    for role, kind in (
        ("navigate_to_pose", "nav2_msgs/action/NavigateToPose"),
        ("navigate_through_poses", "nav2_msgs/action/NavigateThroughPoses"),
        ("navigate_complete_coverage", "opennav_coverage_msgs/action/NavigateCompleteCoverage"),
    ):
        matches = [a for a in model.graph.get("actions", []) if a.get("name") == endpoints[role]]
        if len(matches) != 1 or matches[0].get("action_type") != kind:
            result["unknown_fields"].append("unique_typed_action:" + role)
    for role, kind in (
        ("set_initial_pose", "nav2_msgs/srv/SetInitialPose"),
        ("lease", "std_srvs/srv/SetBool"),
        ("cleaning", "std_srvs/srv/SetBool"),
    ):
        matches = [s for s in model.graph.get("services", []) if s.get("name") == endpoints[role]]
        if len(matches) != 1 or matches[0].get("srv_type") != kind:
            result["unknown_fields"].append("unique_typed_service:" + role)
    matches = [item for item in model.lifecycle if item.name == node]
    if (
        len(matches) != 1
        or matches[0].state != "ACTIVE"
        or matches[0].source != node + "/get_state"
        or not -100 <= (now - matches[0].captured_at).total_seconds() * 1000 <= 5000
    ):
        result["unknown_fields"].append("fresh_active_coverage_lifecycle")
    topics = [t for t in model.graph.get("topics", []) if t.get("name") == observation]
    signals = [s for s in model.signals if s.topic == observation]
    if (
        len(topics) != 1
        or topics[0].get("msg_type") != "std_msgs/msg/String"
        or len(signals) != 1
        or signals[0].publisher_count != 1
        or signals[0].last_message_age_ms is None
        or not 0 <= signals[0].last_message_age_ms < 300
        or not -100 <= (now - signals[0].captured_at).total_seconds() * 1000 <= 5000
    ):
        result["unknown_fields"].append("fresh_unique_typed_independent_observer_stream")
    if result["unknown_fields"]:
        return result
    result.update(
        status="READY_FOR_SIM_INTERFACE_COMPILATION",
        endpoints=dict(endpoints),
        observation_topic=observation,
        coverage_lifecycle_node=node,
        requires_independent_source_and_actual_map_admission=True,
    )
    result["proposal_hash"] = digest(result)
    return result
