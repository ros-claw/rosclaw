"""Synthetic binding proposals do not select a third actual robot."""

import hashlib
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.context.sim_attachment import validate_sim_attachment
from rosclaw.connectors.ros.context.sim_binding import propose_sim_fixture_binding
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.intelligence.system_model import Lifecycle, Signal
from tests.connectors.ros.test_body_discovery_candidate import NOW, fixture
from tests.connectors.ros.test_sim_attachment_region import attachment


def example(namespace="/sample", base="chassis", sensor="laser_mount"):
    model, data = fixture(namespace, base, sensor)
    names = {
        role: namespace + "/" + role
        for role in ("localization", "planner", "controller", "collision_monitor", "navigator")
    }
    model.environment = {"ros_generation": "ros2", "use_sim_time": True}
    model.observations["clock_advancing"] = True
    model.observations["clock_last_receive_age_ms"] = 10
    model.navigation = dict.fromkeys(
        ("localization_ready", "costmaps_fresh", "obstacle_source_configured"), True
    )
    model.lifecycle = [
        Lifecycle(name=n, source=n + "/get_state", state="ACTIVE", captured_at=NOW)
        for n in names.values()
    ]
    model.completeness["lifecycle"] = True
    monitor = names["collision_monitor"]
    model.observations["node_parameters"] = {
        monitor: {
            "use_sim_time": True,
            "observation_sources": ["ranger"],
            "polygons": ["PhysicalApproach"],
            "ranger.topic": namespace + "/scan2",
            "cmd_vel_in_topic": namespace + "/smoothed",
            "cmd_vel_out_topic": namespace + "/guarded",
        }
    }
    model.observations["parameter_captured_at"] = {monitor: NOW.isoformat()}
    service, state = namespace + "/brush_switch", namespace + "/brush_active"
    model.graph["services"] = [{"name": service, "srv_type": "std_srvs/srv/SetBool"}]
    for topic, kind in [
        (namespace + "/smoothed", "geometry_msgs/msg/Twist"),
        (namespace + "/guarded", "geometry_msgs/msg/Twist"),
        (state, "std_msgs/msg/Bool"),
    ]:
        model.graph["topics"].append(
            {
                "name": topic,
                "msg_type": kind,
                "subscribers": [monitor] if topic == namespace + "/smoothed" else [],
                "publishers": [monitor] if topic == namespace + "/guarded" else [],
            }
        )
        model.signals.append(
            Signal(
                topic=topic,
                source="native",
                captured_at=NOW,
                publisher_count=1,
                last_message_age_ms=50,
            )
        )
    pose_topic = namespace + "/measured_pose"
    model.graph["topics"].append(
        {"name": pose_topic, "msg_type": "geometry_msgs/msg/PoseWithCovarianceStamped"}
    )
    model.signals.append(
        Signal(
            topic=pose_topic,
            source="native",
            captured_at=NOW,
            publisher_count=1,
            last_message_age_ms=50,
        )
    )
    model.observations["message_frames"][pose_topic] = {
        "source": pose_topic,
        "captured_at": NOW.isoformat(),
        "frame_id": "world",
    }
    model.seal()
    cleaner = attachment()
    policy = {
        "schema_version": "rosclaw.sim_fixture_body_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "evidence_domain": "SIMULATION",
        "approved": True,
        "policy_id": "synthetic_fixture_policy",
        "body_id": "anonymous_sim",
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": hashlib.sha256(data).hexdigest(),
        "attachment_hash": validate_sim_attachment(cleaner)["attachment_hash"],
        "lifecycle_nodes": names,
        "cleaning_service": service,
        "cleaning_state_topic": state,
    }
    return model, data, cleaner, policy


@pytest.mark.parametrize(
    "namespace,base,sensor",
    [("/sample", "chassis", "laser_mount"), ("/alternate/robot", "renamed_base", "range2")],
)
def test_fresh_explicit_fixture_policy_generates_renamed_candidate_without_capability_or_binding_claim(
    namespace, base, sensor
):
    model, data, cleaner, policy = example(namespace, base, sensor)
    before = model.to_dict()
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "READY_FOR_SIM_WORKSPACE_COMPILATION"
    assert result["unknown_fields"] == []
    assert result["specification"]["frames"]["base"] == base
    assert (
        result["specification"]["ros_capability_bindings"]["safety.collision_monitor"][
            "sensor_topic"
        ]
        == namespace + "/scan2"
    )
    assert (
        result["specification"]["ros_capability_bindings"]["navigation.navigate_to_pose"]["name"]
        == namespace + "/go"
    )
    assert result["specification"]["model_identity"] == "anonymous"
    assert result["capabilities_granted"] == [] and result["binding_verified"] is False
    assert (
        result["physical_acceptance_level"] == "NOT_RUN" and not result["usable_for_real_execution"]
    )
    original_hash = result.pop("proposal_hash")
    assert original_hash == digest(result)
    assert model.to_dict() == before


@pytest.mark.parametrize(
    "fault",
    [
        "real",
        "unapproved",
        "wrong_snapshot",
        "wrong_urdf",
        "wrong_attachment",
        "stale_lifecycle",
        "inactive",
        "wrong_lifecycle_source",
        "missing_lifecycle",
        "stale_parameters",
        "wrong_sensor",
        "same_velocity_topics",
        "multiple_publishers",
        "stale_velocity",
        "no_cleaner",
        "wrong_cleaner_type",
        "no_clock",
        "not_sim",
        "unknown_navigation",
    ],
)
def test_policy_readiness_or_control_source_faults_produce_no_bindable_specification(fault):
    model, data, cleaner, policy = example()
    if fault == "real":
        policy["evidence_domain"] = "REAL"
    elif fault == "unapproved":
        policy["approved"] = False
    elif fault == "wrong_snapshot":
        policy["source_snapshot_hash"] = "old"
    elif fault == "wrong_urdf":
        policy["source_urdf_sha256"] = "other"
    elif fault == "wrong_attachment":
        policy["attachment_hash"] = "other"
    elif fault == "stale_lifecycle":
        model.lifecycle[0].captured_at -= timedelta(seconds=6)
    elif fault == "inactive":
        model.lifecycle[0].state = "INACTIVE"
    elif fault == "wrong_lifecycle_source":
        model.lifecycle[0].source = "/other/get_state"
    elif fault == "missing_lifecycle":
        model.completeness["lifecycle"] = False
    elif fault == "stale_parameters":
        model.observations["parameter_captured_at"]["/sample/collision_monitor"] = (
            NOW - timedelta(seconds=6)
        ).isoformat()
    elif fault == "wrong_sensor":
        model.observations["node_parameters"]["/sample/collision_monitor"]["ranger.topic"] = (
            "/other/scan"
        )
    elif fault == "same_velocity_topics":
        model.observations["node_parameters"]["/sample/collision_monitor"]["cmd_vel_out_topic"] = (
            "/sample/smoothed"
        )
    elif fault == "multiple_publishers":
        next(s for s in model.signals if s.topic == "/sample/guarded").publisher_count = 2
    elif fault == "stale_velocity":
        next(s for s in model.signals if s.topic == "/sample/guarded").captured_at = (
            NOW - timedelta(seconds=6)
        )
    elif fault == "no_cleaner":
        model.graph["services"] = []
    elif fault == "wrong_cleaner_type":
        model.graph["services"][0]["srv_type"] = "std_srvs/srv/Trigger"
    elif fault == "no_clock":
        model.observations["clock_advancing"] = False
    elif fault == "not_sim":
        model.environment["use_sim_time"] = False
    elif fault == "unknown_navigation":
        model.navigation.pop("costmaps_fresh")
    model.seal()
    if fault not in ["wrong_snapshot"]:
        policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "UNKNOWN" and result["unknown_fields"]
    assert "specification" not in result and "proposal_hash" not in result
    assert not result["binding_verified"] and not result["usable_for_real_execution"]


def test_idle_event_driven_velocity_topics_need_fresh_graph_not_prior_motion():
    model, data, cleaner, policy = example()
    for signal in [s for s in model.signals if s.topic in ("/sample/smoothed", "/sample/guarded")]:
        signal.last_message_age_ms = None
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "READY_FOR_SIM_WORKSPACE_COMPILATION"
    assert result["capabilities_granted"] == []


def test_unrelated_velocity_publisher_cannot_prove_the_monitor_control_chain():
    model, data, cleaner, policy = example()
    next(t for t in model.graph["topics"] if t["name"] == "/sample/guarded")["publishers"] = [
        "/unrelated"
    ]
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "UNKNOWN"
    assert "specification" not in result


@pytest.mark.parametrize("source_name", ["scan", "ranger", "renamed_front_lidar"])
def test_bound_collision_monitor_resolves_actual_source_name_without_core_profile(source_name):
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    model, data, cleaner, policy = example()
    params = model.observations["node_parameters"]["/sample/collision_monitor"]
    params["observation_sources"] = [source_name]
    params[source_name + ".topic"] = "/sample/scan2"
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    proposal = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert proposal["status"] == "READY_FOR_SIM_WORKSPACE_COMPILATION"
    model.body = {
        "effective_body_hash": "synthetic_compiled_binding",
        "frames": proposal["specification"]["frames"],
        "ros_capability_bindings": proposal["specification"]["ros_capability_bindings"],
    }
    model.seal()
    monitor = next(
        r
        for r in resolve_capabilities(model, now=NOW)
        if r["semantic_id"] == "safety.collision_monitor"
    )
    assert monitor["status"] == "AVAILABLE"
    assert not monitor["usable_for_real_execution"]


def test_advancing_but_stale_clock_is_not_live_binding_evidence():
    model, data, cleaner, policy = example()
    model.observations["clock_last_receive_age_ms"] = 5000
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "UNKNOWN"
    assert "specification" not in result


def test_named_lifecycle_bindings_resolve_navigation_without_canonical_node_names():
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    model, data, cleaner, policy = example("/isolated/robot", "renamed_base", "range_frame")
    proposal = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    model.body = {
        "effective_body_hash": "synthetic_compiled_binding",
        "frames": proposal["specification"]["frames"],
        "ros_capability_bindings": proposal["specification"]["ros_capability_bindings"],
    }
    model.seal()
    nav = next(
        r
        for r in resolve_capabilities(model, now=NOW)
        if r["semantic_id"] == "navigation.navigate_to_pose"
    )
    assert nav["status"] == "AVAILABLE"
    assert nav["requirements"]["nav2.lifecycle_binding"] is True
    assert not nav["usable_for_real_execution"]
    model.signals[-1].last_message_age_ms = 6000
    model.seal()
    nav = next(
        r
        for r in resolve_capabilities(model, now=NOW)
        if r["semantic_id"] == "navigation.navigate_to_pose"
    )
    assert nav["status"] == "BLOCKED"


def test_fresh_unbound_sensor_cannot_replace_stale_bound_lidar():
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    model, data, cleaner, policy = example()
    proposal = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    model.body = {
        "effective_body_hash": "synthetic_compiled_binding",
        "frames": proposal["specification"]["frames"],
        "ros_capability_bindings": proposal["specification"]["ros_capability_bindings"],
    }
    model.graph["topics"].append({"name": "/unbound_scan", "msg_type": "sensor_msgs/msg/LaserScan"})
    model.signals.append(
        Signal(
            topic="/unbound_scan",
            source="native",
            captured_at=NOW,
            publisher_count=1,
            last_message_age_ms=50,
        )
    )
    next(s for s in model.signals if s.topic == "/sample/scan2").last_message_age_ms = 6000
    model.seal()
    nav = next(
        r
        for r in resolve_capabilities(model, now=NOW)
        if r["semantic_id"] == "navigation.navigate_to_pose"
    )
    assert nav["status"] == "BLOCKED"
    assert nav["requirements"]["sensing.lidar"] is False


@pytest.mark.parametrize(
    "fault", ["missing_pose", "initialpose_only", "wrong_pose_frame", "missing_polygons"]
)
def test_localization_command_or_wrong_frame_never_proves_pose_estimate_readiness(fault):
    model, data, cleaner, policy = example()
    if fault == "missing_pose":
        model.signals = [s for s in model.signals if s.topic != "/sample/measured_pose"]
    elif fault == "initialpose_only":
        next(t for t in model.graph["topics"] if t["name"] == "/sample/measured_pose")["name"] = (
            "/sample/initialpose"
        )
        model.signals[-1].topic = "/sample/initialpose"
    elif fault == "wrong_pose_frame":
        model.observations["message_frames"]["/sample/measured_pose"]["frame_id"] = "odom"
    else:
        model.observations["node_parameters"]["/sample/collision_monitor"].pop("polygons")
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    result = propose_sim_fixture_binding(model, data, attachment=cleaner, policy=policy, now=NOW)
    assert result["status"] == "UNKNOWN" and "specification" not in result
