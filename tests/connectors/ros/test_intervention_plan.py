"""Synthetic data contracts; no ROS, files, Runtime or physical acceptance."""

import copy
from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.engineering.intervention_plan import (
    ParameterInterventionV1,
    RosInterventionPlanV1,
    compile_intervention_plan,
)
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.contracts.common import content_hash

NOW = datetime(2026, 10, 18, 8, tzinfo=UTC)
NODE = "/anonymous/process"


def inputs(age=0):
    params = {
        "use_sim_time": True,
        "transform_tolerance": 30.0,
        "FollowPath.plugin": "missing::Controller",
        "obstacle_layer.scan.sensor_frame": "wrong",
    }
    model = RosSystemModel(
        robot_id="unknown",
        snapshot_id="",
        captured_at=NOW,
        graph={"nodes": [{"name": NODE}], "topics": [], "services": [], "actions": []},
        observations={
            "node_parameters": {NODE: params},
            "parameter_captured_at": {NODE: (NOW - timedelta(milliseconds=age)).isoformat()},
            "node_use_sim_time": {NODE: True},
            "clock_last_receive_age_ms": 0.0,
            "clock_advancing": True,
        },
        environment={"probe_use_sim_time": False},
        completeness={"graph": True, "time": True},
    ).seal()
    graph = compile_operational_graph(model, source_ref="fixture://anonymous/original", now=NOW)
    staging = {
        "nav2.yaml": {
            "/anonymous": {
                "process": {
                    "ros__parameters": {
                        "use_sim_time": True,
                        "transform_tolerance": 30.0,
                        "FollowPath": {"plugin": "missing::Controller"},
                        "obstacle_layer": {"scan": {"sensor_frame": "wrong"}},
                    }
                }
            }
        },
        "bringup.json": {"immutable_control_route": "/declared_only"},
    }
    return graph, model, staging


def request(parameter="transform_tolerance", value=0.3, **changes):
    return {
        "node": NODE,
        "parameter": parameter,
        "operation": "SET",
        "proposed_value": value,
        "expected_after_value": value,
        "hypothesis": "current observed parameter may explain the consumer failure",
        "source_revision": "a" * 40,
        **changes,
    }


def reseal(document):
    value = copy.deepcopy(document)
    value.pop("plan_hash", None)
    return {"plan_hash": content_hash("rosintervention", value), **value}


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("transform_tolerance", 0.3),
        ("FollowPath.plugin", "package::DifferentController"),
        ("obstacle_layer.scan.sensor_frame", "observed_frame"),
    ],
)
def test_single_native_bound_change_and_exact_rollback_without_authority(parameter, value):
    graph, model, staging = inputs()
    original = copy.deepcopy(staging)
    plan = compile_intervention_plan(graph, model, staging, request(parameter, value))
    assert staging == original and plan.status == "CONDITIONAL_SIM_STAGING_PROPOSAL"
    assert plan.rollback_documents == original
    assert plan.rollback_staging_hash == plan.before_staging_hash
    assert plan.execution_authority == "NONE" and plan.current_l1 == "NOT_EVALUATED"
    assert plan.request.max_worlds == 3 and plan.request.max_after_goals == 2
    assert RosInterventionPlanV1.from_dict(plan.model_dump(mode="json")) == plan


def test_reset_removes_only_explicit_override_and_predicts_default_without_claiming_it():
    graph, model, staging = inputs()
    change = request(
        "obstacle_layer.scan.sensor_frame",
        None,
        operation="RESET_COMPONENT_DEFAULT",
        expected_after_value="",
    )
    plan = compile_intervention_plan(graph, model, staging, change)
    params = plan.staged_documents["nav2.yaml"]["/anonymous"]["process"]["ros__parameters"]
    assert params["obstacle_layer"]["scan"] == {}
    assert "installed_component_default_or_class_and_current_frame_checks" in plan.proof_obligations
    assert plan.classification == "INFERRED_CANDIDATE"


@pytest.mark.parametrize("age", [600, -1000])
def test_new_capture_cannot_refresh_old_or_future_readback(age):
    graph, model, staging = inputs(age)
    plan = compile_intervention_plan(graph, model, staging, request())
    assert plan.status == "NEEDS_READ_ONLY_PROBE"
    assert plan.staged_documents is plan.after_staging_hash is None


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("max_vel_x", 99.0),
        ("robot_radius", 0.01),
        ("use_sim_time", False),
        ("transform_tolerance", True),
        ("transform_tolerance", float("nan")),
        ("transform_tolerance", 30.0),
        ("FollowPath.plugin", ""),
    ],
)
def test_protected_or_unbounded_semantics_are_not_generic_patch_paths(parameter, value):
    with pytest.raises(ValueError):
        ParameterInterventionV1.model_validate(request(parameter, value))


def test_ambiguous_yaml_parameter_path_cannot_stage_a_guess():
    graph, model, staging = inputs()
    params = staging["nav2.yaml"]["/anonymous"]["process"]["ros__parameters"]
    params["obstacle_layer.scan.sensor_frame"] = "wrong"
    plan = compile_intervention_plan(
        graph, model, staging, request("obstacle_layer.scan.sensor_frame", "frame")
    )
    assert plan.status == "ABSTAIN" and plan.staged_documents is None


def test_staging_and_original_native_must_match():
    graph, model, staging = inputs()
    staging["nav2.yaml"]["/anonymous"]["process"]["ros__parameters"]["transform_tolerance"] = 20.0
    plan = compile_intervention_plan(graph, model, staging, request())
    assert plan.status == "ABSTAIN" and plan.staged_documents is None


def test_corrupted_model_is_not_accepted_as_current_original():
    graph, model, staging = inputs()
    model.robot_id = "tampered"
    with pytest.raises(ValueError, match="unchanged original"):
        compile_intervention_plan(graph, model, staging, request())


def test_rehashed_extra_change_or_renamed_target_cannot_pass_plan_link_checks():
    graph, model, staging = inputs()
    document = compile_intervention_plan(graph, model, staging, request()).model_dump(mode="json")
    document["staged_documents"]["bringup.json"]["immutable_control_route"] = "/forged"
    document["after_staging_hash"] = content_hash("acistaging", document["staged_documents"])
    with pytest.raises(ValueError, match="single parameter"):
        RosInterventionPlanV1.from_dict(reseal(document))


def test_noop_has_no_stage_payload():
    graph, model, staging = inputs()
    plan = compile_intervention_plan(
        graph, model, staging, request("FollowPath.plugin", "missing::Controller")
    )
    assert plan.status == "NO_CHANGE" and plan.staged_documents is None
