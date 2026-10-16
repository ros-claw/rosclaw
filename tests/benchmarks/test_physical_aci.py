"""Typed proposals and negative controls, without model or physical execution."""

import json
from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import compile_engineering_case
from tests.connectors.ros.test_engineering_case_compiler import NOW, setup


def fixture(fault="footprint", evaluator=None, **options):
    model, urdf, req = setup()
    # These tests use the original operator's supported Nav2 staging field names.
    params = model.observations["node_parameters"]
    nav = {
        "/sample": {
            "amcl": {"ros__parameters": {"tf_broadcast": True}},
            "lifecycle_manager_navigation": {"ros__parameters": {"node_names": ["controller"]}},
            "local_costmap": {"local_costmap": {"ros__parameters": {"robot_radius": 0.4}}},
            "global_costmap": {"global_costmap": {"ros__parameters": {"robot_radius": 0.4}}},
        }
    }
    for topic, name in zip(req["costmap_topics"], ["local_costmap", "global_costmap"], strict=True):
        publisher = f"/sample/{name}/{name}"
        next(t for t in model.graph["topics"] if t["name"] == topic)["publishers"] = [publisher]
        params[publisher] = {
            "robot_radius": 0.1 if fault == "footprint" else 0.4,
            "footprint": "[]",
            "robot_base_frame": "chassis",
        }
        model.observations["parameter_captured_at"][publisher] = NOW.isoformat()
        nav["/sample"][name][name]["ros__parameters"]["robot_radius"] = params[publisher][
            "robot_radius"
        ]
    bringup = {"scan_bridge_state": "paused_after_initialization" if fault == "scan" else "running"}
    if fault == "scan":
        next(s for s in model.signals if s.topic == "/sample/scan2").last_message_age_ms = 700
    if fault == "route":
        next(t for t in model.graph["topics"] if t["name"] == req["velocity_output_topic"])[
            "publishers"
        ].append("/unowned")
    if fault == "tf":
        previous = req["lifecycle_nodes"]["localization"]
        node = "/sample/amcl"
        req["lifecycle_nodes"]["localization"] = node
        for row in model.lifecycle:
            if row.name == previous:
                row.name = node
                row.source = node + "/get_state"
        for topic in model.graph["topics"]:
            if topic["name"] == req["localization_topic"]:
                topic["publishers"] = [node]
        params[node] = {"tf_broadcast": False}
        model.observations["parameter_captured_at"][node] = NOW.isoformat()
        nav["/sample"]["amcl"]["ros__parameters"]["tf_broadcast"] = False
        model.transforms = [edge for edge in model.transforms if edge.parent != "world"]
    if fault == "life":
        next(
            row for row in model.lifecycle if row.name == req["lifecycle_nodes"]["planner"]
        ).state = "INACTIVE"
        manager = "/sample/lifecycle_manager_navigation"
        params[manager] = {"node_names": ["controller"]}
        model.observations["parameter_captured_at"][manager] = NOW.isoformat()
    model.seal()
    case = compile_engineering_case(
        model, urdf, req, source_ref="fixture:original", now=NOW, evidence_domain="SEALED_REPLAY"
    )
    files = {
        "robot.urdf": urdf.decode(),
        "requirements.json": json.dumps(req),
        "native.json": "{}",
        "nav2.yaml": json.dumps(nav),
        "bringup.json": json.dumps(bringup),
    }
    tools = TaskTools(
        files,
        arm="B",
        reviewed_at=NOW.isoformat(),
        minimum_radius_m=case.derived_documents["navigation_observation_review"]["candidate"][
            "geometry"
        ]["physical_radius_m"],
    )
    return PhysicalACI(
        case,
        model,
        tools,
        evaluator=evaluator,
        operator_source="b" * 40 if evaluator else None,
        original_run_id="a" * 32 if evaluator else None,
        **options,
    )


def repair(aci, kind="cover_urdf_envelope", **values):
    request = {
        "operation": "ros_engineer.propose_repair",
        "arguments": {"case_hash": aci.case.case_hash, "repair": {"kind": kind, **values}},
    }
    return aci.call(request)


def test_finish_never_validates_or_changes_staging():
    calls = []
    aci = fixture(evaluator=lambda *a, **k: calls.append(a))
    before = deepcopy(aci.tools.staging)
    result = aci.call({"operation": "finish", "arguments": {}})
    assert (
        result["status"] == "FINISHED_REPORT_ONLY" and calls == [] and aci.tools.staging == before
    )
    assert (
        aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})["status"]
        == "REJECTED_TYPED_ACI_REQUEST"
    )


def test_compact_inspection_and_typed_object_arguments():
    aci = fixture()
    result = aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})
    assert len(json.dumps(result)) < 3000 and result["current_authority"] == "NONE"
    for request in [
        {"operation": "finish", "arguments": "{}"},
        {"operation": "finish", "arguments": {"permit": True}},
        {"operation": "raw_ros_write", "arguments": {}},
    ]:
        assert aci.call(request)["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert PhysicalACI.request_schema()["oneOf"]


def test_minimal_geometry_patch_and_rollback_do_not_mutate_originals():
    aci = fixture()
    before = deepcopy(aci.tools.staging)
    result = repair(aci, radius_m=0.4)
    assert result["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
    proposal = result["proposal"]
    assert len(proposal["changes"]) == 2
    assert all(c["before"] == 0.1 and c["value"] == 0.4 for c in proposal["changes"])
    assert all(c["value"] == 0.1 for c in proposal["rollback"]) and aci.tools.staging == before
    proposal["changes"][0]["value"] = 0.01
    assert aci.proposals[proposal["proposal_id"]]["changes"][0]["value"] == 0.4


@pytest.mark.parametrize("value", [0.01, True, float("nan"), float("inf"), "0.4"])
def test_geometry_shrink_and_untyped_values_rejected(value):
    assert repair(fixture(), radius_m=value)["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_unknown_or_healthy_fault_never_produces_repair():
    assert repair(fixture("healthy"), radius_m=0.4)["status"] == "REJECTED_TYPED_ACI_REQUEST"
    aci = fixture()
    result = repair(aci, "resume_scan_bridge")
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_scan_staging_requires_actual_paused_policy():
    aci = fixture("scan")
    result = repair(aci, "resume_scan_bridge")
    assert result["proposal"]["changes"][0]["value"] == "running"
    aci.tools.staging["bringup.json"]["scan_bridge_state"] = "running"
    assert repair(aci, "resume_scan_bridge")["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_competing_route_cannot_be_disabled_without_owner_proof():
    result = repair(fixture("route"), "disable_owned_fixture_endpoint", ownership_ref="invented")
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_explicit_validation_budget_and_incomplete_success_preserved():
    calls = []

    def evaluator(staging, selected, index, **kwargs):
        calls.append((staging, selected, index))
        return {"verified_success": True, "text": "trust me"}

    aci = fixture(evaluator=evaluator)
    proposal = repair(aci, radius_m=0.4)["proposal"]
    request = {
        "operation": "ros_engineer.validate_proposal",
        "arguments": {"case_hash": aci.case.case_hash, "proposal_id": proposal["proposal_id"]},
    }
    for remaining in [None, 0, float("nan"), 301]:
        assert aci.call(request, remaining_sec=remaining)["status"] == "REJECTED_TYPED_ACI_REQUEST"
    assert not calls
    result = aci.call(request, remaining_sec=100)
    assert not result["verification"]["accepted_success"]
    assert result["verification"]["original_result"]["verified_success"] is True
    aci.call(request, remaining_sec=100)
    assert (
        aci.call(request, remaining_sec=100)["status"] == "REJECTED_TYPED_ACI_REQUEST"
        and len(calls) == 2
    )


def test_changed_originals_staging_or_case_revision_block_validation():
    calls = []
    aci = fixture(evaluator=lambda *a, **k: calls.append(a))
    proposal = repair(aci, radius_m=0.4)["proposal"]
    request = {
        "operation": "ros_engineer.validate_proposal",
        "arguments": {"case_hash": aci.case.case_hash, "proposal_id": proposal["proposal_id"]},
    }
    aci.tools.staging["bringup.json"]["scan_bridge_state"] = "paused_after_initialization"
    assert (
        aci.call(request, remaining_sec=100)["status"] == "REJECTED_TYPED_ACI_REQUEST" and not calls
    )
    request["arguments"]["case_hash"] = "old"
    assert aci.call(request, remaining_sec=100)["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_late_positive_physics_is_not_accepted():
    clock = iter([0.0, 101.0])

    def evaluator(*a, **k):
        return {
            "verified_success": True,
            "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW",
            "source": "b" * 40,
            "new_world_run": "c" * 32,
            "fresh_l1_ref": {"path": "original", "sha256": "1" * 64},
            "canonical_receipts": [
                {"path": "a", "sha256": "2" * 64},
                {"path": "b", "sha256": "3" * 64},
            ],
            "independent_stop_ref": {"path": "stop", "sha256": "4" * 64},
            "strict_source_envelope": {"path": "age", "sha256": "5" * 64},
        }

    aci = fixture(evaluator=evaluator, clock=lambda: next(clock))
    proposal = repair(aci, radius_m=0.4)["proposal"]
    r = aci.call(
        {
            "operation": "ros_engineer.validate_proposal",
            "arguments": {"case_hash": aci.case.case_hash, "proposal_id": proposal["proposal_id"]},
        },
        remaining_sec=100,
    )
    assert r["verification"]["evidence_complete"] and not r["verification"]["accepted_success"]
    assert r["case_L1"] == "NOT_EVALUATED" and r["current_authority"] == "NONE"


def test_readonly_investigation_does_not_fake_a_probe():
    aci = fixture()
    r = aci.call(
        {
            "operation": "ros_engineer.investigate",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "issue_id": "F-GEOM-001",
                "check_id": "independent_full_L1_effective_footprint_review",
            },
        }
    )
    assert r["status"] == "NEEDS_OPERATOR_READ_ONLY_PROBE" and not r["fresh_probe_performed"]


def test_tf_and_managed_node_semantics_compile_minimal_allowed_fields():
    aci = fixture("tf")
    p = repair(aci, "enable_localization_tf")["proposal"]
    assert len(p["changes"]) == 1 and p["changes"][0]["value"] is True
    assert any("competing" in condition for condition in p["preconditions"])
    aci = fixture("life")
    p = repair(aci, "activate_managed_nodes", node_names=["controller", "planner"])["proposal"]
    assert p["changes"][0]["before"] == ["controller"]
    for names in [["planner"], ["controller", "unobserved"], ["controller", "/sample/controller"]]:
        assert (
            repair(aci, "activate_managed_nodes", node_names=names)["status"]
            == "REJECTED_TYPED_ACI_REQUEST"
        )


def test_stale_manager_readback_is_not_an_actionable_cause():
    from datetime import timedelta

    from rosclaw.connectors.ros.engineering import compile_engineering_case

    aci = fixture("life")
    aci.model.observations["parameter_captured_at"]["/sample/lifecycle_manager_navigation"] = (
        NOW - timedelta(seconds=1)
    ).isoformat()
    aci.model.seal()
    aci.case = compile_engineering_case(
        aci.model,
        aci.tools.files["robot.urdf"].encode(),
        aci.requirements,
        source_ref="fixture:original",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    assert (
        repair(aci, "activate_managed_nodes", node_names=["controller", "planner"])["status"]
        == "REJECTED_TYPED_ACI_REQUEST"
    )


def test_yaml_scientific_float_roundtrip_does_not_change_unrelated_staging():
    aci = fixture()
    aci.tools.staging["nav2.yaml"]["/sample"]["unrelated"] = {"epsilon": 1e-5}
    result = repair(aci, radius_m=0.4)
    assert result["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
    assert aci.tools.staging["nav2.yaml"]["/sample"]["unrelated"]["epsilon"] == 1e-5


def test_onboarding_uses_four_observed_interfaces_not_invented_names():
    aci = fixture("healthy")
    request = {
        "operation": "ros_engineer.propose_onboarding",
        "arguments": {
            "case_hash": aci.case.case_hash,
            "interfaces": {
                "mapping": "/sample/floor",
                "odometry": "/sample/position_feedback",
                "lidar": "/sample/scan2",
                "navigate_to_pose": "/sample/go",
            },
        },
    }
    assert aci.call(request)["status"] == "OBSERVED_ONBOARDING_PROPOSAL_NOT_BOUND"
    request["arguments"]["interfaces"]["navigate_to_pose"] = "/invented/action"
    assert aci.call(request)["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_owned_endpoint_review_is_bound_to_case_and_declaration():
    from rosclaw.contracts.common import content_hash

    aci = fixture("route")
    declaration = {
        "enabled": True,
        "topic": aci.requirements["velocity_output_topic"],
        "msg_type": "geometry_msgs/msg/Twist",
    }
    aci.tools.staging["bringup.json"]["additional_command_endpoint"] = declaration
    review = {
        "case_hash": aci.case.case_hash,
        "staging_hash": content_hash("acistaging", aci.tools.staging),
        "scope": "SDK_OWNED_ADDITIONAL_STAGING_ENDPOINT",
        "declaration": declaration,
        "node_name": "/unowned",
        "original_evidence_refs": ["fixture:closed-operator-receipt"],
    }
    aci.ownership_reviews = {"pinned": review}
    result = repair(aci, "disable_owned_fixture_endpoint", ownership_ref="pinned")
    assert result["proposal"]["changes"] == [
        {
            "file": "bringup.json",
            "pointer": "/additional_command_endpoint/enabled",
            "before": True,
            "value": False,
        }
    ]
    aci.ownership_reviews["pinned"]["node_name"] = aci.requirements["lifecycle_nodes"][
        "collision_monitor"
    ]
    assert (
        repair(aci, "disable_owned_fixture_endpoint", ownership_ref="pinned")["status"]
        == "REJECTED_TYPED_ACI_REQUEST"
    )
