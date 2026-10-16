"""Deterministic planning contracts; fixtures confer no physical success evidence."""

import argparse
import json
from copy import deepcopy
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.cli.ros_cli import cmd_ros_engineering_case
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.engineering.contracts import RosEngineeringCaseV1
from rosclaw.connectors.ros.engineering.staging_plan import (
    parse_staging_documents,
    plan_staging_repairs,
)
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_compound_staging import compound
from tests.benchmarks.test_physical_aci import fixture
from tests.connectors.ros.test_engineering_case_compiler import NOW


def plan(aci, **kwargs):
    return plan_staging_repairs(
        aci.case,
        aci.model,
        aci.tools.files["robot.urdf"].encode(),
        aci.requirements,
        aci.tools.staging,
        **kwargs,
    )


@pytest.mark.parametrize("fault", ["tf", "scan", "life", "footprint"])
def test_four_supported_mechanisms_plan_without_model_or_validation(fault):
    aci = fixture(fault)
    before = deepcopy(aci.tools.staging)
    result = plan(aci)
    assert result["status"] == "CONDITIONAL_STAGING_PROPOSAL"
    assert result["current_authority"] == "NONE" and result["current_l1"] == "NOT_EVALUATED"
    assert len(result["proposal"]["proof_obligations"]) == 4
    assert aci.tools.staging == before and not aci.proposals and aci.validation_count == 0
    assert plan(aci) == result


def test_compound_plan_preserves_original_baseline_and_exact_reverse_rollback():
    aci = compound()
    result = plan(aci)
    proposal = result["proposal"]
    assert result["covered_violation_ids"] == ["F-SCAN-001", "F-TF-001"]
    assert len(proposal["changes"]) == 2
    assert proposal["rollback"] == [
        {"file": c["file"], "pointer": c["pointer"], "value": c["before"]}
        for c in reversed(proposal["changes"])
    ]
    proposal["changes"][0]["value"] = "caller mutation"
    assert plan(aci)["proposal"]["changes"][0]["value"] != "caller mutation"


def test_unknown_scan_mechanism_blocks_entire_compound_instead_of_partial_patch():
    aci = compound()
    aci.tools.staging["bringup.json"]["scan_bridge_state"] = "running"
    result = plan(aci)
    assert (
        result["status"] == "NEEDS_OPERATOR_OR_UNSUPPORTED_MECHANISM" and result["proposal"] is None
    )
    assert not aci.proposals


def test_unowned_route_has_no_guessed_shutdown_plan():
    result = plan(fixture("route"))
    assert result["status"] == "NEEDS_OPERATOR_OR_UNSUPPORTED_MECHANISM"
    assert result["proposal"] is None


def test_source_mismatch_and_stale_original_evidence_abstain():
    aci = fixture("tf")
    for now, review, expected in [
        (NOW, {"result": "MISMATCH"}, "ABSTAIN_SOURCE_MISMATCH"),
        (NOW + timedelta(days=1), None, "NEEDS_ORIGINAL_BODY_EVIDENCE"),
    ]:
        aci.case = compile_engineering_case(
            aci.model,
            aci.tools.files["robot.urdf"].encode(),
            aci.requirements,
            source_ref="fixture:review",
            now=now,
            evidence_domain="SEALED_REPLAY",
            source_review=review,
        )
        result = plan(aci)
        assert result["status"] == expected and result["proposal"] is None


def test_wrong_original_uri_or_requirements_cannot_generate_plan():
    aci = fixture("tf")
    with pytest.raises(ValueError, match="URDF"):
        plan_staging_repairs(
            aci.case, aci.model, b"different Body", aci.requirements, aci.tools.staging
        )
    req = deepcopy(aci.requirements)
    req["velocity_output_topic"] = "/different"
    with pytest.raises(ValueError, match="requirements"):
        plan_staging_repairs(
            aci.case, aci.model, aci.tools.files["robot.urdf"].encode(), req, aci.tools.staging
        )


@pytest.mark.parametrize(
    "nav2,bringup",
    [
        (b"x: &x [*x]", b"{}"),
        (b"x: &x [1]\ny: *x", b"{}"),
        (b"[]", b"{}"),
        (b"{}", b"[]"),
        (b"x: !unsafe value", b"{}"),
    ],
)
def test_nonconcrete_or_nonobject_staging_documents_are_rejected(nav2, bringup):
    with pytest.raises(ValueError):
        parse_staging_documents(nav2, bringup)


def test_actual_cli_function_exposes_only_conditional_plan_and_keeps_inputs(tmp_path, capsys):
    aci = fixture("tf")
    paths = {}
    for name, text in {**aci.tools.files, "snapshot.json": json.dumps(aci.model.to_dict())}.items():
        p = tmp_path / name
        p.write_text(text)
        paths[name] = p
    before = {p: p.read_bytes() for p in paths.values()}
    args = argparse.Namespace(
        snapshot=str(paths["snapshot.json"]),
        urdf=str(paths["robot.urdf"]),
        requirements=str(paths["requirements.json"]),
        reviewed_at=NOW.isoformat(),
        plan_staging=True,
        staging_nav2=str(paths["nav2.yaml"]),
        staging_bringup=str(paths["bringup.json"]),
        json=True,
    )
    assert cmd_ros_engineering_case(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["staging_plan"]["status"] == "CONDITIONAL_STAGING_PROPOSAL"
    assert response["staging_plan"]["proposal"]["intent"]["kind"] == "enable_localization_tf"
    assert all(p.read_bytes() == raw for p, raw in before.items())
    args.staging_bringup = None
    assert cmd_ros_engineering_case(args) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_original_urdf_recomputation_prevents_lowered_derived_geometry_floor():
    aci = fixture("footprint")
    payload = aci.case.model_dump(mode="json")
    payload["derived_documents"]["navigation_observation_review"]["candidate"]["geometry"][
        "physical_radius_m"
    ] = 0.00001
    payload.pop("case_hash")
    aci.case = RosEngineeringCaseV1.from_dict(
        {"case_hash": content_hash("roscase", payload), **payload}
    )
    result = plan(aci)
    assert result["proposal"] is None
    assert result["status"] == "NEEDS_OPERATOR_OR_UNSUPPORTED_MECHANISM"
    assert "immutable original URDF envelope cannot shrink" in result["reasons"][0]["reason"]
    assert result["current_l1"] == "NOT_EVALUATED"
