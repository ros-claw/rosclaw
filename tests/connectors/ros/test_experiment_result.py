"""Synthetic phase-link checks cannot promote reported booleans into physics."""

import copy
import hashlib
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.engineering.experiment_result import (
    RosExperimentResultV1,
    bind_experiment_result,
)
from rosclaw.connectors.ros.engineering.intervention_plan import compile_intervention_plan
from tests.connectors.ros.test_intervention_plan import NODE, NOW, inputs, request


def fixtures():
    graph, model, staging = inputs()
    plan = compile_intervention_plan(graph, model, staging, request())
    rows = []
    for index, phase in enumerate(["BEFORE", "AFTER", "ROLLBACK"]):
        kinds = ["native", "physics", "owner"]
        if phase == "AFTER":
            kinds += [
                "L1",
                "source_binding",
                "canonical_receipt",
                "canonical_receipt",
                "independent_stop",
                "strict_age",
                "independent_review",
            ]
        artifacts = [
            {
                "kind": kind,
                "locator": f"fixture://{phase}/{ordinal}/{kind}",
                "sha256": hashlib.sha256(f"{phase}-{ordinal}".encode()).hexdigest(),
            }
            for ordinal, kind in enumerate(kinds)
        ]
        rows.append(
            {
                "phase": phase,
                "run_id": f"run{index}",
                "world_instance_id": f"independent-instance{index}",
                "source_revision": "a" * 40,
                "image_id": "sha256:" + "b" * 64,
                "started_at": NOW + timedelta(seconds=index * 10),
                "finished_at": NOW + timedelta(seconds=index * 10 + 5),
                "native_captured_at": NOW + timedelta(seconds=index * 10 + 2),
                "parameter_captured_at": NOW + timedelta(seconds=index * 10 + 1.9),
                "observed_node": NODE,
                "observed_parameter": "transform_tolerance",
                "observed_value": 0.3 if phase == "AFTER" else 30.0,
                "staging_hash": plan.after_staging_hash
                if phase == "AFTER"
                else plan.before_staging_hash,
                "source_urdf_sha256": "c" * 64,
                "physical_scene_signature": "d" * 64,
                "controllers_sha256": "e" * 64,
                "requirements_sha256": "f" * 64,
                "declared_body_limits_hash": "unchanged-policy",
                "body_snapshot_hash": "after-body" if phase == "AFTER" else None,
                "reported_l1": "READY_FOR_SIM_TEST" if phase == "AFTER" else "NOT_EVALUATED",
                "reported_completed_action_ids": ["goal1", "goal2"] if phase == "AFTER" else [],
                "artifacts": artifacts,
            }
        )
    return plan, rows


def test_complete_link_structure_still_requires_original_independent_verification():
    plan, phases = fixtures()
    result = bind_experiment_result(plan, phases)
    assert result.status == "NEEDS_INDEPENDENT_ORIGINAL_VERIFICATION"
    assert result.original_files_read is False and result.causal_verification == "NOT_ESTABLISHED"
    assert result.execution_authority == "NONE" and result.memory_promotion == "NOT_GRANTED"
    assert RosExperimentResultV1.from_dict(result.model_dump(mode="json")) == result


@pytest.mark.parametrize("field", ["run_id", "world_instance_id"])
def test_reused_world_or_run_is_not_a_new_counterfactual(field):
    plan, phases = fixtures()
    phases[2][field] = phases[0][field]
    with pytest.raises(ValueError, match="independent new"):
        bind_experiment_result(plan, phases)


@pytest.mark.parametrize(
    "field",
    [
        "image_id",
        "source_revision",
        "source_urdf_sha256",
        "physical_scene_signature",
        "controllers_sha256",
        "requirements_sha256",
        "declared_body_limits_hash",
    ],
)
def test_change_in_frozen_identity_geometry_or_limits_rejected(field):
    plan, phases = fixtures()
    phases[1][field] = phases[1][field].replace(phases[1][field][-1], "0")
    with pytest.raises(ValueError, match="protected"):
        bind_experiment_result(plan, phases)


@pytest.mark.parametrize(
    "fault",
    [
        "overlap",
        "budget",
        "staging",
        "readback",
        "stale_readback",
        "wrong_target",
        "duplicate_receipt",
    ],
)
def test_time_staging_and_actual_readback_cannot_be_substituted(fault):
    plan, phases = fixtures()
    if fault == "overlap":
        phases[1]["started_at"] = NOW + timedelta(seconds=1)
    elif fault == "budget":
        phases[2]["finished_at"] = NOW + timedelta(seconds=1000)
    elif fault == "staging":
        phases[1]["staging_hash"] = "different-proposal"
    elif fault == "readback":
        phases[1]["observed_value"] = 30.0
    elif fault == "stale_readback":
        phases[1]["parameter_captured_at"] -= timedelta(seconds=1)
    elif fault == "wrong_target":
        phases[1]["observed_node"] = "/other/process"
    else:
        pins = [p for p in phases[1]["artifacts"] if p["kind"] == "canonical_receipt"]
        pins[1]["sha256"] = pins[0]["sha256"]
    with pytest.raises(ValueError):
        bind_experiment_result(plan, phases)


def test_missing_physics_is_explicit_incomplete_even_with_reported_ready():
    plan, phases = fixtures()
    phases[1]["artifacts"] = [p for p in phases[1]["artifacts"] if p["kind"] != "physics"]
    result = bind_experiment_result(plan, phases)
    assert (
        result.status == "INCOMPLETE_ORIGINAL_EVIDENCE"
        and "AFTER:physics" in result.missing_evidence
    )


def test_extra_success_boolean_is_not_an_attestation_field():
    plan, phases = fixtures()
    phases[1]["verified_success"] = True
    with pytest.raises(ValueError):
        bind_experiment_result(plan, phases)


def test_binding_does_not_mutate_original_reported_records():
    plan, phases = fixtures()
    before = copy.deepcopy(phases)
    bind_experiment_result(plan, phases)
    assert phases == before
