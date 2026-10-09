"""Cohort declarations cannot hide missing failures, model drift or shared Memory."""

import pytest

from rosclaw.connectors.ros.verification.fair_ab import (
    FEATURES,
    FairRunManifestV1,
    NativeInferenceParamsV1,
    PublicEvidenceRefV1,
    audit_fair_ab_cohort,
)
from rosclaw.connectors.ros.verification.model_budget import RegisteredModelBudgetV1


def manifest(arm="A0", seed=1, **changes):
    budget = RegisteredModelBudgetV1(
        provider="openai-codex",
        model="gpt-6.1-sol",
        reasoning_effort="low",
        max_requests=16,
        max_tool_calls=64,
        max_output_tokens_per_request=2048,
        max_total_tokens=120000,
        wall_budget_sec=600,
    )
    inference = NativeInferenceParamsV1(reasoning_effort="low")
    data = {
        "benchmark_id": "synthetic-schema-only",
        "scenario_id": "B1",
        "robot_id": "fixture",
        "arm": arm,
        "phase": "pilot",
        "seed": seed,
        "fixture_hash": "1" * 64,
        "world_hash": "2" * 64,
        "map_hash": "3" * 64,
        "start_pose": (float(seed), 0.0, 0.0),
        "code_commit": "a" * 40,
        "container_digest": "sha256:" + "4" * 64,
        "ROS_distro": "jazzy",
        "nav2_revision": "frozen-nav2",
        "opennav_revision": "frozen-opennav",
        "model_provider": budget.provider,
        "exact_model_id": budget.model,
        "inference_params": inference,
        "inference_params_hash": inference.declaration_digest(),
        "native_protocol_hash": "6" * 64,
        "task_input_hash": "7" * 64,
        "prompt_template_hash": "8" * 64,
        "initial_repository_hash": "9" * 64,
        "daemon_gateway_policy_hash": "a" * 64,
        "model_budget": budget,
        "budget_tokens": 120000,
        "wall_budget": 600,
        "tool_budget": 64,
        "tool_permissions": ("read", "request_action"),
        "feature_flags": FEATURES[arm],
        "memory_mode": "EVIDENCE_GATED" if arm == "A3" else "NONE",
        "permitted_action_mode": "SIM",
        "execution_id": f"exec-{arm}-{seed}",
        "worker_id": f"worker-{arm}-{seed}",
        "memory_namespace": f"memory-{arm}-{seed}",
    }
    return FairRunManifestV1.model_validate({**data, **changes})


def cohort(**changes):
    return tuple(manifest(arm, seed, **changes) for seed in range(1, 6) for arm in FEATURES)


def audit(runs, **changes):
    return audit_fair_ab_cohort(runs, expected_seeds=(1, 2, 3, 4, 5), phase="pilot", **changes)


def evidence():
    return (PublicEvidenceRefV1(path="public/fixture.json", sha256="b" * 64, size_bytes=7),)


def test_manifest_unknown_observations_never_become_success_or_zero():
    run = manifest()
    assert run.result == "NOT_RUN" and run.collision_count is None and run.coverage_ratio is None
    assert run.human_interventions is None and run.actual_model_id is None
    report = audit(cohort())
    assert report["status"] == "INCOMPLETE_DECLARATIONS"
    assert report["expected_runs"] == 20 and report["dispatch_authorized"] is False
    assert report["evidence_bytes_verified"] is False


def test_complete_declarations_retain_failed_runs_and_do_not_claim_execution():
    runs = list(
        cohort(
            result="FAIL",
            failure_code="FIXTURE_FAILED",
            human_interventions=0,
            evidence_refs=evidence(),
        )
    )
    report = audit(tuple(runs))
    assert report["status"] == "DECLARATIONS_MATCH"
    assert len(report["recorded_failures"]) == 20
    assert report["scope"] == "DECLARATIONS_ONLY_NOT_EXECUTION_ACCEPTANCE"
    assert report["robot_authorization"] is False and report["causal_benefit"] == "NOT_MEASURED"
    assert len(set(report["run_declaration_sha256"])) == 20


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_flags": FEATURES["A3"]},
        {"memory_mode": "EVIDENCE_GATED"},
        {"exact_model_id": "fallback-model"},
        {"budget_tokens": 120001},
        {"wall_budget": 601},
        {"tool_budget": 63},
        {"tool_permissions": ("read", "read")},
        {"permitted_action_mode": "REAL"},
        {"collision_count": True},
        {"coverage_ratio": float("nan")},
        {"coverage_ratio": True},
        {"start_pose": (True, 0, 0)},
        {"robot_id": "   "},
        {"container_digest": "moving-tag:latest"},
        {"fixture_hash": "UNKNOWN"},
        {"result": "PASS"},
        {"result": "FAIL", "failure_code": "FAIL"},
    ],
)
def test_invalid_or_overclaimed_manifest_is_rejected(changes):
    with pytest.raises(ValueError):
        manifest(**changes)


@pytest.mark.parametrize("field", ["execution_id", "worker_id", "memory_namespace"])
def test_shared_worker_or_memory_is_not_an_isolated_cohort(field):
    runs = list(cohort())
    runs[1] = runs[1].model_copy(update={field: getattr(runs[0], field)})
    with pytest.raises(ValueError, match="isolated"):
        audit(tuple(runs))


@pytest.mark.parametrize(
    "change",
    [
        {"exact_model_id": "other-model"},
        {"inference_params_hash": "f" * 64},
        {"native_protocol_hash": "f" * 64},
        {"prompt_template_hash": "f" * 64},
        {"daemon_gateway_policy_hash": "f" * 64},
        {"tool_permissions": ("read",)},
        {"start_pose": (100.0, 0.0, 0.0)},
        {"fixture_hash": "f" * 64},
        {
            "model_budget": RegisteredModelBudgetV1(
                provider="openai-codex",
                model="gpt-6.1-sol",
                reasoning_effort="high",
                max_requests=16,
                max_tool_calls=64,
                max_output_tokens_per_request=2048,
                max_total_tokens=120000,
                wall_budget_sec=600,
            )
        },
        {"actual_model_id": "fallback-model"},
    ],
)
def test_paired_controls_or_actual_model_cannot_drift(change):
    runs = list(cohort())
    runs[1] = runs[1].model_copy(update=change)
    with pytest.raises(ValueError):
        audit(tuple(runs))


def test_missing_failed_run_and_duplicate_replacement_cannot_be_dropped():
    runs = cohort()
    with pytest.raises(ValueError, match="missing"):
        audit(runs[:-1])
    with pytest.raises(ValueError, match="unreplaced"):
        audit(runs[:-1] + (runs[0],))


def test_evaluation_cannot_reuse_pilot_or_tuning_seeds():
    with pytest.raises(ValueError, match="reuse"):
        audit(cohort(), prior_seeds=(1, 100))
    with pytest.raises(ValueError):
        audit_fair_ab_cohort(cohort(), expected_seeds=(1, 2, 3, 4, 5), phase="evaluation")


def test_ten_evaluation_seeds_require_all_forty_runs():
    seeds = tuple(range(101, 111))
    runs = tuple(manifest(arm, seed, phase="evaluation") for seed in seeds for arm in FEATURES)
    report = audit_fair_ab_cohort(
        runs, expected_seeds=seeds, phase="evaluation", prior_seeds=(1, 2, 3, 4, 5)
    )
    assert report["expected_runs"] == 40 and report["status"] == "INCOMPLETE_DECLARATIONS"


def test_unknown_version_is_reported_instead_of_invented():
    report = audit(cohort(ROS_distro="UNKNOWN"))
    assert any(row["reason"] == "UNKNOWN_CONTROL" for row in report["incomplete"])


def test_hash_changes_with_outcomes_and_evidence_and_cannot_validate_copied_overrides():
    run = manifest()
    copied = run.model_copy(update={"feature_flags": FEATURES["A3"]})
    assert run.declaration_digest() != copied.declaration_digest()
    runs = list(cohort())
    runs[0] = copied
    with pytest.raises(ValueError):
        audit(tuple(runs))


def test_pass_declaration_requires_cleaning_safety_and_verified_stop():
    values = {
        "result": "PASS",
        "human_interventions": 0,
        "coverage_ratio": 0.99,
        "collision_count": 0,
        "actual_model_id": "gpt-6.1-sol",
        "usage_complete": True,
        "stop_verified": True,
        "evidence_refs": evidence(),
    }
    assert manifest(**values).result == "PASS"
    for change in (
        {"collision_count": 1},
        {"coverage_ratio": 0.97},
        {"stop_verified": None},
        {"actual_model_id": "fallback-model"},
    ):
        with pytest.raises(ValueError):
            manifest(**{**values, **change})


@pytest.mark.parametrize("field", ["parallel_tool_calls", "stream", "store"])
def test_inference_registration_never_coerces_integer_booleans(field):
    with pytest.raises(ValueError):
        NativeInferenceParamsV1(reasoning_effort="low", **{field: 1})
