"""Retry orchestration must execute a same-seed, whitelisted patch."""

import pytest

from rosclaw.how.retry_orchestrator import RetryOrchestrator
from rosclaw.sandbox.backends import ReplayReport
from rosclaw.sandbox.evidence import SimulationEvidenceVerification


def _verified(_receipt: dict) -> SimulationEvidenceVerification:
    replay = ReplayReport(True, True, True, True, 0.0, "strict_replay_verified")
    return SimulationEvidenceVerification(True, replay)


def test_retry_executes_same_scenario_and_records_lineage():
    submitted = []

    def submit(action):
        submitted.append(action)
        return {
            "scenario_id": action["scenario"]["scenario_id"],
            "seed": action["scenario"]["seed"],
            "is_safe": True,
            "physics_executed": True,
            "evidence_domain": "SIMULATION",
            "receipt_verified": True,
            "data_quality_pass": True,
            "replay_report": {"verified": True},
        }

    original = {
        "action_id": "action_original",
        "execution_mode": "SIMULATION",
        "scenario": {"scenario_id": "shield-42", "seed": 42, "friction": 0.7},
        "parameters": {"speed_scale": 1.0},
    }
    result = RetryOrchestrator(submit, receipt_verifier=_verified).execute(
        original, {"speed_scale": 0.5, "safety_clearance": 0.03}, retry_budget=3
    )

    assert result.executed is True
    assert submitted[0]["scenario"] == original["scenario"]
    assert submitted[0]["parameters"]["speed_scale"] == 0.5
    assert submitted[0]["retry_lineage"] == ["action_original"]
    assert original["parameters"]["speed_scale"] == 1.0


def test_retry_rejects_unsafe_patch_before_submit():
    with pytest.raises(ValueError, match="RETRY_PATCH_KEY_FORBIDDEN"):
        RetryOrchestrator(lambda action: {}).execute(
            {
                "action_id": "a",
                "execution_mode": "SIMULATION",
                "scenario": {"scenario_id": "s", "seed": 1},
                "parameters": {},
            },
            {"disable_sandbox": True},
            retry_budget=1,
        )


def test_retry_rejects_scenario_drift_and_repeated_patch():
    original = {
        "action_id": "a",
        "execution_mode": "SIMULATION",
        "scenario": {"scenario_id": "s", "seed": 1},
        "parameters": {},
    }
    orchestrator = RetryOrchestrator(lambda action: {"scenario_id": "different", "seed": 99})
    drift = orchestrator.execute(original, {"timeout_sec": 2.0}, retry_budget=2)
    assert drift.executed is False
    assert drift.reason == "RETRY_SCENARIO_DRIFT"
    repeated = orchestrator.execute(
        original,
        {"timeout_sec": 2.0},
        retry_budget=2,
        prior_patch_hashes=[drift.parameter_patch_hash],
    )
    assert repeated.reason == "REPEATED_PATCH"


def test_retry_never_submits_real_or_unverified_evidence():
    submitted = []
    original = {
        "action_id": "a",
        "execution_mode": "REAL",
        "scenario": {"scenario_id": "s", "seed": 1},
        "parameters": {},
    }
    blocked = RetryOrchestrator(lambda action: submitted.append(action)).execute(
        original, {"speed_scale": 0.5}, retry_budget=1
    )
    assert blocked.reason == "RETRY_SIMULATION_ONLY"
    assert submitted == []

    original["execution_mode"] = "SIMULATION"
    unverified = RetryOrchestrator(
        lambda action: {
            "scenario_id": "s",
            "seed": 1,
            "physics_executed": True,
            "evidence_domain": "SIMULATION",
        }
    ).execute(original, {"speed_scale": 0.5}, retry_budget=1)
    assert unverified.executed is False
    assert unverified.reason == "RETRY_EVIDENCE_UNVERIFIED"

    fabricated = RetryOrchestrator(
        lambda action: {
            "scenario_id": "s",
            "seed": 1,
            "physics_executed": True,
            "evidence_domain": "SIMULATION",
            "receipt_verified": True,
            "data_quality_pass": True,
            "replay_report": {
                "verified": True,
                "environment_match": True,
                "hashes_verified": True,
                "deterministic_label": True,
                "mismatches": [],
            },
        }
    ).execute(original, {"speed_scale": 0.5}, retry_budget=1)
    assert fabricated.executed is False
    assert fabricated.reason == "RETRY_EVIDENCE_UNVERIFIED"


@pytest.mark.parametrize(
    "patch",
    [
        {"speed_scale": 0.0},
        {"timeout_sec": float("nan")},
        {"retry_count": True},
        {"waypoint_offset": [1.0]},
        {"approach_direction": "disable_clearance"},
    ],
)
def test_retry_rejects_out_of_bounds_patch_values(patch):
    with pytest.raises(ValueError, match="RETRY_PATCH_VALUE_INVALID"):
        RetryOrchestrator(lambda action: {}).execute(
            {
                "action_id": "a",
                "execution_mode": "SIMULATION",
                "scenario": {"scenario_id": "s", "seed": 1},
                "parameters": {},
            },
            patch,
            retry_budget=1,
        )


@pytest.mark.parametrize("budget", [True, 0, 11, 1.5, "2"])
def test_retry_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="RETRY_BUDGET_INVALID"):
        RetryOrchestrator(lambda _action: {}).execute(
            {
                "action_id": "a",
                "execution_mode": "SIMULATION",
                "scenario": {"scenario_id": "s", "seed": 1},
                "parameters": {},
            },
            {"speed_scale": 0.5},
            retry_budget=budget,
        )


@pytest.mark.parametrize("seed", [True, -1, 1.5, "42"])
def test_retry_rejects_invalid_scenario_seed(seed):
    result = RetryOrchestrator(lambda _action: {}).execute(
        {
            "action_id": "a",
            "execution_mode": "SIMULATION",
            "scenario": {"scenario_id": "s", "seed": seed},
            "parameters": {},
        },
        {"speed_scale": 0.5},
        retry_budget=1,
    )
    assert result.reason == "SCENARIO_AND_SEED_INVALID"


def test_retry_rejects_malformed_lineage():
    with pytest.raises(ValueError, match="RETRY_LINEAGE_INVALID"):
        RetryOrchestrator(lambda _action: {}).execute(
            {
                "action_id": "a",
                "execution_mode": "SIMULATION",
                "scenario": {"scenario_id": "s", "seed": 1},
                "parameters": {},
            },
            {"speed_scale": 0.5},
            retry_budget=1,
            retry_lineage="not-a-list",
        )
