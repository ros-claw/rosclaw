"""Retry orchestration must execute a same-seed, whitelisted patch."""

import pytest

from rosclaw.how.retry_orchestrator import RetryOrchestrator


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
        }

    original = {
        "action_id": "action_original",
        "scenario": {"scenario_id": "shield-42", "seed": 42, "friction": 0.7},
        "parameters": {"speed_scale": 1.0},
    }
    result = RetryOrchestrator(submit).execute(
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
                "scenario": {"scenario_id": "s", "seed": 1},
                "parameters": {},
            },
            {"disable_sandbox": True},
            retry_budget=1,
        )


def test_retry_rejects_scenario_drift_and_repeated_patch():
    original = {
        "action_id": "a",
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
