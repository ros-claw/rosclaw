from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.mission import MissionStore
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.agentd_bridge import stage_growth_evaluation_candidate


def _evidence(*, passed: bool) -> dict:
    status = "SIM_GATE_PASS" if passed else "REJECTED_BY_SIM_GATE"
    cases = []
    for index in range(8):
        case_passed = passed or index != 6
        cases.append(
            {
                "spec": {"case_id": f"case-{index}", "partition": "generalization"},
                "passed": case_passed,
                "parent_strict_replay": True,
                "candidate_strict_replay": True,
                "non_regression_gate": {
                    "reasons": [] if case_passed else ["jerk_regression"]
                },
            }
        )
    return {
        "schema_version": "rosclaw.growth.g1_residual_recovery_evidence.v1",
        "passed": passed,
        "status": status,
        "activation_ceiling": "SIM_ONLY",
        "evidence_domain": "SIM",
        "activation_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
        "hardware_command_sent": False,
        "candidate_hash": "sha256:" + "1" * 64,
        "request_hash": "sha256:" + "2" * 64,
        "environment_hash": "sha256:" + "3" * 64,
        "implementation_hash": "sha256:" + "4" * 64,
        "cases": cases,
        "development_aggregate_gate": {"passed": passed},
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _failure_curriculum_evidence() -> dict:
    def row(case_id: str, purpose: str, *, success: bool, abstained: bool) -> dict:
        return {
            "case_id": case_id,
            "purpose": purpose,
            "strict_replay": True,
            "quality_accepted": True,
            "critical": False,
            "success": success,
            "abstained": abstained,
            "result": {
                "status": "SUCCESS" if success else "ROBOT_NOT_STABLE",
                "physics_executed": not abstained,
            },
        }

    value = {
        "schema_version": "rosclaw.simforge.g1_failure_curriculum_report.v3",
        "decision": "SIM_CANDIDATE",
        "gate_reasons": [],
        "activation_ceiling": "SIM_ONLY",
        "evidence_domain": "SHADOW",
        "body_hash": "sha256:" + "1" * 64,
        "kick_prior_hash": "sha256:" + "2" * 64,
        "frozen_policy_hash": "sha256:" + "3" * 64,
        "curriculum_commitment": "sha256:" + "4" * 64,
        "validation": [
            row("validation-a", "validation", success=True, abstained=False),
            row("validation-b", "validation", success=True, abstained=False),
            row("validation-risk", "validation", success=False, abstained=True),
        ],
        "holdout": [
            row("holdout-risk", "holdout", success=False, abstained=True),
        ],
        "learning_contract": {
            "failed_rollouts_train_actor": False,
            "sealed_cases_changed_policy": False,
        },
        "calibration_contract": {
            "exact_hidden_dynamics_exposed": False,
            "abstention_counts_as_success": False,
        },
    }
    value["report_hash"] = canonical_hash(value)
    return value


def test_passing_sim_stages_how_without_promotion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "evidence.json"
    _write(evidence, _evidence(passed=True))
    store = MissionStore(tmp_path / "agentd.db")
    try:
        receipt = stage_growth_evaluation_candidate(
            evaluation_path=evidence,
            connection=store.connection,
            source_checkout=source,
        )
        row = store.connection.execute(
            "SELECT * FROM learning_candidates WHERE candidate_id = ?",
            (receipt.candidate_id,),
        ).fetchone()
        assert receipt.learning_kind == "HOW"
        assert receipt.promotion_authorized is False
        assert row["status"] == "CANDIDATE"
        assert row["evidence_class"] == "measured"
        assert json.loads(row["content_json"])["deployable"] is False
    finally:
        store.close()


def test_failed_sim_is_idempotent_negative_memory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "evidence.json"
    _write(evidence, _evidence(passed=False))
    store = MissionStore(tmp_path / "agentd.db")
    try:
        first = stage_growth_evaluation_candidate(
            evaluation_path=evidence,
            connection=store.connection,
            source_checkout=source,
        )
        second = stage_growth_evaluation_candidate(
            evaluation_path=evidence,
            connection=store.connection,
            source_checkout=source,
        )
        row = store.connection.execute(
            "SELECT * FROM learning_candidates WHERE candidate_id = ?",
            (first.candidate_id,),
        ).fetchone()
        content = json.loads(row["content_json"])
        assert first.learning_kind == "MEMORY"
        assert first.disposition == "NEGATIVE_SIM_MEMORY"
        assert second.candidate_id == first.candidate_id
        assert second.staged is False
        assert content["failed_cases"] == [
            {
                "case_id": "case-6",
                "partition": "generalization",
                "reasons": ["jerk_regression"],
            }
        ]
    finally:
        store.close()


def test_bridge_rejects_hardware_or_evidence_inside_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inside = source / "evidence.json"
    value = _evidence(passed=True)
    _write(inside, value)
    store = MissionStore(tmp_path / "agentd.db")
    try:
        with pytest.raises(ValueError, match="outside"):
            stage_growth_evaluation_candidate(
                evaluation_path=inside,
                connection=store.connection,
                source_checkout=source,
            )
        outside = tmp_path / "outside.json"
        value["hardware_command_sent"] = True
        _write(outside, value)
        with pytest.raises(ValueError, match="hardware_command_sent=false"):
            stage_growth_evaluation_candidate(
                evaluation_path=outside,
                connection=store.connection,
                source_checkout=source,
            )
    finally:
        store.close()


def test_bridge_rejects_inconsistent_residual_aggregate_gate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "evidence.json"
    value = _evidence(passed=True)
    value["development_aggregate_gate"] = {"passed": False}
    _write(evidence, value)
    store = MissionStore(tmp_path / "agentd.db")
    try:
        with pytest.raises(ValueError, match="committed gates"):
            stage_growth_evaluation_candidate(
                evaluation_path=evidence,
                connection=store.connection,
                source_checkout=source,
            )
    finally:
        store.close()


def test_failure_curriculum_stages_risk_aware_how_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "failure-curriculum.json"
    _write(evidence, _failure_curriculum_evidence())
    store = MissionStore(tmp_path / "agentd.db")
    try:
        receipt = stage_growth_evaluation_candidate(
            evaluation_path=evidence,
            connection=store.connection,
            source_checkout=source,
        )
        row = store.connection.execute(
            "SELECT * FROM learning_candidates WHERE candidate_id = ?",
            (receipt.candidate_id,),
        ).fetchone()
        content = json.loads(row["content_json"])
        assert receipt.learning_kind == "HOW"
        assert receipt.evaluation_status == "SIM_GATE_PASS"
        assert content["case_count"] == 4
        assert content["failed_cases"] == []
        assert content["activation_authorized"] is False
    finally:
        store.close()


def test_failure_curriculum_bridge_recomputes_hash_and_gates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "failure-curriculum.json"
    value = _failure_curriculum_evidence()
    value["validation"][0]["critical"] = True
    _write(evidence, value)
    store = MissionStore(tmp_path / "agentd.db")
    try:
        with pytest.raises(ValueError, match="hash mismatch"):
            stage_growth_evaluation_candidate(
                evaluation_path=evidence,
                connection=store.connection,
                source_checkout=source,
            )
    finally:
        store.close()
