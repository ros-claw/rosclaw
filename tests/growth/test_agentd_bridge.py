from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.mission import MissionStore
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
