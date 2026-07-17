"""Tests for HOW rule efficacy (§11.1–11.3)."""

from __future__ import annotations

from rosclaw.how.rule_efficacy import (
    PatchProof,
    RecoveryRule,
    record_outcome_atomic,
    rule_rank_score,
    wilson_interval,
    wilson_lower_bound,
)
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore, SQLiteKnowledgeStore


def test_wilson_bound_prevents_tiny_sample_outranking() -> None:
    # 1/1 = 100% must NOT outrank 90/100 = 90%.
    assert wilson_lower_bound(90, 100) > wilson_lower_bound(1, 1)
    assert wilson_lower_bound(0, 0) == 0.0
    # Sanity: bounds within [0, 1] and monotone in successes.
    assert 0.0 <= wilson_lower_bound(1, 1) < wilson_lower_bound(90, 100) <= 1.0


def test_wilson_interval_structure() -> None:
    interval = wilson_interval(90, 100)
    assert interval["lower"] < 0.9 < interval["upper"]
    empty = wilson_interval(0, 0)
    assert empty == {"lower": 0.0, "upper": 1.0, "z": 1.96}


def test_recovery_rule_schema_roundtrip() -> None:
    rule = RecoveryRule(
        rule_id="rule_scissors_overcurrent",
        failure_signature="rh56:scissors:over_current",
        action_template={"kind": "angle_compensation", "joint": "index", "delta": -150},
        applicable_robot_types=["dual_rh56"],
        applicable_body_ids=["rh56_right"],
        safety_level="S1",
        success_count=90,
        failure_count=10,
        evidence_count=90,
    )
    record = rule.to_record()
    assert record["condition"] == "rh56:scissors:over_current"  # legacy compat
    assert record["success_rate"] == 0.9
    assert record["confidence_interval"]
    client = InMemoryKnowledgeStore()
    client.connect()
    client.insert("heuristic_rules", record)
    rows = client.query("heuristic_rules", {"id": "rule_scissors_overcurrent"})
    assert rows and rows[0]["failure_signature"] == "rh56:scissors:over_current"


def test_rule_rank_score() -> None:
    well = RecoveryRule("a", "sig", {}, success_count=90, failure_count=10)
    tiny = RecoveryRule("b", "sig", {}, success_count=1, failure_count=0)
    assert rule_rank_score(well) > rule_rank_score(tiny)


def test_record_outcome_atomic_sqlite(tmp_path) -> None:
    client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    client.connect()
    rule = RecoveryRule("r1", "sig", {})
    client.insert("heuristic_rules", rule.to_record())
    assert record_outcome_atomic(client, "r1", True, evidence_ref="evt_1")
    assert record_outcome_atomic(client, "r1", False)
    row = client.query("heuristic_rules", {"id": "r1"}, limit=1)[0]
    assert row["success_count"] == 1
    assert row["failure_count"] == 1
    assert row["evidence_count"] == 1  # only one evidence_ref given
    assert row["last_validated_at"] is not None
    client.disconnect()


def test_record_outcome_atomic_inmemory() -> None:
    client = InMemoryKnowledgeStore()
    client.connect()
    client.insert("heuristic_rules", RecoveryRule("r2", "sig", {}).to_record())
    assert record_outcome_atomic(client, "r2", True)
    row = client.query("heuristic_rules", {"id": "r2"}, limit=1)[0]
    assert row["success_count"] == 1


def test_patch_proof_lifecycle() -> None:
    proof = PatchProof(suggested_patch={"delta": -150})
    assert not proof.patch_applied
    proof.complete(
        actual_patch={"delta": -150},
        after_metrics={"verified": True, "joint_error_max": 42},
        critic_decision="success",
    )
    record = proof.to_record()
    assert record["patch_applied"]
    assert record["actual_patch"] == record["suggested_patch"]
    assert record["critic_decision"] == "success"
