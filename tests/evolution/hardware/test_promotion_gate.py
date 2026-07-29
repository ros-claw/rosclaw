"""Promotion gate tests (PR-EVO-HW-4 §Phase 7)."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.evolution.hardware.promotion_gate import (
    PromotionDecision,
    evaluate_promotion_gate,
    promoted_rule_record,
)


@dataclass
class _Rec:
    invalid_rate: float
    verified_rate: float


def _arms(a_invalid, b_invalid, c_invalid):
    return {
        "A_no_memory": [_Rec(a_invalid, 1 - a_invalid) for _ in range(3)],
        "B_fixed_cooldown": [_Rec(b_invalid, 1 - b_invalid) for _ in range(3)],
        "C_candidate_canary": [_Rec(c_invalid, 1 - c_invalid) for _ in range(3)],
    }


SAFETY_ZERO = {
    "unsafe_action": 0,
    "protection_event": 0,
    "wrong_body": 0,
    "wrong_joint": 0,
    "wrong_regime": 0,
    "choreography_violation": 0,
    "memory_hurt": 0.0,
}

PROOFS = [
    {
        "suggested_patch": {"inter_round_cooldown_sec": 2.0},
        "actual_patch": {"inter_round_cooldown_sec": 2.0},
        "patch_applied": True,
        "critic_decision": "recovered",
    }
]


def _gate(arm_records, safety=None, proofs=None, config=None):
    return evaluate_promotion_gate(
        candidate_id="cand_x",
        arm_records=arm_records,
        safety=safety or dict(SAFETY_ZERO),
        patch_proofs=PROOFS if proofs is None else proofs,
        promotion_config=config or {},
        stats_fn=None,
    )


def test_promotes_when_all_checks_pass() -> None:
    report = _gate(_arms(0.20, 0.10, 0.05))
    assert report.decision is PromotionDecision.PROMOTED
    assert report.scope == "operator_approved_recurrence"
    assert all(c.passed for c in report.checks)


def test_not_promoted_when_c_not_better_than_a() -> None:
    report = _gate(_arms(0.05, 0.10, 0.20))
    assert report.decision is PromotionDecision.NOT_PROMOTED
    failed = {c.name for c in report.checks if not c.passed}
    assert "c_invalid_lt_a" in failed


def test_not_promoted_when_c_worse_than_b() -> None:
    report = _gate(_arms(0.20, 0.03, 0.10))
    assert report.decision is PromotionDecision.NOT_PROMOTED
    failed = {c.name for c in report.checks if not c.passed}
    assert "c_not_worse_than_b" in failed


def test_safety_violation_rolls_back() -> None:
    safety = dict(SAFETY_ZERO, wrong_regime=1)
    report = _gate(_arms(0.20, 0.10, 0.05), safety=safety)
    assert report.decision is PromotionDecision.ROLLED_BACK
    assert report.scope == "none"


def test_memory_hurt_threshold() -> None:
    safety = dict(SAFETY_ZERO, memory_hurt=0.5)
    report = _gate(_arms(0.20, 0.10, 0.05), safety=safety)
    assert report.decision is PromotionDecision.NOT_PROMOTED
    failed = {c.name for c in report.checks if not c.passed}
    assert "memory_hurt" in failed


def test_incomplete_proofs_block_promotion() -> None:
    report = _gate(_arms(0.20, 0.10, 0.05), proofs=[])
    assert report.decision is PromotionDecision.NOT_PROMOTED
    incomplete = [{"suggested_patch": {}, "actual_patch": None, "patch_applied": True, "critic_decision": None}]
    report2 = _gate(_arms(0.20, 0.10, 0.05), proofs=incomplete)
    assert report2.decision is PromotionDecision.NOT_PROMOTED


def test_promoted_rule_record_shape() -> None:
    report = _gate(_arms(0.20, 0.10, 0.05))
    rule = promoted_rule_record(
        candidate={"candidate_id": "cand_x", "changes": {"inter_round_cooldown_sec": 2.0}},
        gate_report=report,
        canary_sessions=["prac_1"],
    )
    assert rule["rule_id"] == "rule_cand_x"
    assert rule["scope"] == "operator_approved_recurrence"
    assert rule["status"] == "active"
    assert rule["canary_sessions"] == ["prac_1"]
    assert rule["gate_report"]["decision"] == "PROMOTED"


def test_min_sessions_floor_blocks_single_lucky_session() -> None:
    """One lucky session never promotes (§Phase 6 pilot: 3/arm)."""
    records = {
        "A_no_memory": [_Rec(0.20, 0.80)],
        "B_fixed_cooldown": [_Rec(0.25, 0.75)],
        "C_candidate_canary": [_Rec(0.175, 0.825)],
    }
    report = _gate(records)
    assert report.decision is PromotionDecision.NOT_PROMOTED
    failed = {c.name for c in report.checks if not c.passed}
    assert "min_sessions_c" in failed
    # With 3 sessions per arm the same metrics promote.
    full = _arms(0.20, 0.25, 0.175)
    report2 = _gate(full)
    assert report2.decision is PromotionDecision.PROMOTED


def test_other_arm_abort_is_disclosed_not_candidate_fault() -> None:
    """An abort in arm B is an experiment-condition note, never a
    protection_event against the candidate (found 2026-07-26)."""
    report = evaluate_promotion_gate(
        candidate_id="cand_x",
        arm_records=_arms(0.20, 0.25, 0.175),
        safety=dict(SAFETY_ZERO),
        patch_proofs=PROOFS,
        promotion_config={},
        stats_fn=None,
        other_arm_aborts=1,
    )
    # Metrics pass; the other-arm abort appears only as disclosure.
    assert report.decision is PromotionDecision.PROMOTED
    detail = next(c.detail for c in report.checks if c.name == "min_sessions_c")
    assert "matrix incomplete" in detail


def test_repropose_preserves_lifecycle_states() -> None:
    """Re-running propose must never clobber VALIDATED/PROMOTED/terminal
    states (same bug class as the promote() reset found 2026-07-26)."""
    from rosclaw.evolution.hardware.promotion import (
        COLLECTION,
        CandidateRegistry,
        CandidateState,
    )
    from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore

    store = InMemoryKnowledgeStore()
    store.connect()
    registry = CandidateRegistry(store)
    # Simulate an existing VALIDATED row.
    store.insert(
        COLLECTION,
        {
            "id": "cand_x", "candidate_id": "cand_x", "experiment_id": "exp",
            "changes": {}, "state": "VALIDATED", "gate_verdicts": [{"gate": "schema", "passed": True}],
        },
    )
    from rosclaw.evolution.hardware.orchestrator import EvoRpsOrchestrator  # noqa: F401
    from rosclaw.evolution.hardware.promotion import CandidateRecord

    # The propose() preservation logic: load before upsert.
    existing = registry.get("cand_x")
    record = CandidateRecord(
        candidate_id="cand_x", experiment_id="exp", changes={}, source_failure="f", current_regime="r"
    )
    if existing is not None:
        record.state = CandidateState(str(existing["state"]))
        record.gate_verdicts = list(existing.get("gate_verdicts") or [])
    registry.upsert(record)
    fetched = registry.get("cand_x")
    assert fetched["state"] == "VALIDATED"
    assert fetched["gate_verdicts"] == [{"gate": "schema", "passed": True}]


def test_generation_arm_aborts_scopes_arm_and_generation() -> None:
    """Real-campaign regression (2026-07-29): cand_004's C-arm abort from
    the PREVIOUS generation inflated cand_005's protection_event to 2.
    Sessions were generation-scoped; the abort list was not."""
    from rosclaw.evolution.hardware.orchestrator import generation_arm_aborts

    aborted = [
        {"arm": "C_candidate_canary", "recorded_at": 100.0},  # previous generation
        {"arm": "C_candidate_canary", "recorded_at": 200.0},  # current generation
        {"arm": "A_no_memory", "recorded_at": 210.0},  # current generation, wrong arm
    ]
    current = generation_arm_aborts(aborted, 150.0, "C_candidate_canary")
    assert len(current) == 1
    assert current[0]["recorded_at"] == 200.0
    # No generation start → everything in the arm matches.
    assert len(generation_arm_aborts(aborted, 0.0, "C_candidate_canary")) == 2
    # Missing recorded_at never silently counts as current.
    legacy = [{"arm": "C_candidate_canary"}]
    assert generation_arm_aborts(legacy, 150.0, "C_candidate_canary") == []
