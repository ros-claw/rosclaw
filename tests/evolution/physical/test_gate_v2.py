"""PR-PE-7 tests: candidate state machine v2, confirmation gate,
VALIDATED envelope builder."""

from __future__ import annotations

import pytest

from rosclaw.evolution.physical.candidate_state import (
    CandidateLifecycle,
    CandidateStateV2,
    IllegalTransitionError,
    migrate_v1_state,
)
from rosclaw.evolution.physical.envelope_v2 import (
    ExecutionEvidence,
    build_validated_envelope,
)
from rosclaw.evolution.physical.promotion_v2 import (
    ConfirmationVerdict,
    RegimeBlock,
    evaluate_confirmation,
)


def test_state_machine_happy_path() -> None:
    life = CandidateLifecycle("cand_x")
    for to in (
        CandidateStateV2.SCHEMA_VALIDATED,
        CandidateStateV2.SAFETY_VALIDATED,
        CandidateStateV2.SHADOW_VALIDATED,
        CandidateStateV2.CANARY_APPROVED,
        CandidateStateV2.PROVISIONALLY_PROMOTED,
        CandidateStateV2.CONFIRMATION_PENDING,
        CandidateStateV2.VALIDATED_EFFECTIVE,
        CandidateStateV2.ACTIVE,
    ):
        life.advance(to, evidence=f"evidence for {to.value}")
    assert life.state == CandidateStateV2.ACTIVE
    assert len(life.history) == 8


def test_exploratory_cannot_jump_to_effective() -> None:
    life = CandidateLifecycle("cand_x")
    with pytest.raises(IllegalTransitionError, match="not an allowed"):
        life.advance(CandidateStateV2.VALIDATED_EFFECTIVE, evidence="shortcut")
    life.advance(CandidateStateV2.SCHEMA_VALIDATED, evidence="schema ok")
    with pytest.raises(IllegalTransitionError):
        life.advance(CandidateStateV2.ACTIVE, evidence="another shortcut")
    # Provisional must pass through CONFIRMATION_PENDING.
    for to in (
        CandidateStateV2.SAFETY_VALIDATED,
        CandidateStateV2.SHADOW_VALIDATED,
        CandidateStateV2.CANARY_APPROVED,
        CandidateStateV2.PROVISIONALLY_PROMOTED,
    ):
        life.advance(to, evidence="ok")
    with pytest.raises(IllegalTransitionError):
        life.advance(CandidateStateV2.ACTIVE, evidence="skip confirmation")


def test_v1_migration() -> None:
    assert migrate_v1_state("validated") == CandidateStateV2.CANARY_APPROVED
    assert migrate_v1_state("promoted") == CandidateStateV2.CONFIRMATION_PENDING
    assert migrate_v1_state("rolled_back") == CandidateStateV2.ROLLED_BACK
    with pytest.raises(ValueError):
        migrate_v1_state("nonsense")


def _block(
    diff: float, *, regime: str = "warm", window: str = "w1", safety: int = 0
) -> RegimeBlock:
    a = 0.30
    return RegimeBlock(
        regime_bin=regime,
        arm_a_invalid=a,
        arm_c_invalid=a - diff,
        start_temp_delta_c=1.0,
        time_window=window,
        safety_events=safety,
    )


def test_confirmation_validated_effective() -> None:
    blocks = [
        _block(0.08, window="w1"),
        _block(0.10, window="w2"),
        _block(0.06, window="w2"),
        _block(0.12, regime="hot_but_safe", window="w3"),
    ]
    report = evaluate_confirmation(blocks)
    assert report.verdict == ConfirmationVerdict.VALIDATED_EFFECTIVE
    assert report.effect_pp == pytest.approx(9.0)
    assert report.ci_low > 0
    assert report.recurrence_sessions == 4
    assert report.time_windows == 3


def test_confirmation_insufficient_when_thin() -> None:
    blocks = [_block(0.10, window="w1"), _block(0.08, window="w1")]
    report = evaluate_confirmation(blocks)
    assert report.verdict == ConfirmationVerdict.INSUFFICIENT_EVIDENCE
    assert any("recurrence_sessions" in f for f in report.failed_checks)
    assert any("time_windows" in f for f in report.failed_checks)


def test_confirmation_refuted_when_effect_fails() -> None:
    blocks = [
        _block(0.01, window="w1"),
        _block(-0.01, window="w2"),
        _block(0.02, window="w3"),
    ]
    report = evaluate_confirmation(blocks)
    assert report.verdict == ConfirmationVerdict.REFUTED
    assert any("practical_effect" in f for f in report.failed_checks)


def test_confirmation_zero_tolerance_safety_and_ci() -> None:
    blocks = [_block(0.10, window="w1", safety=1)] + [_block(0.10, window="w2")] * 2
    report = evaluate_confirmation(blocks)
    assert report.verdict == ConfirmationVerdict.REFUTED
    assert any("safety_events=1" in f for f in report.failed_checks)

    spread = [
        _block(0.20, window="w1"),
        _block(-0.05, window="w2"),
        _block(0.15, window="w3"),
    ]
    report2 = evaluate_confirmation(spread)
    assert any("paired_ci" in f for f in report2.failed_checks)


def test_cold_regime_harm_blocks() -> None:
    blocks = [
        _block(0.20, window="w1"),
        _block(0.20, window="w2"),
        _block(-0.05, regime="cold", window="w3"),
    ]
    report = evaluate_confirmation(blocks, cold_blocks=[blocks[-1]])
    assert report.verdict == ConfirmationVerdict.REFUTED
    assert any("cold_regime_harm" in f for f in report.failed_checks)


def _evidence(
    practice: str,
    *,
    helpful: bool = True,
    recurrence: bool = False,
    safety: int = 0,
    cand_hash: str = "cand_a",
    body_hash: str = "body_a",
    cal_hash: str = "cal_a",
) -> ExecutionEvidence:
    return ExecutionEvidence(
        practice_id=practice,
        candidate_hash=cand_hash,
        body_hash=body_hash,
        calibration_hash=cal_hash,
        critic_helpful=helpful,
        safety_events=safety,
        is_recurrence=recurrence,
        temperature_range=(44.0, 50.0),
        recent_failure_rate=0.25,
    )


def test_validated_envelope_happy_path() -> None:
    evidence = [
        _evidence("prac_1"),
        _evidence("prac_2"),
        _evidence("prac_3", recurrence=True),
    ]
    report = build_validated_envelope("mem_x", evidence)
    assert report.ok
    assert report.envelope is not None
    assert report.envelope.envelope_type == "validated"
    assert report.envelope.evidence_count == 3
    assert report.envelope.temperature_min == 44.0
    assert report.envelope.confidence == 0.9


def test_single_session_never_validates() -> None:
    report = build_validated_envelope("mem_x", [_evidence("prac_1", recurrence=True)])
    assert not report.ok
    assert any("critic_helpful_sessions=1" in f for f in report.failed_requirements)


def test_validated_requires_recurrence_and_hash_consistency() -> None:
    no_recurrence = build_validated_envelope("mem_x", [_evidence("prac_1"), _evidence("prac_2")])
    assert not no_recurrence.ok
    assert any("recurrence_sessions=0" in f for f in no_recurrence.failed_requirements)

    drifted = build_validated_envelope(
        "mem_x",
        [
            _evidence("prac_1"),
            _evidence("prac_2", cand_hash="cand_b"),
            _evidence("prac_3", recurrence=True),
        ],
    )
    assert not drifted.ok
    assert any("candidate_hash inconsistent" in f for f in drifted.failed_requirements)

    unsafe = build_validated_envelope(
        "mem_x",
        [_evidence("prac_1", safety=1), _evidence("prac_2"), _evidence("prac_3", recurrence=True)],
    )
    assert not unsafe.ok
    assert any("safety_events=1" in f for f in unsafe.failed_requirements)
