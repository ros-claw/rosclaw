"""VALIDATED applicability envelope builder (Physical Evolution Lab §9.7, PR-PE-7).

OBSERVED envelopes say "this memory was seen in this regime".
VALIDATED envelopes say "this intervention was EXECUTED + critic-passed
in this regime, repeatedly, across independent sessions, without safety
events" — the ONLY upgrade path (v3 §9.7):

    candidate really executed on hardware
    + critic judged helpful
    + multiple sessions
    + independent recurrence reproduced
    + zero safety events
    + zero wrong body / joint / regime
    + candidate/body/calibration hashes consistent

A single session can NEVER produce a VALIDATED envelope — and a memory
can never certify itself (v3 §16.7: 禁止单场 Session 生成 VALIDATED
Envelope).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.memory.v2.regime.envelope import ApplicabilityEnvelope, EnvelopeType, new_envelope_id

BUILDER_VERSION = "rosclaw.validated_envelope.v1"
MIN_EXECUTION_SESSIONS = 2
MIN_RECURRENCE_SESSIONS = 1


@dataclass(frozen=True)
class ExecutionEvidence:
    """One real-hardware execution of the candidate with critic verdict."""

    practice_id: str
    candidate_hash: str
    body_hash: str
    calibration_hash: str
    critic_helpful: bool
    safety_events: int = 0
    wrong_body: int = 0
    wrong_joint: int = 0
    wrong_regime: int = 0
    temperature_range: tuple[float | None, float | None] = (None, None)
    recent_failure_rate: float | None = None
    is_recurrence: bool = False


@dataclass
class ValidatedEnvelopeReport:
    ok: bool
    envelope: ApplicabilityEnvelope | None
    failed_requirements: list[str] = field(default_factory=list)
    evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "builder_version": BUILDER_VERSION,
            "ok": self.ok,
            "failed_requirements": self.failed_requirements,
            "evidence_count": self.evidence_count,
            "envelope_id": self.envelope.envelope_id if self.envelope else None,
        }


def build_validated_envelope(
    memory_id: str,
    evidence: list[ExecutionEvidence],
    *,
    gestures: list[str] | None = None,
    failure_types: list[str] | None = None,
    body_ids: list[str] | None = None,
) -> ValidatedEnvelopeReport:
    """The ONLY VALIDATED upgrade path — every requirement named."""
    failed: list[str] = []

    executed = [e for e in evidence if e.critic_helpful]
    if len(executed) < MIN_EXECUTION_SESSIONS:
        failed.append(f"critic_helpful_sessions={len(executed)} < {MIN_EXECUTION_SESSIONS}")

    recurrence = [e for e in executed if e.is_recurrence]
    if len(recurrence) < MIN_RECURRENCE_SESSIONS:
        failed.append(f"recurrence_sessions={len(recurrence)} < {MIN_RECURRENCE_SESSIONS}")

    safety = sum(e.safety_events for e in evidence)
    if safety:
        failed.append(f"safety_events={safety} (max 0)")
    for label, count in (
        ("wrong_body", sum(e.wrong_body for e in evidence)),
        ("wrong_joint", sum(e.wrong_joint for e in evidence)),
        ("wrong_regime", sum(e.wrong_regime for e in evidence)),
    ):
        if count:
            failed.append(f"{label}={count} (max 0)")

    # Hash consistency: every session must agree on candidate/body/cal.
    for field_name in ("candidate_hash", "body_hash", "calibration_hash"):
        values = {getattr(e, field_name) for e in evidence}
        if len(values) > 1:
            failed.append(f"{field_name} inconsistent across sessions: {sorted(values)}")

    if failed:
        return ValidatedEnvelopeReport(ok=False, envelope=None, failed_requirements=failed)

    temps = [t for e in executed for t in e.temperature_range if t is not None]
    rates = [e.recent_failure_rate for e in executed if e.recent_failure_rate is not None]
    envelope = ApplicabilityEnvelope(
        memory_id=memory_id,
        envelope_id=new_envelope_id(),
        body_ids=body_ids or sorted({e.body_hash for e in executed}),
        gestures=gestures or [],
        failure_types=failure_types or [],
        temperature_min=min(temps) if temps else None,
        temperature_max=max(temps) if temps else None,
        recent_failure_rate_min=min(rates) if rates else None,
        recent_failure_rate_max=max(rates) if rates else None,
        envelope_type=EnvelopeType.VALIDATED.value,
        evidence_count=len(executed),
        success_count=len(executed),
        confidence=0.9,
        reason=f"{BUILDER_VERSION}: executed+critic+recurrence validated",
        evidence_refs=[e.practice_id for e in executed],
        created_at=time.time(),
        updated_at=time.time(),
    )
    return ValidatedEnvelopeReport(
        ok=True, envelope=envelope, failed_requirements=[], evidence_count=len(executed)
    )
