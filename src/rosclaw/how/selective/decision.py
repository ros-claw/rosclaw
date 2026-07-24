"""Selective intervention decisions (数据库优化v4 §7, PR-HOW-3).

The system is NOT required to answer every failure with a patch.  It acts
only when evidence, regime match, and safety constraints are sufficient —
otherwise it ABSTAINS (or ESCALATES).  ABSTAIN is a first-class, auditable
decision, not the absence of one.

Note the deliberate name: :class:`SelectiveInterventionDecision` — the HOW
intervention package already has an ``InterventionDecision`` for runtime
strategy; this one is the memory-driven APPLY/SUGGEST/ABSTAIN/ESCALATE
verdict of the selective pipeline.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InterventionAction(StrEnum):
    """v4 §7.1 — the four selective actions."""

    APPLY = "APPLY"
    SUGGEST = "SUGGEST"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


# Reason codes (v4 §7.3 forced-ABSTAIN conditions + pipeline states).
REASON_NO_SAME_BODY_MEMORY = "no_same_body_memory"
REASON_NO_SAME_JOINT_MEMORY = "no_same_joint_memory"
REASON_REGIME_FEATURE_MISSING = "regime_required_feature_missing"
REASON_REGIME_SCORE_LOW = "regime_score_below_threshold"
REASON_CONTRAINDICATED = "contraindicated_envelope_hit"
REASON_CONFLICTING_MEMORIES = "conflicting_memories"
REASON_TOP12_CLOSE_DIFFERENT = "top1_top2_close_but_different"
REASON_NO_PATCHPROOF = "no_patchproof_success_evidence"
REASON_CONTROL_PROFILE_MISMATCH = "control_profile_hash_mismatch"
REASON_PROVIDER_DEGRADED_HIGH_RISK = "embedding_provider_degraded_high_risk"
REASON_SANDBOX_UNAVAILABLE = "sandbox_unavailable"
REASON_CHOREOGRAPHY_UNAVAILABLE = "choreography_validator_unavailable"
REASON_NO_CANDIDATE = "no_retrieval_candidate"
REASON_VALIDATED_MATCH = "validated_regime_match"
REASON_SUGGEST_BAND = "applicability_suggest_band"
REASON_SAFETY_ESCALATION = "safety_severity_escalation"


@dataclass
class SelectiveInterventionDecision:
    """The pipeline verdict (v4 §7.1)."""

    decision_id: str
    action: InterventionAction

    failure_signature: str
    selected_memory_id: str | None
    selected_rule_id: str | None

    retrieval_confidence: float
    applicability_score: float
    regime_confidence: float
    evidence_confidence: float

    expected_benefit: float
    estimated_harm: float
    uncertainty: float

    reason_codes: list[str] = field(default_factory=list)
    explanation: str = ""

    suggested_patch: dict[str, Any] | None = None
    safety_requirements: list[str] = field(default_factory=list)

    regime_label: str | None = None
    matched_envelope_id: str | None = None
    abstention_considered: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "action": self.action.value,
            "failure_signature": self.failure_signature,
            "selected_memory_id": self.selected_memory_id,
            "selected_rule_id": self.selected_rule_id,
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "applicability_score": round(self.applicability_score, 4),
            "regime_confidence": round(self.regime_confidence, 4),
            "evidence_confidence": round(self.evidence_confidence, 4),
            "expected_benefit": round(self.expected_benefit, 4),
            "estimated_harm": round(self.estimated_harm, 4),
            "uncertainty": round(self.uncertainty, 4),
            "reason_codes": self.reason_codes,
            "explanation": self.explanation,
            "suggested_patch": self.suggested_patch,
            "safety_requirements": self.safety_requirements,
            "regime_label": self.regime_label,
            "matched_envelope_id": self.matched_envelope_id,
            "abstention_considered": self.abstention_considered,
            "created_at": self.created_at,
        }


def new_decision_id() -> str:
    return f"dec_{uuid.uuid4().hex[:16]}"
