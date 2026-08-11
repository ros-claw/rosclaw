"""Fail-closed, task-neutral evidence gates for bounded growth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth._validation import (
    require_hash,
    require_identifier,
    unique_identifiers,
)
from rosclaw.growth.contracts import EvidenceLevel, MetricDirection


def _strict_number(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


@dataclass(frozen=True)
class CandidateExecutionEvidence:
    """Bind one evidence record to the artifact that was actually evaluated."""

    candidate_artifact_hash: str
    evidence_hash: str
    evidence_level: EvidenceLevel
    physics_executed: bool
    strict_replay: bool
    independently_verified: bool
    execution_receipt_hash: str | None = None
    schema_version: str = "rosclaw.growth.candidate_execution_evidence.v1"

    def __post_init__(self) -> None:
        require_hash("candidate_artifact_hash", self.candidate_artifact_hash)
        require_hash("evidence_hash", self.evidence_hash)
        if not isinstance(self.evidence_level, EvidenceLevel):
            raise ValueError("evidence_level must be an EvidenceLevel")
        for label in ("physics_executed", "strict_replay", "independently_verified"):
            if not isinstance(getattr(self, label), bool):
                raise ValueError(f"{label} must be boolean")
        if self.execution_receipt_hash is not None:
            require_hash("execution_receipt_hash", self.execution_receipt_hash)
        if self.physics_executed and self.execution_receipt_hash is None:
            raise ValueError("physics-executed evidence requires an execution receipt")

    @property
    def is_executed_promotion_truth(self) -> bool:
        return (
            self.physics_executed
            and self.execution_receipt_hash is not None
            and self.evidence_level
            in {EvidenceLevel.EXECUTION_RECEIPT, EvidenceLevel.PHYSICS_REPLAY}
        )


class CandidateEvidenceStatus(StrEnum):
    PASSED = "passed"
    NEEDS_EXECUTION = "needs_execution"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CandidateEvidenceDecision:
    status: CandidateEvidenceStatus
    candidate_artifact_hash: str
    accepted_evidence_hashes: tuple[str, ...]
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.growth.candidate_evidence_decision.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, CandidateEvidenceStatus):
            raise ValueError("status must be a CandidateEvidenceStatus")
        require_hash("candidate_artifact_hash", self.candidate_artifact_hash)
        hashes = tuple(self.accepted_evidence_hashes)
        if len(hashes) != len(set(hashes)):
            raise ValueError("accepted_evidence_hashes must be unique")
        for value in hashes:
            require_hash("accepted_evidence_hashes", value)
        object.__setattr__(self, "accepted_evidence_hashes", hashes)
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(
                self.reasons,
                label="reasons",
                allow_empty=self.status is CandidateEvidenceStatus.PASSED,
            ),
        )
        if self.status is CandidateEvidenceStatus.PASSED and self.reasons:
            raise ValueError("a passed evidence decision cannot carry rejection reasons")
        if self.status is CandidateEvidenceStatus.PASSED and not hashes:
            raise ValueError("a passed evidence decision requires accepted evidence")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def activation_allowed(self) -> bool:
        return False

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "accepted_evidence_hashes": list(self.accepted_evidence_hashes),
            "reasons": list(self.reasons),
            "activation_allowed": self.activation_allowed,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class CandidateEvidenceGate:
    minimum_executed_receipts: int = 1
    require_strict_replay: bool = True
    require_independent_verification: bool = True
    schema_version: str = "rosclaw.growth.candidate_evidence_gate.v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_executed_receipts, bool)
            or not isinstance(self.minimum_executed_receipts, int)
            or not 1 <= self.minimum_executed_receipts <= 10_000
        ):
            raise ValueError("minimum_executed_receipts must be in [1, 10000]")
        if not isinstance(self.require_strict_replay, bool) or not isinstance(
            self.require_independent_verification, bool
        ):
            raise ValueError("candidate evidence gate flags must be boolean")

    def evaluate(
        self,
        *,
        candidate_artifact_hash: str,
        evidence: tuple[CandidateExecutionEvidence, ...],
    ) -> CandidateEvidenceDecision:
        require_hash("candidate_artifact_hash", candidate_artifact_hash)
        reasons: list[str] = []
        if any(not isinstance(item, CandidateExecutionEvidence) for item in evidence):
            reasons.append("invalid_evidence_record")
        records = tuple(item for item in evidence if isinstance(item, CandidateExecutionEvidence))
        if any(item.candidate_artifact_hash != candidate_artifact_hash for item in records):
            reasons.append("candidate_hash_mismatch")
        bound = tuple(
            item for item in records if item.candidate_artifact_hash == candidate_artifact_hash
        )
        executed = tuple(item for item in bound if item.is_executed_promotion_truth)
        unique_executed = tuple({item.evidence_hash: item for item in executed}.values())
        if len(unique_executed) != len(executed):
            reasons.append("duplicate_evidence")
        if len(unique_executed) < self.minimum_executed_receipts:
            reasons.append("insufficient_executed_evidence")
        if self.require_strict_replay and any(not item.strict_replay for item in unique_executed):
            reasons.append("strict_replay_missing")
        if self.require_independent_verification and any(
            not item.independently_verified for item in unique_executed
        ):
            reasons.append("independent_verification_missing")

        hard_rejection_reasons = {
            "candidate_hash_mismatch",
            "duplicate_evidence",
            "independent_verification_missing",
            "invalid_evidence_record",
            "strict_replay_missing",
        }
        unique_reasons = tuple(dict.fromkeys(reasons))
        if not unique_reasons:
            status = CandidateEvidenceStatus.PASSED
        elif hard_rejection_reasons.intersection(unique_reasons):
            status = CandidateEvidenceStatus.REJECTED
        else:
            status = CandidateEvidenceStatus.NEEDS_EXECUTION
        return CandidateEvidenceDecision(
            status=status,
            candidate_artifact_hash=candidate_artifact_hash,
            accepted_evidence_hashes=tuple(item.evidence_hash for item in unique_executed),
            reasons=unique_reasons,
        )


@dataclass(frozen=True)
class RobustnessEvidence:
    artifact_hash: str
    metric_id: str
    direction: MetricDirection
    values: tuple[float, ...]
    evidence_hash: str
    safety_violation_count: int = 0
    schema_version: str = "rosclaw.growth.robustness_evidence.v1"

    def __post_init__(self) -> None:
        require_hash("artifact_hash", self.artifact_hash)
        require_identifier("metric_id", self.metric_id)
        if not isinstance(self.direction, MetricDirection):
            raise ValueError("direction must be a MetricDirection")
        values = tuple(_strict_number(value, label="robustness value") for value in self.values)
        if not values:
            raise ValueError("robustness values must be non-empty and finite")
        require_hash("evidence_hash", self.evidence_hash)
        if (
            isinstance(self.safety_violation_count, bool)
            or not isinstance(self.safety_violation_count, int)
            or self.safety_violation_count < 0
        ):
            raise ValueError("safety_violation_count must be a non-negative integer")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class RobustnessProfile:
    minimum_samples: int
    tail_fraction: float = 0.1
    maximum_worst_regression: float = 0.0
    maximum_cvar_regression: float = 0.0
    schema_version: str = "rosclaw.growth.robustness_profile.v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or not 2 <= self.minimum_samples <= 1_000_000
        ):
            raise ValueError("minimum_samples must be in [2, 1000000]")
        tail_fraction = _strict_number(self.tail_fraction, label="tail_fraction")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be finite and in (0, 1]")
        for label in ("maximum_worst_regression", "maximum_cvar_regression"):
            value = _strict_number(getattr(self, label), label=label)
            if value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
            object.__setattr__(self, label, value)
        object.__setattr__(self, "tail_fraction", tail_fraction)


class RobustnessStatus(StrEnum):
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RobustnessDecision:
    status: RobustnessStatus
    parent_worst_score: float
    candidate_worst_score: float
    parent_cvar_score: float
    candidate_cvar_score: float
    reasons: tuple[str, ...]
    parent_evidence_hash: str
    candidate_evidence_hash: str
    schema_version: str = "rosclaw.growth.robustness_decision.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, RobustnessStatus):
            raise ValueError("status must be a RobustnessStatus")
        for label in (
            "parent_worst_score",
            "candidate_worst_score",
            "parent_cvar_score",
            "candidate_cvar_score",
        ):
            if not math.isfinite(getattr(self, label)):
                raise ValueError(f"{label} must be finite")
        for label in ("parent_evidence_hash", "candidate_evidence_hash"):
            require_hash(label, getattr(self, label))
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(
                self.reasons,
                label="reasons",
                allow_empty=self.status is RobustnessStatus.PASSED,
            ),
        )
        if self.status is RobustnessStatus.PASSED and self.reasons:
            raise ValueError("a passed robustness decision cannot carry rejection reasons")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "parent_worst_score": self.parent_worst_score,
            "candidate_worst_score": self.candidate_worst_score,
            "parent_cvar_score": self.parent_cvar_score,
            "candidate_cvar_score": self.candidate_cvar_score,
            "reasons": list(self.reasons),
            "parent_evidence_hash": self.parent_evidence_hash,
            "candidate_evidence_hash": self.candidate_evidence_hash,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class RobustnessGate:
    profile: RobustnessProfile
    schema_version: str = "rosclaw.growth.robustness_gate.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.profile, RobustnessProfile):
            raise ValueError("profile must be a RobustnessProfile")

    def evaluate(
        self,
        *,
        parent: RobustnessEvidence,
        candidate: RobustnessEvidence,
    ) -> RobustnessDecision:
        if parent.metric_id != candidate.metric_id or parent.direction is not candidate.direction:
            raise ValueError("parent and candidate robustness evidence must describe one metric")
        parent_scores = _oriented_scores(parent.values, parent.direction)
        candidate_scores = _oriented_scores(candidate.values, candidate.direction)
        parent_worst = min(parent_scores)
        candidate_worst = min(candidate_scores)
        parent_cvar = _lower_tail_cvar(parent_scores, self.profile.tail_fraction)
        candidate_cvar = _lower_tail_cvar(candidate_scores, self.profile.tail_fraction)
        reasons: list[str] = []
        if parent.artifact_hash == candidate.artifact_hash:
            reasons.append("candidate_equals_parent")
        if len(parent_scores) < self.profile.minimum_samples:
            reasons.append("insufficient_parent_samples")
        if len(candidate_scores) < self.profile.minimum_samples:
            reasons.append("insufficient_candidate_samples")
        if candidate.safety_violation_count > 0:
            reasons.append("candidate_safety_violation")
        if candidate_worst < parent_worst - self.profile.maximum_worst_regression - 1e-12:
            reasons.append("candidate_worst_regression")
        if candidate_cvar < parent_cvar - self.profile.maximum_cvar_regression - 1e-12:
            reasons.append("candidate_cvar_regression")
        return RobustnessDecision(
            status=RobustnessStatus.REJECTED if reasons else RobustnessStatus.PASSED,
            parent_worst_score=parent_worst,
            candidate_worst_score=candidate_worst,
            parent_cvar_score=parent_cvar,
            candidate_cvar_score=candidate_cvar,
            reasons=tuple(reasons),
            parent_evidence_hash=parent.evidence_hash,
            candidate_evidence_hash=candidate.evidence_hash,
        )


def _oriented_scores(values: tuple[float, ...], direction: MetricDirection) -> tuple[float, ...]:
    return values if direction is MetricDirection.MAXIMIZE else tuple(-value for value in values)


def _lower_tail_cvar(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    tail_count = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:tail_count]) / tail_count


@dataclass(frozen=True)
class ApplicabilityEvidence:
    candidate_artifact_hash: str
    parent_artifact_hash: str
    context_hash: str
    evidence_hash: str
    in_distribution: bool
    confidence: float
    support_distance: float
    schema_version: str = "rosclaw.growth.applicability_evidence.v1"

    def __post_init__(self) -> None:
        for label in (
            "candidate_artifact_hash",
            "parent_artifact_hash",
            "context_hash",
            "evidence_hash",
        ):
            require_hash(label, getattr(self, label))
        if self.candidate_artifact_hash == self.parent_artifact_hash:
            raise ValueError("candidate and parent artifact hashes must differ")
        if not isinstance(self.in_distribution, bool):
            raise ValueError("in_distribution must be boolean")
        confidence = _strict_number(self.confidence, label="confidence")
        support_distance = _strict_number(self.support_distance, label="support_distance")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if support_distance < 0.0:
            raise ValueError("support_distance must be finite and non-negative")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "support_distance", support_distance)


@dataclass(frozen=True)
class ApplicabilityDecision:
    selected_artifact_hash: str
    used_candidate: bool
    evidence_hash: str
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.growth.applicability_decision.v1"

    def __post_init__(self) -> None:
        require_hash("selected_artifact_hash", self.selected_artifact_hash)
        require_hash("evidence_hash", self.evidence_hash)
        if not isinstance(self.used_candidate, bool):
            raise ValueError("used_candidate must be boolean")
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(
                self.reasons,
                label="reasons",
                allow_empty=self.used_candidate,
            ),
        )
        if self.used_candidate and self.reasons:
            raise ValueError("an applicable candidate cannot carry fallback reasons")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def activation_allowed(self) -> bool:
        return False

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_artifact_hash": self.selected_artifact_hash,
            "used_candidate": self.used_candidate,
            "evidence_hash": self.evidence_hash,
            "reasons": list(self.reasons),
            "activation_allowed": self.activation_allowed,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class ApplicabilityGate:
    minimum_confidence: float
    maximum_support_distance: float
    schema_version: str = "rosclaw.growth.applicability_gate.v1"

    def __post_init__(self) -> None:
        minimum_confidence = _strict_number(
            self.minimum_confidence,
            label="minimum_confidence",
        )
        maximum_support_distance = _strict_number(
            self.maximum_support_distance,
            label="maximum_support_distance",
        )
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be finite and in [0, 1]")
        if maximum_support_distance < 0:
            raise ValueError("maximum_support_distance must be finite and non-negative")
        object.__setattr__(self, "minimum_confidence", minimum_confidence)
        object.__setattr__(self, "maximum_support_distance", maximum_support_distance)

    def evaluate(self, evidence: ApplicabilityEvidence) -> ApplicabilityDecision:
        reasons: list[str] = []
        if not evidence.in_distribution:
            reasons.append("out_of_distribution")
        if evidence.confidence < self.minimum_confidence:
            reasons.append("low_applicability_confidence")
        if evidence.support_distance > self.maximum_support_distance:
            reasons.append("support_distance_exceeded")
        use_candidate = not reasons
        return ApplicabilityDecision(
            selected_artifact_hash=(
                evidence.candidate_artifact_hash if use_candidate else evidence.parent_artifact_hash
            ),
            used_candidate=use_candidate,
            evidence_hash=evidence.evidence_hash,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class ModeSelectionContext:
    context_hash: str
    feature_schema_hash: str
    available_mode_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.mode_selection_context.v1"

    def __post_init__(self) -> None:
        require_hash("context_hash", self.context_hash)
        require_hash("feature_schema_hash", self.feature_schema_hash)
        object.__setattr__(
            self,
            "available_mode_ids",
            unique_identifiers(self.available_mode_ids, label="available_mode_ids"),
        )


@dataclass(frozen=True)
class ModeDecision:
    context_hash: str
    selected_mode_id: str | None
    evidence_hash: str
    reasons: tuple[str, ...] = ()
    schema_version: str = "rosclaw.growth.mode_decision.v1"

    def __post_init__(self) -> None:
        require_hash("context_hash", self.context_hash)
        require_hash("evidence_hash", self.evidence_hash)
        if self.selected_mode_id is not None:
            require_identifier("selected_mode_id", self.selected_mode_id)
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(
                self.reasons,
                label="reasons",
                allow_empty=self.selected_mode_id is not None,
            ),
        )
        if self.selected_mode_id is not None and self.reasons:
            raise ValueError("a selected mode cannot carry rejection reasons")

    @property
    def hardware_authorized(self) -> bool:
        return False


@runtime_checkable
class ModeGate(Protocol):
    """Adapter boundary for domain-defined regime or mode selection."""

    def evaluate(self, context: ModeSelectionContext) -> ModeDecision: ...


__all__ = [
    "ApplicabilityDecision",
    "ApplicabilityEvidence",
    "ApplicabilityGate",
    "CandidateEvidenceDecision",
    "CandidateEvidenceGate",
    "CandidateEvidenceStatus",
    "CandidateExecutionEvidence",
    "ModeDecision",
    "ModeGate",
    "ModeSelectionContext",
    "RobustnessDecision",
    "RobustnessEvidence",
    "RobustnessGate",
    "RobustnessProfile",
    "RobustnessStatus",
]
