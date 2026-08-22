"""Task-neutral PARC experience, provenance, and failure contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth._validation import (
    finite_mapping,
    require_hash,
    require_identifier,
    unique_hashes,
    unique_identifiers,
)
from rosclaw.growth.contracts import EvidenceLevel


@dataclass(frozen=True)
class DerivedExperienceLineage:
    """Bind derived observations to their source events and time alignment.

    A transform that combines multiple events must carry a synchronization
    receipt and prove that observed skew stayed within its declared bound.
    This prevents separately sampled state channels from being presented as
    one physical observation without auditable alignment evidence.
    """

    source_artifact_hash: str
    source_event_hashes: tuple[str, ...]
    transform_hash: str
    clock_id: str
    maximum_skew_sec: float
    observed_skew_sec: float
    synchronization_receipt_hash: str | None = None
    schema_version: str = "rosclaw.growth.derived_experience_lineage.v1"

    def __post_init__(self) -> None:
        require_hash("source_artifact_hash", self.source_artifact_hash)
        require_hash("transform_hash", self.transform_hash)
        object.__setattr__(
            self,
            "source_event_hashes",
            unique_hashes(self.source_event_hashes, label="source_event_hashes"),
        )
        require_identifier("clock_id", self.clock_id)
        if not math.isfinite(self.maximum_skew_sec) or self.maximum_skew_sec < 0.0:
            raise ValueError("maximum_skew_sec must be finite and non-negative")
        if not math.isfinite(self.observed_skew_sec) or self.observed_skew_sec < 0.0:
            raise ValueError("observed_skew_sec must be finite and non-negative")
        if self.observed_skew_sec > self.maximum_skew_sec:
            raise ValueError("observed skew exceeds the declared synchronization bound")
        if len(self.source_event_hashes) > 1:
            if self.synchronization_receipt_hash is None:
                raise ValueError("multi-event derivation requires a synchronization receipt")
            require_hash("synchronization_receipt_hash", self.synchronization_receipt_hash)
        elif self.synchronization_receipt_hash is not None:
            require_hash("synchronization_receipt_hash", self.synchronization_receipt_hash)

    @property
    def lineage_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "source_event_hashes": list(self.source_event_hashes),
            "transform_hash": self.transform_hash,
            "clock_id": self.clock_id,
            "maximum_skew_sec": self.maximum_skew_sec,
            "observed_skew_sec": self.observed_skew_sec,
            "synchronization_receipt_hash": self.synchronization_receipt_hash,
        }


class PhysicalAdvantageLabel(StrEnum):
    UNLABELED = "unlabeled"
    ADVANTAGE_POSITIVE = "advantage_positive"
    ADVANTAGE_NEGATIVE = "advantage_negative"
    ADVANTAGE_NEUTRAL = "advantage_neutral"
    UNSAFE_NEGATIVE = "unsafe_negative"


@dataclass(frozen=True)
class FailureSignature:
    primary_type: str
    contributors: tuple[str, ...]
    confidence: float
    affected_capability_ids: tuple[str, ...]
    reusable_evidence_ids: tuple[str, ...]
    recommended_learner_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.failure_signature.v1"

    def __post_init__(self) -> None:
        require_identifier("primary_type", self.primary_type)
        for label in ("contributors", "affected_capability_ids", "recommended_learner_ids"):
            object.__setattr__(
                self,
                label,
                unique_identifiers(tuple(getattr(self, label)), label=label),
            )
        object.__setattr__(
            self,
            "reusable_evidence_ids",
            unique_identifiers(
                self.reusable_evidence_ids,
                label="reusable_evidence_ids",
                allow_empty=True,
            ),
        )
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

    @property
    def signature_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary_type": self.primary_type,
            "contributors": list(self.contributors),
            "confidence": self.confidence,
            "affected_capability_ids": list(self.affected_capability_ids),
            "reusable_evidence_ids": list(self.reusable_evidence_ids),
            "recommended_learner_ids": list(self.recommended_learner_ids),
        }


@dataclass(frozen=True)
class ActionTraceCommitment:
    commanded_action_hash: str
    executed_action_hash: str
    safety_projected_action_hash: str
    policy_version: str
    controller_hash: str
    projection_applied: bool
    schema_version: str = "rosclaw.growth.action_trace_commitment.v1"

    def __post_init__(self) -> None:
        for label in (
            "commanded_action_hash",
            "executed_action_hash",
            "safety_projected_action_hash",
            "controller_hash",
        ):
            require_hash(label, getattr(self, label))
        require_identifier("policy_version", self.policy_version)
        if not self.projection_applied and (
            self.executed_action_hash != self.safety_projected_action_hash
        ):
            raise ValueError("an unprojected action must match the executed action")

    @property
    def commitment_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "commanded_action_hash": self.commanded_action_hash,
            "executed_action_hash": self.executed_action_hash,
            "safety_projected_action_hash": self.safety_projected_action_hash,
            "policy_version": self.policy_version,
            "controller_hash": self.controller_hash,
            "projection_applied": self.projection_applied,
        }


@dataclass(frozen=True)
class ExperienceSegment:
    """One physically meaningful PARC segment, independent of task semantics."""

    segment_id: str
    episode_id: str
    skill_id: str
    phase: str
    start_time_sec: float
    end_time_sec: float
    body_hash: str
    regime_hash: str
    source_evidence_level: EvidenceLevel
    lineage: DerivedExperienceLineage
    base_policy_version: str
    residual_policy_version: str | None
    state_start_hash: str
    observation_sequence_hash: str
    self_state_hash: str
    world_state_hash: str
    action: ActionTraceCommitment
    reward_vector: Mapping[str, float]
    cost_vector: Mapping[str, float]
    terminal_state_hash: str
    advantage_label: PhysicalAdvantageLabel = PhysicalAdvantageLabel.UNLABELED
    label_confidence: float = 0.0
    failure_signature: FailureSignature | None = None
    schema_version: str = "rosclaw.growth.experience_segment.v1"

    def __post_init__(self) -> None:
        for label in ("segment_id", "episode_id", "skill_id", "phase", "base_policy_version"):
            require_identifier(label, getattr(self, label))
        if self.residual_policy_version is not None:
            require_identifier("residual_policy_version", self.residual_policy_version)
        if not math.isfinite(self.start_time_sec) or not math.isfinite(self.end_time_sec):
            raise ValueError("segment times must be finite")
        if self.start_time_sec < 0.0 or self.end_time_sec <= self.start_time_sec:
            raise ValueError("segment times must satisfy 0 <= start < end")
        for label in (
            "body_hash",
            "regime_hash",
            "state_start_hash",
            "observation_sequence_hash",
            "self_state_hash",
            "world_state_hash",
            "terminal_state_hash",
        ):
            require_hash(label, getattr(self, label))
        if not isinstance(self.source_evidence_level, EvidenceLevel):
            raise ValueError("source_evidence_level must be an EvidenceLevel")
        if not isinstance(self.lineage, DerivedExperienceLineage):
            raise ValueError("lineage must be a DerivedExperienceLineage")
        if not isinstance(self.action, ActionTraceCommitment):
            raise ValueError("action must be an ActionTraceCommitment")
        if not isinstance(self.advantage_label, PhysicalAdvantageLabel):
            raise ValueError("advantage_label must be a PhysicalAdvantageLabel")
        object.__setattr__(
            self,
            "reward_vector",
            finite_mapping(self.reward_vector, label="reward_vector", non_negative=False),
        )
        object.__setattr__(
            self,
            "cost_vector",
            finite_mapping(self.cost_vector, label="cost_vector", non_negative=True),
        )
        if not math.isfinite(self.label_confidence) or not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be finite and in [0, 1]")
        negative = self.advantage_label in {
            PhysicalAdvantageLabel.ADVANTAGE_NEGATIVE,
            PhysicalAdvantageLabel.UNSAFE_NEGATIVE,
        }
        if negative and self.failure_signature is None:
            raise ValueError("negative advantage segments require a failure signature")
        if self.failure_signature is not None and not isinstance(
            self.failure_signature, FailureSignature
        ):
            raise ValueError("failure_signature must be a FailureSignature")
        if self.advantage_label is PhysicalAdvantageLabel.UNSAFE_NEGATIVE and not any(
            value > 0.0 for value in self.cost_vector.values()
        ):
            raise ValueError("unsafe-negative segments require a non-zero safety cost")
        if self.advantage_label is PhysicalAdvantageLabel.ADVANTAGE_POSITIVE and any(
            value > 0.0 for value in self.cost_vector.values()
        ):
            raise ValueError("positive advantage cannot contain a safety cost")

    @property
    def segment_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "segment_id": self.segment_id,
            "episode_id": self.episode_id,
            "skill_id": self.skill_id,
            "phase": self.phase,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "identity": {
                "body_hash": self.body_hash,
                "regime_hash": self.regime_hash,
                "base_policy_version": self.base_policy_version,
                "residual_policy_version": self.residual_policy_version,
            },
            "source": {
                "evidence_level": self.source_evidence_level.value,
                "lineage": self.lineage.to_dict(),
            },
            "state": {
                "start_hash": self.state_start_hash,
                "observation_sequence_hash": self.observation_sequence_hash,
                "self_state_hash": self.self_state_hash,
                "world_state_hash": self.world_state_hash,
            },
            "action": self.action.to_dict(),
            "outcome": {
                "reward_vector": dict(sorted(self.reward_vector.items())),
                "cost_vector": dict(sorted(self.cost_vector.items())),
                "terminal_state_hash": self.terminal_state_hash,
            },
            "labels": {
                "advantage": self.advantage_label.value,
                "confidence": self.label_confidence,
                "failure_signature": (
                    self.failure_signature.to_dict() if self.failure_signature else None
                ),
            },
            "promotion_truth_allowed": self.source_evidence_level
            in {EvidenceLevel.EXECUTION_RECEIPT, EvidenceLevel.PHYSICS_REPLAY},
            "hardware_authorized": False,
        }


__all__ = [
    "ActionTraceCommitment",
    "DerivedExperienceLineage",
    "ExperienceSegment",
    "FailureSignature",
    "PhysicalAdvantageLabel",
]
