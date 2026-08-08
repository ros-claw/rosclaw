"""Immutable, fail-closed contracts for evidence-backed skill growth.

The Growth Plane may propose and evaluate artifacts, but these contracts never
authorize a physical actuator.  They make evidence provenance, retention and
rollback lineage explicit so low-trust synthetic data cannot silently become
promotion truth.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


def _unique_nonempty(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be non-empty and unique")
    if any(not item.strip() for item in normalized):
        raise ValueError(f"{label} must not contain empty values")
    return normalized


def _hash_tuple(
    values: tuple[str, ...], *, label: str, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for item in normalized:
        _require_hash(label, item)
    return normalized


class EvidenceLevel(StrEnum):
    """Evidence strength, from executed reality to unverified external data."""

    EXECUTION_RECEIPT = "e0_execution_receipt"
    PHYSICS_REPLAY = "e1_physics_replay"
    ACCELERATED_SIM = "e2_accelerated_sim"
    WORLD_MODEL = "e3_world_model"
    EXTERNAL_APPLICABLE = "e4_external_applicable"
    EXTERNAL_UNVERIFIED = "e5_external_unverified"


class TrainingEligibility(StrEnum):
    DENIED = "denied"
    CONDITIONAL = "conditional"
    ALLOWED = "allowed"


@dataclass(frozen=True)
class EvidenceUsePolicy:
    """Canonical capabilities of one evidence level.

    The permissions are computed rather than caller supplied.  In particular,
    E2--E5 evidence can discover or train candidates but cannot independently
    approve one.
    """

    level: EvidenceLevel
    schema_version: str = "rosclaw.growth.evidence_use_policy.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.level, EvidenceLevel):
            raise ValueError("level must be a recognized EvidenceLevel")

    @property
    def training(self) -> TrainingEligibility:
        if self.level is EvidenceLevel.EXTERNAL_UNVERIFIED:
            return TrainingEligibility.DENIED
        if self.level is EvidenceLevel.EXTERNAL_APPLICABLE:
            return TrainingEligibility.CONDITIONAL
        return TrainingEligibility.ALLOWED

    @property
    def candidate_discovery_allowed(self) -> bool:
        return True

    @property
    def promotion_truth_allowed(self) -> bool:
        return self.level in {EvidenceLevel.EXECUTION_RECEIPT, EvidenceLevel.PHYSICS_REPLAY}

    @property
    def required_replay_level(self) -> EvidenceLevel | None:
        if self.promotion_truth_allowed:
            return None
        return EvidenceLevel.PHYSICS_REPLAY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "level": self.level.value,
            "training": self.training.value,
            "candidate_discovery_allowed": self.candidate_discovery_allowed,
            "promotion_truth_allowed": self.promotion_truth_allowed,
            "required_replay_level": (
                self.required_replay_level.value if self.required_replay_level else None
            ),
        }


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class GrowthMetricSpec:
    metric_id: str
    direction: MetricDirection
    primary: bool = False
    minimum_relative_improvement: float = 0.05
    confidence_level: float = 0.95
    require_ci_lower_bound_positive: bool = True
    schema_version: str = "rosclaw.growth.metric_spec.v1"

    def __post_init__(self) -> None:
        _require_identifier("metric_id", self.metric_id)
        if not isinstance(self.direction, MetricDirection):
            raise ValueError("direction must be a recognized MetricDirection")
        if not math.isfinite(self.minimum_relative_improvement):
            raise ValueError("minimum_relative_improvement must be finite")
        if self.minimum_relative_improvement < 0.0:
            raise ValueError("minimum_relative_improvement must be non-negative")
        if not math.isfinite(self.confidence_level) or not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be finite and in (0.5, 1.0)")

    @property
    def metric_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "direction": self.direction.value,
            "primary": self.primary,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "confidence_level": self.confidence_level,
            "require_ci_lower_bound_positive": self.require_ci_lower_bound_positive,
        }


@dataclass(frozen=True)
class SkillGrowthSpec:
    """Task-neutral growth target pinned to semantic and safety contracts."""

    skill_id: str
    adapter_id: str
    body_hashes: tuple[str, ...]
    capability_ids: tuple[str, ...]
    observation_contract_hash: str
    action_contract_hash: str
    reward_contract_hash: str
    cost_contract_hash: str
    practice_source_ids: tuple[str, ...]
    collective_source_ids: tuple[str, ...]
    allowed_dream_types: tuple[str, ...]
    allowed_learner_ids: tuple[str, ...]
    historical_anchor_hashes: tuple[str, ...]
    boundary_suite_hash: str
    metrics: tuple[GrowthMetricSpec, ...]
    promotion_profile_hash: str
    rollback_policy_hash: str
    schema_version: str = "rosclaw.growth.skill_spec.v1"

    def __post_init__(self) -> None:
        _require_identifier("skill_id", self.skill_id)
        _require_identifier("adapter_id", self.adapter_id)
        for label, value in (
            ("observation_contract_hash", self.observation_contract_hash),
            ("action_contract_hash", self.action_contract_hash),
            ("reward_contract_hash", self.reward_contract_hash),
            ("cost_contract_hash", self.cost_contract_hash),
            ("boundary_suite_hash", self.boundary_suite_hash),
            ("promotion_profile_hash", self.promotion_profile_hash),
            ("rollback_policy_hash", self.rollback_policy_hash),
        ):
            _require_hash(label, value)
        object.__setattr__(self, "body_hashes", _hash_tuple(self.body_hashes, label="body_hashes"))
        object.__setattr__(
            self,
            "historical_anchor_hashes",
            _hash_tuple(self.historical_anchor_hashes, label="historical_anchor_hashes"),
        )
        for label in (
            "capability_ids",
            "practice_source_ids",
            "allowed_dream_types",
            "allowed_learner_ids",
        ):
            object.__setattr__(self, label, _unique_nonempty(getattr(self, label), label=label))
        object.__setattr__(
            self,
            "collective_source_ids",
            tuple(self.collective_source_ids),
        )
        if len(self.collective_source_ids) != len(set(self.collective_source_ids)):
            raise ValueError("collective_source_ids must be unique")
        metrics = tuple(self.metrics)
        if any(not isinstance(metric, GrowthMetricSpec) for metric in metrics):
            raise ValueError("metrics must contain GrowthMetricSpec records")
        if not metrics or sum(metric.primary for metric in metrics) != 1:
            raise ValueError("metrics must contain exactly one primary metric")
        if next(metric for metric in metrics if metric.primary).minimum_relative_improvement <= 0.0:
            raise ValueError("the primary metric must require a positive relative improvement")
        if len({metric.metric_id for metric in metrics}) != len(metrics):
            raise ValueError("metric ids must be unique")
        object.__setattr__(self, "metrics", metrics)

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "adapter_id": self.adapter_id,
            "body_hashes": list(self.body_hashes),
            "capability_ids": list(self.capability_ids),
            "observation_contract_hash": self.observation_contract_hash,
            "action_contract_hash": self.action_contract_hash,
            "reward_contract_hash": self.reward_contract_hash,
            "cost_contract_hash": self.cost_contract_hash,
            "practice_source_ids": list(self.practice_source_ids),
            "collective_source_ids": list(self.collective_source_ids),
            "allowed_dream_types": list(self.allowed_dream_types),
            "allowed_learner_ids": list(self.allowed_learner_ids),
            "historical_anchor_hashes": list(self.historical_anchor_hashes),
            "boundary_suite_hash": self.boundary_suite_hash,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "promotion_profile_hash": self.promotion_profile_hash,
            "rollback_policy_hash": self.rollback_policy_hash,
        }


class GateName(StrEnum):
    LEARNING = "learning"
    RETENTION = "retention"
    SAFETY = "safety"
    APPLICABILITY = "applicability"
    DARWIN = "darwin"


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"


@dataclass(frozen=True)
class GateResult:
    name: GateName
    status: GateStatus
    report_hash: str | None = None
    detail: str = ""
    schema_version: str = "rosclaw.growth.gate_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, GateName):
            raise ValueError("name must be a recognized GateName")
        if not isinstance(self.status, GateStatus):
            raise ValueError("status must be a recognized GateStatus")
        if self.status is GateStatus.MISSING:
            if self.report_hash is not None:
                raise ValueError("a missing gate cannot carry a report hash")
        elif self.report_hash is None:
            raise ValueError("a completed gate requires a report hash")
        else:
            _require_hash("report_hash", self.report_hash)

    @property
    def result_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name.value,
            "status": self.status.value,
            "report_hash": self.report_hash,
            "detail": self.detail,
        }


class ConsolidationDecision(StrEnum):
    CONSOLIDATE_SIM = "consolidate_sim"
    REJECT = "reject"
    NEED_MORE_EVIDENCE = "need_more_evidence"


@dataclass(frozen=True)
class ConsolidationManifest:
    """Auditable result of all five growth gates.

    V1 intentionally supports simulation consolidation only.  A separate,
    operator-mediated mechanism is required for any physical deployment.
    """

    skill_growth_spec_hash: str
    candidate_artifact_hash: str
    parent_artifact_hash: str
    rollback_artifact_hash: str
    learned_changes: Mapping[str, str]
    new_capability_ids: tuple[str, ...]
    retained_capability_ids: tuple[str, ...]
    forgotten_capability_ids: tuple[str, ...]
    gate_results: tuple[GateResult, ...]
    darwin_report_hash: str
    decision: ConsolidationDecision
    schema_version: str = "rosclaw.growth.consolidation_manifest.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ConsolidationDecision):
            raise ValueError("decision must be a recognized ConsolidationDecision")
        for label, value in (
            ("skill_growth_spec_hash", self.skill_growth_spec_hash),
            ("candidate_artifact_hash", self.candidate_artifact_hash),
            ("parent_artifact_hash", self.parent_artifact_hash),
            ("rollback_artifact_hash", self.rollback_artifact_hash),
            ("darwin_report_hash", self.darwin_report_hash),
        ):
            _require_hash(label, value)
        if self.candidate_artifact_hash == self.parent_artifact_hash:
            raise ValueError("candidate artifact must differ from its parent")
        if self.rollback_artifact_hash != self.parent_artifact_hash:
            raise ValueError("v1 rollback artifact must be the pinned parent artifact")
        learned_changes = {str(key): str(value) for key, value in self.learned_changes.items()}
        if not learned_changes:
            raise ValueError("learned_changes must not be empty")
        for value in learned_changes.values():
            _require_hash("learned_changes value", value)
        object.__setattr__(self, "learned_changes", MappingProxyType(learned_changes))
        for label in (
            "new_capability_ids",
            "retained_capability_ids",
            "forgotten_capability_ids",
        ):
            values = tuple(getattr(self, label))
            if len(values) != len(set(values)) or any(not value.strip() for value in values):
                raise ValueError(f"{label} must contain unique non-empty values")
            object.__setattr__(self, label, values)
        gates = tuple(self.gate_results)
        if any(not isinstance(gate, GateResult) for gate in gates):
            raise ValueError("gate_results must contain GateResult records")
        if {gate.name for gate in gates} != set(GateName) or len(gates) != len(GateName):
            raise ValueError("gate_results must contain each required gate exactly once")
        object.__setattr__(self, "gate_results", gates)
        darwin_gate = next(gate for gate in gates if gate.name is GateName.DARWIN)
        if (
            darwin_gate.report_hash is not None
            and darwin_gate.report_hash != self.darwin_report_hash
        ):
            raise ValueError("darwin_report_hash must match the Darwin gate report")
        expected = self._expected_decision(gates)
        if self.decision is not expected:
            raise ValueError(f"decision must be {expected.value} for the supplied gate results")
        if self.decision is ConsolidationDecision.CONSOLIDATE_SIM:
            if self.forgotten_capability_ids:
                raise ValueError("a consolidatable candidate cannot forget declared capabilities")
            if not self.retained_capability_ids:
                raise ValueError("a consolidatable candidate must prove retained capabilities")

    @staticmethod
    def _expected_decision(gates: tuple[GateResult, ...]) -> ConsolidationDecision:
        if any(gate.status is GateStatus.FAIL for gate in gates):
            return ConsolidationDecision.REJECT
        if any(gate.status is GateStatus.MISSING for gate in gates):
            return ConsolidationDecision.NEED_MORE_EVIDENCE
        return ConsolidationDecision.CONSOLIDATE_SIM

    @property
    def evidence_domain(self) -> str:
        return "SIM_ONLY"

    @property
    def hardware_authorized(self) -> bool:
        return False

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_growth_spec_hash": self.skill_growth_spec_hash,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "parent_artifact_hash": self.parent_artifact_hash,
            "rollback_artifact_hash": self.rollback_artifact_hash,
            "learned_changes": dict(sorted(self.learned_changes.items())),
            "new_capability_ids": list(self.new_capability_ids),
            "retained_capability_ids": list(self.retained_capability_ids),
            "forgotten_capability_ids": list(self.forgotten_capability_ids),
            "gate_results": [gate.to_dict() for gate in self.gate_results],
            "darwin_report_hash": self.darwin_report_hash,
            "decision": self.decision.value,
            "evidence_domain": self.evidence_domain,
            "hardware_authorized": self.hardware_authorized,
        }
