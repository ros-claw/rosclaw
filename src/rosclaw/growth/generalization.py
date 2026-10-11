"""Fail-closed admission gate for promoting domain discoveries into Core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth._validation import (
    require_hash,
    require_identifier,
    unique_identifiers,
)


@dataclass(frozen=True)
class DomainValidationProof:
    domain_id: str
    task_id: str
    adapter_id: str
    evidence_hash: str
    schema_version: str = "rosclaw.growth.domain_validation_proof.v1"

    def __post_init__(self) -> None:
        for label in ("domain_id", "task_id", "adapter_id"):
            require_identifier(label, getattr(self, label))
        require_hash("evidence_hash", self.evidence_hash)

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "domain_id": self.domain_id,
            "task_id": self.task_id,
            "adapter_id": self.adapter_id,
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True)
class GeneralizationEvidence:
    component_id: str
    source_paths: tuple[str, ...]
    imported_modules: tuple[str, ...]
    adapter_entry_point_group: str
    synthetic_test_report_hash: str
    plugin_removal_test_report_hash: str
    domain_proofs: tuple[DomainValidationProof, ...]
    schema_version: str = "rosclaw.growth.generalization_evidence.v1"

    def __post_init__(self) -> None:
        require_identifier("component_id", self.component_id)
        object.__setattr__(
            self,
            "source_paths",
            _unique_strings(self.source_paths, label="source_paths"),
        )
        object.__setattr__(
            self,
            "imported_modules",
            _unique_strings(self.imported_modules, label="imported_modules", allow_empty=True),
        )
        require_identifier("adapter_entry_point_group", self.adapter_entry_point_group)
        require_hash("synthetic_test_report_hash", self.synthetic_test_report_hash)
        require_hash("plugin_removal_test_report_hash", self.plugin_removal_test_report_hash)
        proofs = tuple(self.domain_proofs)
        if not proofs or any(not isinstance(proof, DomainValidationProof) for proof in proofs):
            raise ValueError("domain_proofs must contain DomainValidationProof records")
        if len({(proof.domain_id, proof.task_id) for proof in proofs}) != len(proofs):
            raise ValueError("domain proofs must be unique by domain and task")
        object.__setattr__(self, "domain_proofs", proofs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "component_id": self.component_id,
            "source_paths": list(self.source_paths),
            "imported_modules": list(self.imported_modules),
            "adapter_entry_point_group": self.adapter_entry_point_group,
            "synthetic_test_report_hash": self.synthetic_test_report_hash,
            "plugin_removal_test_report_hash": self.plugin_removal_test_report_hash,
            "domain_proofs": [proof.to_dict() for proof in self.domain_proofs],
        }


class GeneralizationStatus(StrEnum):
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GeneralizationDecision:
    status: GeneralizationStatus
    evidence_hash: str
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.growth.generalization_decision.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, GeneralizationStatus):
            raise ValueError("status must be a GeneralizationStatus")
        require_hash("evidence_hash", self.evidence_hash)
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(self.reasons, label="reasons", allow_empty=True),
        )
        if (self.status is GeneralizationStatus.PASSED) != (not self.reasons):
            raise ValueError("a passed decision has no rejection reasons")

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "evidence_hash": self.evidence_hash,
            "reasons": list(self.reasons),
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class CoreGeneralizationGate:
    """Evaluate CI-collected facts; never execute a domain plugin or robot."""

    downstream_import_prefixes: tuple[str, ...]
    domain_tokens: tuple[str, ...]
    minimum_distinct_domains: int = 2
    schema_version: str = "rosclaw.growth.core_generalization_gate.v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "downstream_import_prefixes",
            _unique_strings(
                self.downstream_import_prefixes,
                label="downstream_import_prefixes",
                allow_empty=True,
            ),
        )
        normalized_tokens = tuple(token.strip().lower() for token in self.domain_tokens)
        if any(not token for token in normalized_tokens) or len(normalized_tokens) != len(
            set(normalized_tokens)
        ):
            raise ValueError("domain_tokens must be unique non-empty strings")
        object.__setattr__(self, "domain_tokens", normalized_tokens)
        if not 2 <= self.minimum_distinct_domains <= 32:
            raise ValueError("minimum_distinct_domains must be in [2, 32]")

    def evaluate(self, evidence: GeneralizationEvidence) -> GeneralizationDecision:
        if not isinstance(evidence, GeneralizationEvidence):
            raise ValueError("evidence must be GeneralizationEvidence")
        reasons: list[str] = []
        searchable_names = (evidence.component_id, *evidence.source_paths)
        if any(
            token in name.lower().replace("-", "_")
            for token in self.domain_tokens
            for name in searchable_names
        ):
            reasons.append("domain_specific_name")
        if any(
            module == prefix or module.startswith(prefix + ".")
            for module in evidence.imported_modules
            for prefix in self.downstream_import_prefixes
        ):
            reasons.append("downstream_import")
        if not evidence.adapter_entry_point_group:
            reasons.append("missing_adapter_injection")
        domain_count = len({proof.domain_id for proof in evidence.domain_proofs})
        task_count = len({proof.task_id for proof in evidence.domain_proofs})
        adapter_count = len({proof.adapter_id for proof in evidence.domain_proofs})
        if domain_count < self.minimum_distinct_domains:
            reasons.append("insufficient_domains")
        if task_count < self.minimum_distinct_domains:
            reasons.append("insufficient_tasks")
        if adapter_count < self.minimum_distinct_domains:
            reasons.append("insufficient_adapters")
        status = GeneralizationStatus.REJECTED if reasons else GeneralizationStatus.PASSED
        return GeneralizationDecision(
            status=status,
            evidence_hash=canonical_hash(evidence.to_dict()),
            reasons=tuple(reasons),
        )


def _unique_strings(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must contain unique non-empty strings")
    return normalized


__all__ = [
    "CoreGeneralizationGate",
    "DomainValidationProof",
    "GeneralizationDecision",
    "GeneralizationEvidence",
    "GeneralizationStatus",
]
