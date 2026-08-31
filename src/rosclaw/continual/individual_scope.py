"""Agent-isolated Parent/Candidate/Champion lifecycle contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from rosclaw.continual.contracts import PolicyVersion
from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a valid stable identifier")


@dataclass(frozen=True)
class FrozenPartner:
    """One immutable collaborator/opponent used for counterfactual evaluation."""

    agent_id: str
    champion_policy_hash: str
    body_hash: str
    foundation_policy_hash: str
    capability_profile_hash: str

    def __post_init__(self) -> None:
        _require_identifier("agent_id", self.agent_id)
        for label, value in (
            ("champion_policy_hash", self.champion_policy_hash),
            ("body_hash", self.body_hash),
            ("foundation_policy_hash", self.foundation_policy_hash),
            ("capability_profile_hash", self.capability_profile_hash),
        ):
            _require_hash(label, value)

    def to_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "champion_policy_hash": self.champion_policy_hash,
            "body_hash": self.body_hash,
            "foundation_policy_hash": self.foundation_policy_hash,
            "capability_profile_hash": self.capability_profile_hash,
        }


@dataclass(frozen=True)
class FrozenPartnerSet:
    """Content-addressed roster around one focal learner."""

    focal_agent_id: str
    partners: tuple[FrozenPartner, ...]
    numerical_contract_hash: str
    scenario_contract_hash: str
    schema_version: str = "rosclaw.continual.frozen_partner_set.v1"

    def __post_init__(self) -> None:
        _require_identifier("focal_agent_id", self.focal_agent_id)
        _require_hash("numerical_contract_hash", self.numerical_contract_hash)
        _require_hash("scenario_contract_hash", self.scenario_contract_hash)
        partner_ids = tuple(partner.agent_id for partner in self.partners)
        if self.focal_agent_id in partner_ids:
            raise ValueError("focal agent cannot appear in its frozen partner set")
        if len(partner_ids) != len(set(partner_ids)):
            raise ValueError("frozen partner identifiers must be unique")

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "focal_agent_id": self.focal_agent_id,
            "partners": [partner.to_dict() for partner in self.partners],
            "numerical_contract_hash": self.numerical_contract_hash,
            "scenario_contract_hash": self.scenario_contract_hash,
        }


@dataclass(frozen=True)
class IndividualPromotionEvidence:
    """Identity-bound evidence required to promote one agent only."""

    agent_id: str
    parent_policy_hash: str
    candidate_policy_hash: str
    frozen_partner_snapshot_hash: str
    matched_seed_commitment_hash: str
    gate_report_hash: str
    personal_memory_namespace: str
    retention_passed: bool
    safety_passed: bool
    team_compatibility_passed: bool
    schema_version: str = "rosclaw.continual.individual_promotion_evidence.v1"

    def __post_init__(self) -> None:
        _require_identifier("agent_id", self.agent_id)
        _require_identifier("personal_memory_namespace", self.personal_memory_namespace)
        for label, value in (
            ("parent_policy_hash", self.parent_policy_hash),
            ("candidate_policy_hash", self.candidate_policy_hash),
            ("frozen_partner_snapshot_hash", self.frozen_partner_snapshot_hash),
            ("matched_seed_commitment_hash", self.matched_seed_commitment_hash),
            ("gate_report_hash", self.gate_report_hash),
        ):
            _require_hash(label, value)

    @property
    def passed(self) -> bool:
        return self.retention_passed and self.safety_passed and self.team_compatibility_passed

    @property
    def evidence_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "parent_policy_hash": self.parent_policy_hash,
            "candidate_policy_hash": self.candidate_policy_hash,
            "frozen_partner_snapshot_hash": self.frozen_partner_snapshot_hash,
            "matched_seed_commitment_hash": self.matched_seed_commitment_hash,
            "gate_report_hash": self.gate_report_hash,
            "personal_memory_namespace": self.personal_memory_namespace,
            "retention_passed": self.retention_passed,
            "safety_passed": self.safety_passed,
            "team_compatibility_passed": self.team_compatibility_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class IndividualGrowthScope:
    """Immutable growth state whose candidate can affect exactly one agent."""

    agent_id: str
    body_hash: str
    body_state_hash: str
    foundation_policy_hash: str
    personal_adapter_hash: str
    role_policy_hash: str
    residual_policy_hash: str
    capability_profile_hash: str
    career_lineage_hash: str
    personal_memory_namespace: str
    failure_memory_namespace: str
    parent_policy: PolicyVersion
    champion_policy: PolicyVersion
    candidate_policy: PolicyVersion | None = None
    candidate_partner_snapshot_hash: str | None = None
    generation: int = 0
    last_promotion_evidence_hash: str | None = None
    schema_version: str = "rosclaw.continual.individual_growth_scope.v1"

    def __post_init__(self) -> None:
        _require_identifier("agent_id", self.agent_id)
        _require_identifier("personal_memory_namespace", self.personal_memory_namespace)
        _require_identifier("failure_memory_namespace", self.failure_memory_namespace)
        for label, value in (
            ("body_hash", self.body_hash),
            ("body_state_hash", self.body_state_hash),
            ("foundation_policy_hash", self.foundation_policy_hash),
            ("personal_adapter_hash", self.personal_adapter_hash),
            ("role_policy_hash", self.role_policy_hash),
            ("residual_policy_hash", self.residual_policy_hash),
            ("capability_profile_hash", self.capability_profile_hash),
            ("career_lineage_hash", self.career_lineage_hash),
        ):
            _require_hash(label, value)
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.parent_policy.version_hash != self.champion_policy.version_hash:
            raise ValueError("parent and champion must identify the current deployed policy")
        for policy in (self.parent_policy, self.champion_policy):
            if policy.body_hash != self.body_hash:
                raise ValueError("scope body does not match parent/champion policy")
        if (self.candidate_policy is None) != (self.candidate_partner_snapshot_hash is None):
            raise ValueError("candidate and frozen partner snapshot must be staged together")
        if self.candidate_policy is not None:
            self._validate_candidate(self.candidate_policy)
            _require_hash(
                "candidate_partner_snapshot_hash",
                self.candidate_partner_snapshot_hash or "",
            )
        if self.last_promotion_evidence_hash is not None:
            _require_hash("last_promotion_evidence_hash", self.last_promotion_evidence_hash)

    @property
    def scope_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def stage_candidate(
        self,
        candidate: PolicyVersion,
        *,
        partners: FrozenPartnerSet,
    ) -> IndividualGrowthScope:
        if self.candidate_policy is not None:
            raise ValueError("a candidate is already staged")
        if partners.focal_agent_id != self.agent_id:
            raise ValueError("partner snapshot belongs to another focal agent")
        self._validate_candidate(candidate)
        return replace(
            self,
            candidate_policy=candidate,
            candidate_partner_snapshot_hash=partners.snapshot_hash,
        )

    def reject_candidate(self) -> IndividualGrowthScope:
        """Discard only the focal agent's staged candidate."""

        return replace(
            self,
            candidate_policy=None,
            candidate_partner_snapshot_hash=None,
        )

    def promote_candidate(
        self,
        evidence: IndividualPromotionEvidence,
    ) -> IndividualGrowthScope:
        candidate = self.candidate_policy
        if candidate is None or self.candidate_partner_snapshot_hash is None:
            raise ValueError("no candidate is staged")
        if evidence.agent_id != self.agent_id:
            raise ValueError("promotion evidence belongs to another agent")
        if evidence.personal_memory_namespace != self.personal_memory_namespace:
            raise ValueError("promotion evidence is bound to another memory namespace")
        if evidence.parent_policy_hash != self.champion_policy.version_hash:
            raise ValueError("promotion evidence parent does not match the champion")
        if evidence.candidate_policy_hash != candidate.version_hash:
            raise ValueError("promotion evidence candidate does not match the staged policy")
        if evidence.frozen_partner_snapshot_hash != self.candidate_partner_snapshot_hash:
            raise ValueError("promotion evidence used a different partner snapshot")
        if not evidence.passed:
            raise ValueError("candidate failed an individual promotion gate")
        return replace(
            self,
            parent_policy=candidate,
            champion_policy=candidate,
            candidate_policy=None,
            candidate_partner_snapshot_hash=None,
            generation=self.generation + 1,
            last_promotion_evidence_hash=evidence.evidence_hash,
        )

    def _validate_candidate(self, candidate: PolicyVersion) -> None:
        if candidate.parent_version_hash != self.champion_policy.version_hash:
            raise ValueError("candidate parent does not match the focal champion")
        if candidate.body_hash != self.body_hash:
            raise ValueError("candidate body does not match the individual scope")
        if candidate.controller_snapshot_hash != self.champion_policy.controller_snapshot_hash:
            raise ValueError("candidate controller snapshot changed inside an individual scope")
        if candidate.safety_kernel_hash != self.champion_policy.safety_kernel_hash:
            raise ValueError("candidate safety kernel changed inside an individual scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "body_hash": self.body_hash,
            "body_state_hash": self.body_state_hash,
            "foundation_policy_hash": self.foundation_policy_hash,
            "personal_adapter_hash": self.personal_adapter_hash,
            "role_policy_hash": self.role_policy_hash,
            "residual_policy_hash": self.residual_policy_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "career_lineage_hash": self.career_lineage_hash,
            "personal_memory_namespace": self.personal_memory_namespace,
            "failure_memory_namespace": self.failure_memory_namespace,
            "parent_policy": self.parent_policy.to_dict(),
            "champion_policy": self.champion_policy.to_dict(),
            "candidate_policy": (
                None if self.candidate_policy is None else self.candidate_policy.to_dict()
            ),
            "candidate_partner_snapshot_hash": self.candidate_partner_snapshot_hash,
            "generation": self.generation,
            "last_promotion_evidence_hash": self.last_promotion_evidence_hash,
        }


__all__ = [
    "FrozenPartner",
    "FrozenPartnerSet",
    "IndividualGrowthScope",
    "IndividualPromotionEvidence",
]
