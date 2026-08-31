"""Single-focal plasticity leases for multi-agent continual learning."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class AgentUpdateMode(StrEnum):
    PLASTIC = "PLASTIC"
    FROZEN = "FROZEN"


@dataclass(frozen=True)
class AgentPolicyBinding:
    agent_id: str
    policy_hash: str
    mode: AgentUpdateMode

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.agent_id) or not _SHA256.fullmatch(self.policy_hash):
            raise ValueError("agent policy binding is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "policy_hash": self.policy_hash,
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class PlasticityLease:
    """Grant candidate updates to exactly one focal agent for a bounded run."""

    lease_id: str
    bindings: tuple[AgentPolicyBinding, ...]
    dataset_manifest_hash: str
    scenario_contract_hash: str
    maximum_optimizer_steps: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.plasticity_lease.v1"

    def __post_init__(self) -> None:
        ids = tuple(binding.agent_id for binding in self.bindings)
        plastic = tuple(
            binding.agent_id for binding in self.bindings if binding.mode is AgentUpdateMode.PLASTIC
        )
        if (
            not _IDENTIFIER.fullmatch(self.lease_id)
            or len(self.bindings) < 2
            or len(ids) != len(set(ids))
            or len(plastic) != 1
            or any(
                not _SHA256.fullmatch(value)
                for value in (self.dataset_manifest_hash, self.scenario_contract_hash)
            )
            or not 1 <= self.maximum_optimizer_steps <= 100_000_000
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("plasticity lease is invalid")

    @property
    def focal_agent_id(self) -> str:
        return next(
            binding.agent_id for binding in self.bindings if binding.mode is AgentUpdateMode.PLASTIC
        )

    @property
    def lease_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lease_id": self.lease_id,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "scenario_contract_hash": self.scenario_contract_hash,
            "maximum_optimizer_steps": self.maximum_optimizer_steps,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class PlasticityLeaseAudit:
    lease_hash: str
    optimizer_steps: int
    changed_agent_ids: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]
    before_policy_hashes: Mapping[str, str]
    after_policy_hashes: Mapping[str, str]
    schema_version: str = "rosclaw.continual.plasticity_lease_audit.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.lease_hash) or self.optimizer_steps < 0:
            raise ValueError("plasticity lease audit identity is invalid")
        before = _policy_hash_mapping(self.before_policy_hashes)
        after = _policy_hash_mapping(self.after_policy_hashes)
        if set(before) != set(after):
            raise ValueError("plasticity audit before/after rosters differ")
        object.__setattr__(self, "before_policy_hashes", MappingProxyType(before))
        object.__setattr__(self, "after_policy_hashes", MappingProxyType(after))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lease_hash": self.lease_hash,
            "optimizer_steps": self.optimizer_steps,
            "changed_agent_ids": list(self.changed_agent_ids),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "before_policy_hashes": dict(self.before_policy_hashes),
            "after_policy_hashes": dict(self.after_policy_hashes),
        }


def audit_plasticity_lease(
    *,
    lease: PlasticityLease,
    optimizer_steps: int,
    before_policy_hashes: Mapping[str, str],
    after_policy_hashes: Mapping[str, str],
) -> PlasticityLeaseAudit:
    before = _policy_hash_mapping(before_policy_hashes)
    after = _policy_hash_mapping(after_policy_hashes)
    expected = {binding.agent_id for binding in lease.bindings}
    if set(before) != expected or set(after) != expected:
        raise ValueError("plasticity audit roster does not match its lease")
    changed = tuple(
        sorted(agent_id for agent_id in expected if before[agent_id] != after[agent_id])
    )
    reasons = []
    if optimizer_steps < 0 or optimizer_steps > lease.maximum_optimizer_steps:
        reasons.append("OPTIMIZER_STEP_BUDGET_EXCEEDED")
    unexpected = tuple(agent_id for agent_id in changed if agent_id != lease.focal_agent_id)
    if unexpected:
        reasons.append("FROZEN_AGENT_CHANGED:" + ",".join(unexpected))
    return PlasticityLeaseAudit(
        lease_hash=lease.lease_hash,
        optimizer_steps=optimizer_steps,
        changed_agent_ids=changed,
        passed=not reasons,
        reasons=tuple(reasons),
        before_policy_hashes=before,
        after_policy_hashes=after,
    )


def _policy_hash_mapping(values: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(agent_id): str(policy_hash) for agent_id, policy_hash in values.items()}
    if any(
        not _IDENTIFIER.fullmatch(agent_id) or not _SHA256.fullmatch(policy_hash)
        for agent_id, policy_hash in normalized.items()
    ):
        raise ValueError("policy hash mapping is invalid")
    return dict(sorted(normalized.items()))


__all__ = [
    "AgentPolicyBinding",
    "AgentUpdateMode",
    "PlasticityLease",
    "PlasticityLeaseAudit",
    "audit_plasticity_lease",
]
