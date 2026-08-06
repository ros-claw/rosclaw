"""MissionGrantV1 — public authorization scope (ADR-0006).

The grant *public* object is what the agent may see and reference. The
signature/private permit material stays inside the Operator Broker / daemon
and never enters this contract (no ``signature``/``permit`` fields exist).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class GrantScope(ContractModel):
    SCHEMA = "rosclaw.grant_scope.v1"

    schema_version: Literal["rosclaw.grant_scope.v1"] = "rosclaw.grant_scope.v1"
    action_families: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    tier: Literal["EXACT_ACTION", "PLAN", "MISSION", "SITE_POLICY"] = "EXACT_ACTION"
    exact_action_intent: str | None = Field(
        None, description="EXACT_ACTION grants bind one exact action intent hash"
    )


class GrantBudgets(ContractModel):
    SCHEMA = "rosclaw.grant_budgets.v1"

    schema_version: Literal["rosclaw.grant_budgets.v1"] = "rosclaw.grant_budgets.v1"
    max_actions: int = 1
    max_velocity_m_s: float | None = None
    max_energy_j: float | None = None


class MissionGrantV1(ContractModel):
    SCHEMA = "rosclaw.mission_grant.v1"
    HASH_PREFIX = "grantpub"
    HASH_EXCLUDE_FIELD = "public_hash"

    schema_version: Literal["rosclaw.mission_grant.v1"] = "rosclaw.mission_grant.v1"
    grant_id: str
    principal: str
    body_id: str
    effective_body_hash: str
    mode: Literal["SIMULATION", "SHADOW", "REAL"] = "SIMULATION"
    scope: GrantScope
    risk_ceiling: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    budgets: GrantBudgets = Field(default_factory=GrantBudgets)
    policy_hash: str
    issued_at: str
    expires_at: str
    public_hash: str = ""

    def finalize_hash(self) -> str:
        self.public_hash = self.canonical_hash()
        return self.public_hash
