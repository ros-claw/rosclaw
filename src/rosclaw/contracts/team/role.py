"""RoleLeaseV1 (总纲 §10.4).

Roles are leases scoped to a team epoch, never permanent identities. On
expiry, epoch change, or member loss they lapse automatically. Conflicting
leases trigger the conservative conflict policy: stop contesting, retreat
to a safe region, request re-coordination.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class RoleScope(ContractModel):
    SCHEMA = "rosclaw.role_scope.v1"

    schema_version: Literal["rosclaw.role_scope.v1"] = "rosclaw.role_scope.v1"
    region: str | None = None
    task_types: list[str] = Field(default_factory=list)


class RoleLeaseV1(ContractModel):
    SCHEMA = "rosclaw.role_lease.v1"
    HASH_PREFIX = "rlease"

    schema_version: Literal["rosclaw.role_lease.v1"] = "rosclaw.role_lease.v1"
    team_id: str
    team_epoch: int
    role: str = Field(..., description="e.g. defender:left")
    holder: str
    scope: RoleScope = Field(default_factory=RoleScope)
    issued_at: str
    expires_at: str
    renew_after_ms: int = 500
    priority: int = 50
    conflict_key: str
    policy_hash: str
    state: Literal["ACTIVE", "EXPIRED", "REVOKED", "CONTESTED"] = "ACTIVE"
