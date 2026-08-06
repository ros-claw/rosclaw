"""Team Fabric (ADR-0004): multi-robot control plane.

Maturity: **experimental**. Every robot stays an autonomous safety unit;
team assignments are contracts the local agent + rosclawd may reject.
No consensus is invented here: a single logical Coordinator with
term/epoch, durable and idempotently replayable awards (总纲 §10.7).
"""

from rosclaw.team.allocator import AllocationResult, ContractNetAllocator
from rosclaw.team.coordinator import (
    DegradedPolicy,
    TeamCoordinator,
    TeamError,
)
from rosclaw.team.membership import MemberState, TeamMembership
from rosclaw.team.roles import RoleConflictError, RoleLeaseStore
from rosclaw.team.transport import LocalSimTransport
from rosclaw.team.world import ClockSkewError, StaleWorldError, WorldModel

__all__ = [
    "AllocationResult",
    "ClockSkewError",
    "ContractNetAllocator",
    "DegradedPolicy",
    "LocalSimTransport",
    "MemberState",
    "RoleConflictError",
    "RoleLeaseStore",
    "StaleWorldError",
    "TeamCoordinator",
    "TeamError",
    "TeamMembership",
    "WorldModel",
]

MATURITY = "experimental"
