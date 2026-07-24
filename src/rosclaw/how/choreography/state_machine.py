"""Patch safety & execution state machine (数据库优化v4 §9).

    PROPOSED
    → APPLICABILITY_VALIDATED
    → CHOREOGRAPHY_VALIDATED
    → SANDBOX_VALIDATED
    → APPROVED
    → APPLIED
    → OBSERVED
    → CRITIC_VALIDATED
    → LEARNED

Any failed gate → BLOCKED.  The machine only allows the declared
transitions; a patch can never reach APPLIED without having passed every
gate upstream, and Rule Efficacy updates only from LEARNED (patch_applied
+ critic decision + regime match — the v4 §9 rule).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PatchState(StrEnum):
    PROPOSED = "PROPOSED"
    APPLICABILITY_VALIDATED = "APPLICABILITY_VALIDATED"
    CHOREOGRAPHY_VALIDATED = "CHOREOGRAPHY_VALIDATED"
    SANDBOX_VALIDATED = "SANDBOX_VALIDATED"
    APPROVED = "APPROVED"
    APPLIED = "APPLIED"
    OBSERVED = "OBSERVED"
    CRITIC_VALIDATED = "CRITIC_VALIDATED"
    LEARNED = "LEARNED"
    BLOCKED = "BLOCKED"


_FORWARD: dict[PatchState, PatchState] = {
    PatchState.PROPOSED: PatchState.APPLICABILITY_VALIDATED,
    PatchState.APPLICABILITY_VALIDATED: PatchState.CHOREOGRAPHY_VALIDATED,
    PatchState.CHOREOGRAPHY_VALIDATED: PatchState.SANDBOX_VALIDATED,
    PatchState.SANDBOX_VALIDATED: PatchState.APPROVED,
    PatchState.APPROVED: PatchState.APPLIED,
    PatchState.APPLIED: PatchState.OBSERVED,
    PatchState.OBSERVED: PatchState.CRITIC_VALIDATED,
    PatchState.CRITIC_VALIDATED: PatchState.LEARNED,
}


@dataclass
class PatchStateRecord:
    """One patch's journey through the gates (auditable)."""

    patch_id: str
    state: PatchState = PatchState.PROPOSED
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "state": self.state.value,
            "history": self.history,
        }


class PatchStateMachine:
    """Declared-transitions-only patch lifecycle (v4 §9)."""

    def __init__(self, patch_id: str) -> None:
        self._record = PatchStateRecord(patch_id=patch_id)

    @property
    def record(self) -> PatchStateRecord:
        return self._record

    @property
    def state(self) -> PatchState:
        return self._record.state

    def advance(self, *, evidence: dict[str, Any] | None = None) -> PatchState:
        """Move to the next state.  Raises on any non-declared transition."""
        current = self._record.state
        if current in (PatchState.LEARNED, PatchState.BLOCKED):
            raise RuntimeError(f"terminal state {current.value} cannot advance")
        nxt = _FORWARD[current]
        self._record.history.append(
            {"from": current.value, "to": nxt.value, "at": time.time(), "evidence": evidence or {}}
        )
        self._record.state = nxt
        return nxt

    def block(self, *, reason: str, evidence: dict[str, Any] | None = None) -> PatchState:
        """Any gate failure → BLOCKED (terminal)."""
        current = self._record.state
        self._record.history.append(
            {
                "from": current.value,
                "to": PatchState.BLOCKED.value,
                "at": time.time(),
                "reason": reason,
                "evidence": evidence or {},
            }
        )
        self._record.state = PatchState.BLOCKED
        return PatchState.BLOCKED


def efficacy_learnable(state: PatchState) -> bool:
    """v4 §9: Rule Efficacy updates only from LEARNED — patch applied,
    critic decision explicit, regime matched (all proven by the history)."""
    return state is PatchState.LEARNED
