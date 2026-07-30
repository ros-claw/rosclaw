"""Candidate lifecycle v2 (Physical Evolution Lab §9.4, PR-PE-7).

The v1 states were too strong: ``VALIDATED`` meant "passed the seven
gates" and ``PROMOTED`` meant "won ONE canary generation" — neither is
evidence of effectiveness (v3 §9.4: 当前 VALIDATED 和 PROMOTED 语义过
强).  v2 names the semantics:

* SCHEMA_VALIDATED   — well-formed, in the declared space;
* SAFETY_VALIDATED   — cannot violate constraints (choreography + invariants);
* SHADOW_VALIDATED   — execution chain correct, NO real-effect evidence;
* CANARY_APPROVED    — operator may run real low-traffic canary;
* PROVISIONALLY_PROMOTED — exploratory small-sample signal EXISTS;
* CONFIRMATION_PENDING   — provisional, never a resting state;
* VALIDATED_EFFECTIVE    — frozen-protocol independent confirmation passed;
* ACTIVE                 — usable inside declared regime scopes;
* REJECTED / ROLLED_BACK / RETIRED — terminal states with reasons.

Exploratory evidence can never jump to VALIDATED_EFFECTIVE: the only
path is through the confirmation campaign (v3 §9.5/§9.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

STATE_VERSION = "rosclaw.candidate_state.v2"


class CandidateStateV2(StrEnum):
    PROPOSED = "proposed"
    SCHEMA_VALIDATED = "schema_validated"
    SAFETY_VALIDATED = "safety_validated"
    SHADOW_VALIDATED = "shadow_validated"
    CANARY_APPROVED = "canary_approved"
    PROVISIONALLY_PROMOTED = "provisionally_promoted"
    CONFIRMATION_PENDING = "confirmation_pending"
    VALIDATED_EFFECTIVE = "validated_effective"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"


# Allowed transitions (from → frozenset of to).  Everything else is an
# IllegalTransition error with the attempted edge named.
_TRANSITIONS: dict[CandidateStateV2, frozenset[CandidateStateV2]] = {
    CandidateStateV2.PROPOSED: frozenset(
        {CandidateStateV2.SCHEMA_VALIDATED, CandidateStateV2.REJECTED}
    ),
    CandidateStateV2.SCHEMA_VALIDATED: frozenset(
        {CandidateStateV2.SAFETY_VALIDATED, CandidateStateV2.REJECTED}
    ),
    CandidateStateV2.SAFETY_VALIDATED: frozenset(
        {CandidateStateV2.SHADOW_VALIDATED, CandidateStateV2.REJECTED}
    ),
    CandidateStateV2.SHADOW_VALIDATED: frozenset(
        {CandidateStateV2.CANARY_APPROVED, CandidateStateV2.REJECTED}
    ),
    CandidateStateV2.CANARY_APPROVED: frozenset(
        {
            CandidateStateV2.PROVISIONALLY_PROMOTED,
            CandidateStateV2.REJECTED,
            CandidateStateV2.ROLLED_BACK,
        }
    ),
    CandidateStateV2.PROVISIONALLY_PROMOTED: frozenset(
        {CandidateStateV2.CONFIRMATION_PENDING, CandidateStateV2.ROLLED_BACK}
    ),
    CandidateStateV2.CONFIRMATION_PENDING: frozenset(
        {
            CandidateStateV2.VALIDATED_EFFECTIVE,
            CandidateStateV2.REJECTED,
            CandidateStateV2.ROLLED_BACK,
            CandidateStateV2.RETIRED,
        }
    ),
    CandidateStateV2.VALIDATED_EFFECTIVE: frozenset(
        {CandidateStateV2.ACTIVE, CandidateStateV2.ROLLED_BACK, CandidateStateV2.RETIRED}
    ),
    CandidateStateV2.ACTIVE: frozenset({CandidateStateV2.ROLLED_BACK, CandidateStateV2.RETIRED}),
    CandidateStateV2.REJECTED: frozenset({CandidateStateV2.RETIRED}),
    CandidateStateV2.ROLLED_BACK: frozenset({CandidateStateV2.RETIRED}),
    CandidateStateV2.RETIRED: frozenset(),
}

TERMINAL = {CandidateStateV2.REJECTED, CandidateStateV2.ROLLED_BACK, CandidateStateV2.RETIRED}


class IllegalTransitionError(RuntimeError):
    pass


@dataclass
class CandidateLifecycle:
    """One candidate's state + the evidence trail that moved it."""

    candidate_id: str
    state: CandidateStateV2 = CandidateStateV2.PROPOSED
    history: list[dict] = field(default_factory=list)

    def advance(self, to: CandidateStateV2, *, evidence: str, actor: str = "system") -> None:
        allowed = _TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            raise IllegalTransitionError(
                f"{self.candidate_id}: {self.state} → {to} is not an allowed "
                f"transition (allowed: {sorted(allowed)})"
            )
        self.history.append(
            {"from": self.state.value, "to": to.value, "evidence": evidence, "actor": actor}
        )
        self.state = to


def migrate_v1_state(v1_state: str) -> CandidateStateV2:
    """v1 → v2 mapping (v3 §9.4: cand_003 应调整为 PROVISIONALLY_PROMOTED
    / CONFIRMATION_PENDING).

    * v1 VALIDATED (gates passed, no canary) → SHADOW_VALIDATED + the
      CANARY_APPROVED rung reached by the real canary run — the honest
      v2 landing for gate-passed, canary-tested, not-promoted candidates
      is CANARY_APPROVED.
    * v1 PROMOTED (one canary generation won) → CONFIRMATION_PENDING
      (via PROVISIONALLY_PROMOTED semantics: exploratory signal, never
      effective).
    * v1 ROLLED_BACK → ROLLED_BACK.  v1 PROPOSED → PROPOSED.
    """
    mapping = {
        "proposed": CandidateStateV2.PROPOSED,
        "validated": CandidateStateV2.CANARY_APPROVED,
        "promoted": CandidateStateV2.CONFIRMATION_PENDING,
        "rolled_back": CandidateStateV2.ROLLED_BACK,
        "rejected": CandidateStateV2.REJECTED,
    }
    try:
        return mapping[v1_state.lower()]
    except KeyError:
        raise ValueError(f"unknown v1 candidate state: {v1_state!r}") from None
