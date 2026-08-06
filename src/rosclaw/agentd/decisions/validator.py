"""DecisionValidator — gate between model output and system action (PR-NA-022).

Checks (fail closed on any violation):

- schema validity (pydantic contract, closed enums);
- the decision is bound to the *current* context id + revision — stale
  decisions are rejected and must be re-planned (总纲 §5.4 step 13);
- mission id matches;
- ``proposed_operation.type`` is in the allowlist for the declared intent;
- REQUEST_ACTION requires verification + evidence;
- decisions never carry mode escalation (SIMULATION→REAL) or permit material.
"""

from __future__ import annotations

from rosclaw.contracts.agent.decision import DecisionV1, NextIntent
from rosclaw.contracts.common import ValidationError


class DecisionRejectedError(ValidationError):
    """The model's decision failed validation; the loop must re-plan."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


#: Which operation types each intent may propose.
INTENT_OPERATION_ALLOWLIST: dict[NextIntent, frozenset[str]] = {
    NextIntent.ANSWER: frozenset(),
    NextIntent.OBSERVE: frozenset({"observe", "refresh_state"}),
    NextIntent.PLAN_PATCH: frozenset({"task_graph_patch"}),
    NextIntent.HIRE_WORKER: frozenset({"create_work_order"}),
    NextIntent.TEAM_COORDINATE: frozenset({"team_message", "team_bid", "team_task_claim"}),
    NextIntent.REQUEST_APPROVAL: frozenset({"approval_request"}),
    NextIntent.REQUEST_ACTION: frozenset({"request_action"}),
    NextIntent.VERIFY: frozenset({"verify_receipt", "verify_observation"}),
    NextIntent.WAIT: frozenset(),
    NextIntent.PAUSE: frozenset(),
    NextIntent.FAIL_SAFE: frozenset({"estop_request"}),
}

#: Intents that demand evidence references before dispatch.
_EVIDENCE_REQUIRED = frozenset({NextIntent.REQUEST_ACTION, NextIntent.HIRE_WORKER})

#: Intents meaningless without a proposed operation (force repair/PAUSE).
_OPERATION_REQUIRED = frozenset({NextIntent.TEAM_COORDINATE})

_FORBIDDEN_OP_FIELDS = ("mode", "permit", "signature", "credential")


class DecisionValidator:
    def __init__(self, *, current_context_id: str, current_context_revision: int) -> None:
        self._context_id = current_context_id
        self._context_revision = current_context_revision

    def validate(self, decision: DecisionV1, *, mission_id: str) -> DecisionV1:
        if decision.mission_id != mission_id:
            raise DecisionRejectedError(
                "mission_mismatch",
                f"decision mission {decision.mission_id!r} != {mission_id!r}",
            )
        if decision.context_id != self._context_id:
            raise DecisionRejectedError(
                "context_mismatch",
                f"decision context {decision.context_id!r} != current {self._context_id!r}",
            )
        if decision.context_revision != self._context_revision:
            raise DecisionRejectedError(
                "stale_context",
                f"decision bound to revision {decision.context_revision}, "
                f"current is {self._context_revision} — replan required",
            )
        op = decision.proposed_operation
        if op is None and decision.next_intent in _OPERATION_REQUIRED:
            raise DecisionRejectedError(
                "missing_operation",
                f"intent {decision.next_intent.value} requires a proposed_operation "
                "(see the allowed type list in the system instructions)",
            )
        if op is not None:
            allowed = INTENT_OPERATION_ALLOWLIST[decision.next_intent]
            if op.type not in allowed:
                raise DecisionRejectedError(
                    "operation_not_allowed",
                    f"intent {decision.next_intent.value} may not propose "
                    f"operation {op.type!r} (allowed: {sorted(allowed)})",
                )
            payload = op.payload or {}
            for field in _FORBIDDEN_OP_FIELDS:
                if field in payload:
                    raise DecisionRejectedError(
                        "forbidden_operation_field",
                        f"operation payload may not carry {field!r}",
                    )
        if decision.next_intent in _EVIDENCE_REQUIRED:
            if not decision.evidence_refs and op is None:
                raise DecisionRejectedError(
                    "missing_evidence",
                    f"intent {decision.next_intent.value} requires evidence_refs "
                    "or a proposed operation payload reference",
                )
            if decision.verification is None or not decision.verification.verifiers:
                raise DecisionRejectedError(
                    "missing_verification",
                    f"intent {decision.next_intent.value} requires verification verifiers",
                )
        return decision
