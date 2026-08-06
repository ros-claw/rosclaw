"""DecisionV1 — the only structured output of the Native Agent (总纲 §6.2).

The system records ``public_rationale`` and evidence, never private
chain-of-thought. A decision is bound to a context revision; stale decisions
are rejected by the DecisionValidator.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class NextIntent(StrEnum):
    ANSWER = "ANSWER"
    OBSERVE = "OBSERVE"
    PLAN_PATCH = "PLAN_PATCH"
    HIRE_WORKER = "HIRE_WORKER"
    TEAM_COORDINATE = "TEAM_COORDINATE"
    REQUEST_APPROVAL = "REQUEST_APPROVAL"
    REQUEST_ACTION = "REQUEST_ACTION"
    VERIFY = "VERIFY"
    WAIT = "WAIT"
    PAUSE = "PAUSE"
    FAIL_SAFE = "FAIL_SAFE"


class Assumption(ContractModel):
    SCHEMA = "rosclaw.decision_assumption.v1"

    schema_version: Literal["rosclaw.decision_assumption.v1"] = "rosclaw.decision_assumption.v1"
    claim: str
    evidence_ref: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Uncertainty(ContractModel):
    SCHEMA = "rosclaw.decision_uncertainty.v1"

    schema_version: Literal["rosclaw.decision_uncertainty.v1"] = "rosclaw.decision_uncertainty.v1"
    level: Literal["LOW", "MODERATE", "HIGH"] = "MODERATE"
    reasons: list[str] = Field(default_factory=list)


class ProposedOperation(ContractModel):
    SCHEMA = "rosclaw.proposed_operation.v1"

    schema_version: Literal["rosclaw.proposed_operation.v1"] = "rosclaw.proposed_operation.v1"
    type: str = Field(..., description="e.g. create_work_order, task_graph_patch")
    payload_ref: str | None = None
    payload: dict[str, Any] | None = None


class Verification(ContractModel):
    SCHEMA = "rosclaw.decision_verification.v1"

    schema_version: Literal["rosclaw.decision_verification.v1"] = "rosclaw.decision_verification.v1"
    schema_ref: str | None = None
    verifiers: list[str] = Field(default_factory=list)


class OnFailure(ContractModel):
    SCHEMA = "rosclaw.decision_on_failure.v1"

    schema_version: Literal["rosclaw.decision_on_failure.v1"] = "rosclaw.decision_on_failure.v1"
    intent: NextIntent = NextIntent.PAUSE
    reason: str = ""


class DecisionV1(ContractModel):
    SCHEMA = "rosclaw.decision.v1"
    HASH_PREFIX = "dec"

    schema_version: Literal["rosclaw.decision.v1"] = "rosclaw.decision.v1"
    decision_id: str
    mission_id: str
    context_id: str
    context_revision: int
    next_intent: NextIntent
    summary: str = ""
    assumptions: list[Assumption] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    proposed_operation: ProposedOperation | None = None
    verification: Verification | None = None
    on_failure: OnFailure | None = None
    public_rationale: str = ""
