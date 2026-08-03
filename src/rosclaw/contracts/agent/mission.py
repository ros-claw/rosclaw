"""MissionSessionV1 — cognitive/dialogue session (ADR-0002, 总纲 §4.2).

Distinct from the daemon's physical ``AgentSession``. Owned and persisted by
rosclaw-agentd; recovered from the event journal, not from chat history.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import ContractModel


class ExecutionMode(StrEnum):
    SIMULATION = "SIMULATION"
    SHADOW = "SHADOW"
    REAL = "REAL"


class MissionState(StrEnum):
    """Native Agent main state machine (总纲 §3.3)."""

    IDLE = "IDLE"
    UNDERSTAND = "UNDERSTAND"
    GROUND = "GROUND"
    WAIT_INPUT = "WAIT_INPUT"
    PLAN = "PLAN"
    STAFF = "STAFF"
    VALIDATE = "VALIDATE"
    WAIT_APPROVAL = "WAIT_APPROVAL"
    DISPATCH = "DISPATCH"
    MONITOR = "MONITOR"
    VERIFY = "VERIFY"
    LEARN = "LEARN"
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"


# Legal transitions of the mission state machine (总纲 §3.3). The MissionStore
# rejects any edge not listed here — an LLM can never invent states.
# Refinements over 总纲 §3.3 (documented, ADR-0002):
# - UNDERSTAND/GROUND/PLAN → FAILED: unrecoverable failure or operator cancel
#   must be possible in early states too.
# - VALIDATE → LEARN: pure-answer missions (no dispatch) complete honestly
#   instead of being forced through DISPATCH.
# - PLAN/VALIDATE → WAIT_INPUT: budget/token/money exhaustion or missing
#   information parks the mission for operator input (§4.2).
MISSION_TRANSITIONS: dict[MissionState, frozenset[MissionState]] = {
    MissionState.IDLE: frozenset({MissionState.UNDERSTAND}),
    MissionState.UNDERSTAND: frozenset({MissionState.GROUND, MissionState.FAILED}),
    MissionState.GROUND: frozenset(
        {MissionState.PLAN, MissionState.WAIT_INPUT, MissionState.FAILED}
    ),
    MissionState.WAIT_INPUT: frozenset(
        {MissionState.GROUND, MissionState.SUSPENDED, MissionState.FAILED}
    ),
    MissionState.PLAN: frozenset(
        {
            MissionState.STAFF,
            MissionState.VALIDATE,
            MissionState.WAIT_INPUT,
            MissionState.FAILED,
        }
    ),
    MissionState.STAFF: frozenset(
        {MissionState.VALIDATE, MissionState.WAIT_INPUT, MissionState.FAILED}
    ),
    MissionState.VALIDATE: frozenset(
        {
            MissionState.WAIT_APPROVAL,
            MissionState.DISPATCH,
            MissionState.WAIT_INPUT,
            MissionState.LEARN,
            MissionState.FAILED,
        }
    ),
    MissionState.WAIT_APPROVAL: frozenset(
        {MissionState.DISPATCH, MissionState.FAILED, MissionState.SUSPENDED}
    ),
    MissionState.DISPATCH: frozenset({MissionState.MONITOR, MissionState.FAILED}),
    MissionState.MONITOR: frozenset(
        {
            MissionState.VERIFY,
            MissionState.PLAN,
            MissionState.SUSPENDED,
            MissionState.FAILED,
        }
    ),
    MissionState.VERIFY: frozenset(
        {
            MissionState.LEARN,
            MissionState.FAILED,
            MissionState.PLAN,
            # 验证阶段可能需要新的授权动作（如 LIMO 验收的第二个动作卡）。
            MissionState.WAIT_APPROVAL,
        }
    ),
    MissionState.LEARN: frozenset({MissionState.IDLE}),
    MissionState.SUSPENDED: frozenset({MissionState.PLAN, MissionState.FAILED}),
    MissionState.FAILED: frozenset({MissionState.IDLE}),
}

TERMINAL_STATES: frozenset[MissionState] = frozenset({MissionState.IDLE, MissionState.FAILED})


class SuccessCriterion(ContractModel):
    SCHEMA = "rosclaw.success_criterion.v1"

    schema_version: Literal["rosclaw.success_criterion.v1"] = "rosclaw.success_criterion.v1"
    type: str = Field(..., description="e.g. object_pose_in_region")
    parameters: dict[str, Any] = Field(default_factory=dict)


class Goal(ContractModel):
    SCHEMA = "rosclaw.mission_goal.v1"

    schema_version: Literal["rosclaw.mission_goal.v1"] = "rosclaw.mission_goal.v1"
    text: str
    language: str = "zh-CN"
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)


class BodyBinding(ContractModel):
    SCHEMA = "rosclaw.body_binding.v1"

    schema_version: Literal["rosclaw.body_binding.v1"] = "rosclaw.body_binding.v1"
    body_id: str
    effective_body_hash: str


class Budgets(ContractModel):
    SCHEMA = "rosclaw.mission_budgets.v1"

    schema_version: Literal["rosclaw.mission_budgets.v1"] = "rosclaw.mission_budgets.v1"
    wall_time_sec: int = 900
    model_tokens: int = 200_000
    monetary_microunits: int = 5_000_000
    worker_concurrency: int = 3
    physical_action_count: int = 0
    max_tool_rounds: int = 12


class AuthorizationBinding(ContractModel):
    """Grant reference only — private permits never enter this object."""

    SCHEMA = "rosclaw.mission_authorization.v1"

    schema_version: Literal["rosclaw.mission_authorization.v1"] = "rosclaw.mission_authorization.v1"
    mission_grant_id: str | None = None
    allowed_risk_tiers: list[str] = Field(default_factory=lambda: ["LOW"])


class MissionSessionV1(ContractModel):
    SCHEMA = "rosclaw.mission_session.v1"
    HASH_PREFIX = "mis"
    HASH_EXCLUDE_FIELD = "updated_at"

    schema_version: Literal["rosclaw.mission_session.v1"] = "rosclaw.mission_session.v1"
    mission_id: str
    owner_principal: str = Field(..., description="e.g. user:local:1000")
    goal: Goal
    body_binding: BodyBinding
    mode: ExecutionMode = ExecutionMode.SIMULATION
    state: MissionState = MissionState.IDLE
    budgets: Budgets = Field(default_factory=Budgets)
    authorization: AuthorizationBinding = Field(default_factory=AuthorizationBinding)
    context_revision: int = 0
    task_graph_revision: int = 0
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def _check_real_mode_constraints(self) -> MissionSessionV1:
        # REAL without any grant reference is not necessarily invalid at
        # contract level (EXACT_ACTION approvals may be per-action), but a
        # REAL mission may never declare physical_action_count budget 0 —
        # that combination means "real actions with no budget", fail closed.
        if (
            self.mode is ExecutionMode.REAL
            and not self.authorization.mission_grant_id
            and self.budgets.physical_action_count <= 0
        ):
            raise ValueError(
                "REAL mode requires physical_action_count budget > 0 "
                "(fail closed: no implicit real-action allowance)"
            )
        return self

    def can_transition(self, to_state: MissionState) -> bool:
        return to_state in MISSION_TRANSITIONS[self.state]
