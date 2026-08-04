"""ApprovalRequestV2 (总纲 §11.1) — the card a human approves or denies."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"


class ActionDisplayV1(ContractModel):
    """Human-readable risk display (aligned with interaction ActionDisplay)."""

    SCHEMA = "rosclaw.action_display.v1"

    schema_version: Literal["rosclaw.action_display.v1"] = "rosclaw.action_display.v1"
    title: str
    summary: str
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    expected_effect: str = ""
    failure_handling: str = ""
    parameters: dict = Field(default_factory=dict)


class ApprovalRequestV2(ContractModel):
    SCHEMA = "rosclaw.approval_request.v2"

    schema_version: Literal["rosclaw.approval_request.v2"] = "rosclaw.approval_request.v2"
    request_id: str
    mission_id: str
    task_id: str | None = None
    principal: str
    body_id: str
    effective_body_hash: str
    mode: Literal["SIMULATION", "SHADOW", "REAL"] = "SIMULATION"
    action_display: ActionDisplayV1
    context_id: str
    context_revision: int
    requested_tier: Literal["EXACT_ACTION", "PLAN", "MISSION"] = "EXACT_ACTION"
    created_at: str
    expires_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    # 二次复核 R1：daemon-backed 卡片的显式链接（创建 daemon proposal
    # 后回填；SIM 卡为空）。P0-6 receipt 精确比对依赖这些字段。
    daemon_proposal_id: str = ""
    daemon_action_id: str = ""
    daemon_capability_id: str = ""
    daemon_action_intent_hash: str = ""
