"""Tool Bridge 合约（重构规格 §17，PR-PNA-3）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class PiToolRequestV1(ContractModel):
    SCHEMA = "rosclaw.pi_tool_request.v1"

    schema_version: Literal["rosclaw.pi_tool_request.v1"] = "rosclaw.pi_tool_request.v1"
    request_id: str
    pi_session_id: str
    mission_id: str
    context_revision: int = 0
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_at: str
    idempotency_key: str
    actor: dict[str, Any] = Field(default_factory=dict)


class PiToolResultV1(ContractModel):
    SCHEMA = "rosclaw.pi_tool_result.v1"

    schema_version: Literal["rosclaw.pi_tool_result.v1"] = "rosclaw.pi_tool_result.v1"
    request_id: str
    ok: bool
    status: str
    summary: str = ""
    decision_id: str | None = None
    mission_revision: int = 0
    context_revision: int = 0
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    approval_id: str | None = None
    receipt_id: str | None = None
    retryable: bool = False
    error_code: str | None = None
