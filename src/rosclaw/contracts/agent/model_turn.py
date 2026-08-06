"""ModelTurnResultV1 — unified model turn output (总纲 §7.1).

The full assistant message is preserved verbatim for provider-specific
reasoning continuity (Kimi K3 requires appending the complete assistant
message in tool loops), but public traces must apply field-level redaction
before persisting ``assistant_message``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class ToolCall(ContractModel):
    SCHEMA = "rosclaw.tool_call.v1"

    schema_version: Literal["rosclaw.tool_call.v1"] = "rosclaw.tool_call.v1"
    call_id: str
    name: str
    arguments_json: str = "{}"


class ModelUsage(ContractModel):
    SCHEMA = "rosclaw.model_usage.v1"

    schema_version: Literal["rosclaw.model_usage.v1"] = "rosclaw.model_usage.v1"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_microunits: int = 0


class ModelTurnResultV1(ContractModel):
    SCHEMA = "rosclaw.model_turn.v1"
    HASH_PREFIX = "mturn"

    schema_version: Literal["rosclaw.model_turn.v1"] = "rosclaw.model_turn.v1"
    turn_id: str
    mission_id: str | None = None
    provider: str
    model: str
    profile: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Verbatim provider assistant message (protocol continuity). Redact
    #: before writing to public traces.
    assistant_message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    provider_request_id: str | None = None
    latency_ms: int = 0
    context_id: str | None = None
    context_revision: int | None = None
