"""CommandSpecV1 / CommandRequestV1 / CommandResultV1 (补充实施文档 §5.2).

命令是控制协议，不是聊天文本：所有 ``/`` 命令由客户端先解析并路由到
对应 owner 的控制 API；/approve、/login、/estop 等永远不进入模型上下文。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class CommandOwner(StrEnum):
    LOCAL_UI = "LOCAL_UI"  # TUI 本地处理（/clear-screen、/quit）
    AGENT_CONTROL = "AGENT_CONTROL"  # AgentService 通用控制（/cancel）
    MODEL_CONTROL = "MODEL_CONTROL"  # modeld（/model、/login）
    MISSION_CONTROL = "MISSION_CONTROL"  # Mission 语义（/new、/compact、/rename）
    SAFETY_CONTROL = "SAFETY_CONTROL"  # Operator/rosclawd 专用（/estop、/approve）


class CommandCategory(StrEnum):
    HELP_UI = "help_ui"
    MODEL = "model"
    MISSION = "mission"
    SAFETY = "safety"
    EXECUTION = "execution"


class CommandSpecV1(ContractModel):
    SCHEMA = "rosclaw.ui.command_spec.v1"

    schema_version: Literal["rosclaw.ui.command_spec.v1"] = "rosclaw.ui.command_spec.v1"
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    argument_hint: str = ""
    category: CommandCategory = CommandCategory.MISSION
    owner: CommandOwner = CommandOwner.MISSION_CONTROL
    #: mission states where the command is available; empty = always
    availability: list[str] = Field(default_factory=list)
    #: may run while a turn is in flight
    during_turn: bool = False
    #: NONE | CONTROL_STATE | PERSISTED
    mutability: Literal["NONE", "CONTROL_STATE", "PERSISTED"] = "NONE"
    confirmation: Literal["NONE", "CONFIRM", "TYPE_NAME"] = "NONE"
    required_capabilities: list[str] = Field(default_factory=list)
    #: server-side handler id (empty for LOCAL_UI commands)
    handler: str = ""
    #: set when unavailable: human-readable reason shown disabled in UI
    disabled_reason: str = ""


class CommandRequestV1(ContractModel):
    SCHEMA = "rosclaw.ui.command_request.v1"

    schema_version: Literal["rosclaw.ui.command_request.v1"] = "rosclaw.ui.command_request.v1"
    request_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    command_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    mission_id: str | None = None


class CommandResultV1(ContractModel):
    SCHEMA = "rosclaw.ui.command_result.v1"

    schema_version: Literal["rosclaw.ui.command_result.v1"] = "rosclaw.ui.command_result.v1"
    request_id: str
    command_name: str
    ok: bool
    #: human-readable summary for the transcript (never secrets)
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
