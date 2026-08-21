"""ToolResultEnvelopeV2（PR-N5B，调整方案 §三.N5B）——canonical 输出合约。

每个模型可调用能力必须声明 output_schema；执行结果统一为 canonical
envelope。模型文本、UI 展示、CLI JSON 都是 canonical value 的投影；
executor 不得返回裸字符串冒充结构化结果；无效输出统一转为
INVALID_CAPABILITY_OUTPUT（诚实 FAILED，不抛出、不猜测）。

``presentation_meta`` 只能由可信 projection（Registry/展示层）生成——
executor 或模型提交即拒绝（Registry 层强制）。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class ToolResultStatusV1(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"


class ToolResultErrorV1(ContractModel):
    """稳定错误码——模型/Harness 据 code 做恢复决策，不解析 message。"""

    code: str = Field(min_length=1)
    message: str = ""
    retryable: bool = False
    recovery: list[str] = Field(default_factory=list)


class ToolResultEnvelopeV2(ContractModel):
    """一次能力调用的 canonical 结果。"""

    SCHEMA: ClassVar[str] = "rosclaw.tool_result_envelope.v2"
    HASH_PREFIX: ClassVar[str] = "tool_result_envelope"

    schema_version: Literal["rosclaw.tool_result_envelope.v2"] = (
        "rosclaw.tool_result_envelope.v2"
    )
    call_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    status: ToolResultStatusV1
    value: dict[str, Any] | None = None
    error: ToolResultErrorV1 | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    operation_id: str | None = None
    #: 只能由可信 projection 生成；模型/executor 永远不得提交。
    presentation_meta: Any | None = None


__all__ = ["ToolResultEnvelopeV2", "ToolResultErrorV1", "ToolResultStatusV1"]
