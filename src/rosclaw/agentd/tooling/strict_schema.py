"""ToolDescriptorV2 → StrictTool conversion (model-facing schema).

Only model-callable OBSERVE/COMPUTE tools may be converted; attempting to
convert a PHYSICAL_ACTION descriptor raises (defense in depth on top of the
contract validator and the resolver hard filter).
"""

from __future__ import annotations

from rosclaw.agentd.models.gateway import StrictTool
from rosclaw.contracts.agent.tool import ExecutionClass, ToolDescriptorV2
from rosclaw.contracts.common import ValidationError


#: OpenAI function-name 约束（^[a-zA-Z][a-zA-Z0-9_-]*$）不允许点号——
#: wire 上把 MCP 工具 id 的 "." 映射为 "__"（catalog 注册时拒绝含 "__" 的
#: 原生 id，保证映射单射）。模型只见 sanitized 名；执行时反解回 canonical。
def wire_name(tool_id: str) -> str:
    return tool_id.replace(".", "__")


def canonical_name(wire: str) -> str:
    return wire.replace("__", ".")


def to_strict_tool(descriptor: ToolDescriptorV2) -> StrictTool:
    if descriptor.execution_class is ExecutionClass.PHYSICAL_ACTION:
        raise ValidationError(
            f"tool {descriptor.tool_id!r}: PHYSICAL_ACTION can never become a model tool"
        )
    if not descriptor.model_callable:
        raise ValidationError(f"tool {descriptor.tool_id!r}: not model_callable")
    schema = dict(descriptor.input_schema) if descriptor.input_schema else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    props = schema.get("properties", {})
    # OpenAI strict convention: every property listed in required.
    schema["required"] = list(props.keys())
    description = descriptor.description or descriptor.tool_id
    description += (
        f" [id: {descriptor.tool_id}; evidence_class: {descriptor.evidence_class.value}; "
        f"modes: {','.join(descriptor.supported_modes)}]"
    )
    return StrictTool(name=wire_name(descriptor.tool_id), description=description, parameters=schema)
