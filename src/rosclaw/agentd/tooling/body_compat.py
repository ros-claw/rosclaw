"""Body—Capability 兼容性判定（六审 §6.2/PR-SIX-3）。

PHYSICAL_ACTION 不再是 body-agnostic：descriptor 必须声明
required_body_types（body type/body ID scope），且当前绑定本体必须
在 scope 内。缺失 → quarantine；不匹配 → 建卡前拒绝。
"""

from __future__ import annotations

from rosclaw.contracts.agent.tool import ToolDescriptorV2


def check_body_compatibility(
    descriptor: ToolDescriptorV2, body_id: str
) -> str | None:
    """返回 None 表示兼容；否则返回 reason code（fail closed）。"""
    if descriptor.execution_class.value != "PHYSICAL_ACTION":
        return None
    if not descriptor.required_body_types:
        return "BODY_SCOPE_MISSING"
    if body_id not in descriptor.required_body_types:
        return "BODY_CAPABILITY_MISMATCH"
    return None
