"""EffectResolver（PR-N5C，调整方案 §三.N5C）——单一 Effect Contract。

唯一事实源：
- catalog 路由能力 → canonical CapabilityDescriptorV2.effect（N5A 适
  配链）；
- 通用入口（rosclaw_observe/compute/execute）→ 按参数中的
  capability_id 计算实际 effect（相同入口调 SIM/REAL 能力 effect
  不同——不得静态写死）；
- dispatcher 直分支 rosclaw_* 工具 → 下方 DIRECT_TOOL_EFFECTS
  （Python 唯一声明，替代被删的 TS 手写 EFFECT_BY_TOOL）；
- TS 面由 export_generated_effects 生成（effects.generated.json）——
  手写分类表不存在第二份。

执行前冻结：dispatcher 把 resolve 结果写 tool.effect_resolved 事件；
审批/并发/Verifier 读冻结结果或同一 resolver（同输入同输出）。

不可解析一律 fail closed（EffectUnresolvableError）——永不默认良性。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from rosclaw.contracts.agent.capability import EffectClassV1
from rosclaw.contracts.common import ValidationError


class EffectUnresolvableError(ValidationError):
    """effect 不可解析——fail closed，调用必须被拒绝。"""


@dataclass(frozen=True)
class FrozenEffect:
    """一次调用在执行前冻结的 effect（写事件；审批/并发/Verifier 共读）。"""

    tool_name: str
    effect_class: str
    domain: str
    reversible: bool
    risk_tier: str
    #: capability:<id>（catalog 路由）或 dispatch:<name>（直分支声明）
    source: str
    capability_digest: str
    arguments_digest: str

    def to_event_payload(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "effect_class": self.effect_class,
            "domain": self.domain,
            "reversible": self.reversible,
            "risk_tier": self.risk_tier,
            "source": self.source,
            "capability_digest": self.capability_digest,
            "arguments_digest": self.arguments_digest,
        }


#: 通用入口——effect 由 arguments["capability_id"] 指向的能力决定。
GENERIC_ENTRY_TOOLS = frozenset({
    "rosclaw_observe",
    "rosclaw_compute",
    "rosclaw_execute",
})

#: dispatcher 直分支工具的唯一 effect 声明（这些工具不经 catalog
#: 路由）。TS 手写 EFFECT_BY_TOOL 已删除，以此为准。
DIRECT_TOOL_EFFECTS: dict[str, EffectClassV1] = {
    "rosclaw_status": EffectClassV1.READ_ONLY,
    "rosclaw_capabilities": EffectClassV1.READ_ONLY,
    "rosclaw_verify": EffectClassV1.READ_ONLY,
    "rosclaw_wait_operation": EffectClassV1.READ_ONLY,
    "rosclaw_stop_operation": EffectClassV1.HOST_PROCESS,
    "rosclaw_memory_query": EffectClassV1.READ_ONLY,
    "rosclaw_inspect": EffectClassV1.READ_ONLY,
    "rosclaw_fail_safe": EffectClassV1.SHADOW_PROPOSAL,
    "rosclaw_request_action": EffectClassV1.PHYSICAL_EFFECT,
    "rosclaw_process_start": EffectClassV1.HOST_PROCESS,
    "rosclaw_process_status": EffectClassV1.READ_ONLY,
    "rosclaw_process_output": EffectClassV1.READ_ONLY,
    "rosclaw_process_stop": EffectClassV1.HOST_PROCESS,
    "rosclaw_artifact_register": EffectClassV1.WORKSPACE_WRITE,
    # P0-D：模型面唯一交付入口（幂等 deliver——与 register 同效应）。
    "rosclaw_deliver": EffectClassV1.WORKSPACE_WRITE,
    "rosclaw_task_finish": EffectClassV1.WORKSPACE_WRITE,
    "rosclaw_task_blocked": EffectClassV1.WORKSPACE_WRITE,
    # SIM 管线宏（当前唯一实现）；goal 变化不改变仿真效应本质。
    "rosclaw_task": EffectClassV1.SIMULATED_EFFECT,
}


def _arguments_digest(arguments: dict) -> str:
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EffectResolver:
    """从 canonical 能力注册表解析调用 effect。"""

    def __init__(self, catalog) -> None:
        self._catalog = catalog

    def resolve(self, tool_name: str, arguments: dict) -> FrozenEffect:
        args_digest = _arguments_digest(arguments)
        if tool_name in GENERIC_ENTRY_TOOLS:
            capability_id = str(arguments.get("capability_id", ""))
            if not capability_id:
                raise EffectUnresolvableError(
                    f"{tool_name} 缺 capability_id——effect 不可解析"
                )
            cap = self._catalog.capability(capability_id)
            if cap is None:
                raise EffectUnresolvableError(
                    f"{tool_name} 指向未知能力 {capability_id!r}——fail closed"
                )
            return FrozenEffect(
                tool_name=tool_name,
                effect_class=cap.effect.class_.value,
                domain=cap.effect.domain,
                reversible=cap.effect.reversible,
                risk_tier=cap.effect.risk_tier,
                source=f"capability:{cap.capability_id}",
                capability_digest=cap.canonical_hash(),
                arguments_digest=args_digest,
            )
        if tool_name in DIRECT_TOOL_EFFECTS:
            effect = DIRECT_TOOL_EFFECTS[tool_name]
            cap = self._catalog.capability(tool_name)
            return FrozenEffect(
                tool_name=tool_name,
                effect_class=effect.value,
                domain=(
                    cap.effect.domain if cap is not None
                    else ("simulation_state"
                          if effect is EffectClassV1.SIMULATED_EFFECT else "")
                ),
                reversible=True,
                risk_tier=(
                    cap.effect.risk_tier if cap is not None else "LOW"
                ),
                source=f"dispatch:{tool_name}",
                capability_digest=(
                    cap.canonical_hash() if cap is not None else ""
                ),
                arguments_digest=args_digest,
            )
        cap = self._catalog.capability(tool_name)
        if cap is not None:
            return FrozenEffect(
                tool_name=tool_name,
                effect_class=cap.effect.class_.value,
                domain=cap.effect.domain,
                reversible=cap.effect.reversible,
                risk_tier=cap.effect.risk_tier,
                source=f"capability:{cap.capability_id}",
                capability_digest=cap.canonical_hash(),
                arguments_digest=args_digest,
            )
        raise EffectUnresolvableError(
            f"tool {tool_name!r} 无 effect 来源——fail closed"
        )


def export_generated_effects(catalog=None) -> dict[str, str]:
    """生成 TS 效果表内容（rosclaw_* 产品面）。

    通用入口诚实标记 DYNAMIC（运行时按参数解析）；直分支工具取
    DIRECT_TOOL_EFFECTS。workspace 原语（read/bash/…）是 TS 原生，
    不在此表——由 workspace-pack 同位声明。
    """
    generated: dict[str, str] = {
        name: effect.value for name, effect in sorted(DIRECT_TOOL_EFFECTS.items())
    }
    for name in sorted(GENERIC_ENTRY_TOOLS):
        generated[name] = "DYNAMIC"
    return generated


__all__ = [
    "DIRECT_TOOL_EFFECTS",
    "EffectResolver",
    "EffectUnresolvableError",
    "FrozenEffect",
    "GENERIC_ENTRY_TOOLS",
    "export_generated_effects",
]
