"""Task Compiler（十六审 P0-B）：TaskSpec → 结构化 ExecutionPlan。

不变量（审计根因：「装依赖跑脚本」被编译成只读 scout）：
- profile 由结构化需求编译（capabilities + effects +
  runtime_requirements），不按 capability 名称前缀猜；
- 授权是集合包含：required_effects ⊆ granted_effects；不满足 →
  编译期 blocked_reason（启动前 BLOCKED，零 Worker 预算燃烧）；
- profile 是授权信封，不是执行者的智力水平——Runtime 始终是 Pi；
- runtime_requirements（python_packages 等）进编译产物，由 Runtime
  Manager 预置——Worker 不负责装环境（P0-C 接线）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 副作用词表（结构化 effects 只允许这些值——未知值 fail-closed）。
EFFECT_VOCAB = frozenset({
    "process.exec",
    "workspace.write",
    "network.read",
    "network.write",
    "physical.shadow",
    "physical.real",
})

#: 兼容旧枚举字符串 → 细粒度集合。
_EFFECTS_ENUM: dict[str, frozenset[str]] = {
    "": frozenset(),
    "none": frozenset(),
    "workspace_only": frozenset({"process.exec", "workspace.write"}),
    "simulation_only": frozenset({"process.exec", "workspace.write"}),
    "physical_shadow": frozenset({"physical.shadow"}),
    "physical_real": frozenset({"physical.real"}),
}

#: capability → 所需副作用（显式表优先，前缀兜底）。
_CAPABILITY_EFFECTS: dict[str, frozenset[str]] = {
    "code.develop": frozenset({"process.exec", "workspace.write"}),
    "code.implement": frozenset({"process.exec", "workspace.write"}),
    "code.repository_analysis": frozenset(),
    "analysis.text": frozenset(),
    "simulation.build": frozenset({"process.exec", "workspace.write"}),
    "simulation.execute": frozenset({"process.exec", "workspace.write"}),
    "simulation.planar_trajectory": frozenset({"process.exec", "workspace.write"}),
}
_PREFIX_EFFECTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("code.", frozenset({"process.exec", "workspace.write"})),
    ("capability.", frozenset({"process.exec", "workspace.write"})),
    ("simulation.", frozenset({"process.exec", "workspace.write"})),
    ("repo.", frozenset()),
    ("research.", frozenset()),
    ("analysis.", frozenset()),
)

#: profile 授权信封（与实际工具面逐字一致——profiles.ts 是运行时执行）。
PROFILE_GRANTS: dict[str, frozenset[str]] = {
    "developer": frozenset({"process.exec", "workspace.write"}),
    "sim-builder": frozenset({"process.exec", "workspace.write"}),
    "scout": frozenset(),
    "analyst": frozenset(),
}
PROFILE_CAPABILITY = {
    "developer": "code.develop",
    "sim-builder": "simulation.build",
    "scout": "code.repository_analysis",
    "analyst": "analysis.text",
}
PROFILE_EFFECT_CLASS = {
    "developer": "sandbox_process",
    "sim-builder": "sandbox_process",
    "scout": "none",
    "analyst": "none",
}


def _capability_effects(capability: str) -> frozenset[str]:
    if capability in _CAPABILITY_EFFECTS:
        return _CAPABILITY_EFFECTS[capability]
    for prefix, effects in _PREFIX_EFFECTS:
        if capability.startswith(prefix):
            return effects
    # 未知 capability fail-closed：按需要写+执行编译（宁可授权够，
    # 不再出现只读 scout 被要求装软件）；物理副作用永不由猜测引入。
    return frozenset({"process.exec", "workspace.write"})


def normalize_effects(raw: Any) -> frozenset[str]:
    """effects 字段归一：旧枚举字符串或细粒度列表。"""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        if raw not in _EFFECTS_ENUM:
            raise ValueError(f"未知 effects 枚举 {raw!r}")
        return _EFFECTS_ENUM[raw]
    values = {str(v) for v in raw}
    unknown = values - EFFECT_VOCAB
    if unknown:
        raise ValueError(f"未知副作用 {sorted(unknown)}（词表 {sorted(EFFECT_VOCAB)}）")
    return frozenset(values)


@dataclass(frozen=True)
class ExecutionPlan:
    """编译产物：授权信封 + 运行时需求（确定性、可审计）。"""

    profile: str
    capability: str
    effect_class: str
    required_effects: frozenset[str] = field(default_factory=frozenset)
    granted_effects: frozenset[str] = field(default_factory=frozenset)
    runtime_requirements: dict[str, Any] = field(default_factory=dict)
    blocked_reason: str = ""


def compile_task(spec: dict) -> ExecutionPlan:
    """TaskSpec → ExecutionPlan。不可满足时 blocked_reason 非空
    （调用方启动前 BLOCKED，不得抱侥幸执行）。"""
    capabilities = [str(c) for c in (spec.get("required_capabilities") or [])]
    required = set(normalize_effects(spec.get("effects")))
    for capability in capabilities:
        required |= _capability_effects(capability)
    runtime_requirements = dict(spec.get("runtime_requirements") or {})

    physical = {e for e in required if e.startswith("physical.")}
    if physical:
        # 物理副作用不走 harness/executor（router 单独处理）；编译器
        # 只出信封，不授权。
        return ExecutionPlan(
            profile="none",
            capability="physical.proposal",
            effect_class="none",
            required_effects=frozenset(required),
            granted_effects=frozenset(),
            runtime_requirements=runtime_requirements,
        )

    if not required:
        profile = "analyst" if any(c.startswith("analysis.") for c in capabilities) else "scout"
    elif required <= PROFILE_GRANTS["developer"]:
        # 仿真资产构建用 sim-builder（写工具 + 仿真语境 prompt），
        # 其余写任务 developer。
        profile = (
            "sim-builder"
            if any(c.startswith("simulation.") for c in capabilities)
            and not any(c.startswith("code.") for c in capabilities)
            else "developer"
        )
    else:
        excess = sorted(required - PROFILE_GRANTS["developer"])
        return ExecutionPlan(
            profile="none",
            capability="",
            effect_class="none",
            required_effects=frozenset(required),
            granted_effects=frozenset(),
            runtime_requirements=runtime_requirements,
            blocked_reason=(
                f"副作用 {excess} 超出所有内置执行 profile 授权——"
                "网络写入由 Runtime Manager/管理员通道处理，物理副作用走 "
                "rosclawd 准入链；不以只读 scout 侥幸启动"
            ),
        )
    grants = PROFILE_GRANTS[profile]
    return ExecutionPlan(
        profile=profile,
        capability=PROFILE_CAPABILITY[profile],
        effect_class=PROFILE_EFFECT_CLASS[profile],
        required_effects=frozenset(required),
        granted_effects=grants,
        runtime_requirements=runtime_requirements,
    )


#: Worker BLOCKED 报告里的结构化缺能力标记（harness 协议行）。
def missing_capability_of(text: str) -> str:
    """从 BLOCKED 摘要提取 `MISSING CAPABILITY: x` / `missing capability: x`
    （harness 协议标记的确定性解析，非自由文本状态推断）。"""
    import re

    match = re.search(
        r"missing capability:\s*([a-z0-9._-]+)", text or "", re.IGNORECASE
    )
    return match.group(1) if match else ""


#: 缺能力标记 → 所需副作用（升级判定用；network/physical 永不可升级）。
_CAPABILITY_TOKEN_EFFECTS = {
    "process.exec": "process.exec",
    "shell": "process.exec",
    "bash": "process.exec",
    "workspace.write": "workspace.write",
    "write": "workspace.write",
    "edit": "workspace.write",
}


def escalation_profile_for(missing: str, current_profile: str) -> str:
    """缺能力 → 可升级的 profile（无授权路径返回 ""——诚实 BLOCKED）。"""
    effect = _CAPABILITY_TOKEN_EFFECTS.get(missing, "")
    if not effect:
        return ""  # network/physical/未知——无内置升级路径
    for profile in ("developer", "sim-builder"):
        if effect in PROFILE_GRANTS[profile] and profile != current_profile:
            return profile
    return ""
