"""ToolDescriptorV2 → CapabilityDescriptorV2 迁移适配器（PR-N5A）。

旧 ToolDescriptorV2 是**兼容输入**：catalog 注册时统一适配成
CapabilityDescriptorV2 作为 canonical 存储。N11 再删除旧合约——
不并行维护两套正式目录（canonical 只有一份，legacy 是派生视图）。

映射规则（显式可查，测试钉住）：

| 旧字段 | → V2 |
|---|---|
| PHYSICAL_ACTION | effect.class=PHYSICAL_EFFECT |
| effect_domain=SIMULATION_STATE_ONLY | effect.class=SIMULATED_EFFECT |
| COMPUTE + evidence=SIMULATED | effect.class=SIMULATED_EFFECT |
| COMPUTE 其余 | effect.class=PURE_COMPUTE |
| OBSERVE | effect.class=READ_ONLY |
| side_effect_class NONE/REVERSIBLE | reversible=True |
| side_effect_class IRREVERSIBLE | reversible=False |
| supported_modes / required_body_types / required_capabilities | compatibility |
| idempotent / timeout_ms | execution |
| evidence_class / verifier | evidence |
"""

from __future__ import annotations

from rosclaw.contracts.agent.capability import (
    CapabilityCompatibilityV1,
    CapabilityDescriptorV2,
    CapabilityEffectV1,
    CapabilityEvidenceV1,
    CapabilityExecutionV1,
    EffectClassV1,
)
from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)

_SIM_DOMAIN = "SIMULATION_STATE_ONLY"


def _effect_class_of(d: ToolDescriptorV2) -> EffectClassV1:
    if d.execution_class is ExecutionClass.PHYSICAL_ACTION:
        return EffectClassV1.PHYSICAL_EFFECT
    if d.effect_domain == _SIM_DOMAIN:
        return EffectClassV1.SIMULATED_EFFECT
    if d.execution_class is ExecutionClass.COMPUTE:
        if d.evidence_class is ToolEvidenceClass.SIMULATED:
            return EffectClassV1.SIMULATED_EFFECT
        return EffectClassV1.PURE_COMPUTE
    return EffectClassV1.READ_ONLY


#: legacy effect_domain → canonical 域名（N5C：域名也单一化）。
_DOMAIN_ALIASES = {"simulation_state_only": "simulation_state"}


def _effect_domain_of(d: ToolDescriptorV2, effect: EffectClassV1) -> str:
    if d.effect_domain:
        lowered = d.effect_domain.lower()
        return _DOMAIN_ALIASES.get(lowered, lowered)
    if effect is EffectClassV1.SIMULATED_EFFECT:
        return "simulation_state"
    if effect is EffectClassV1.PHYSICAL_EFFECT:
        return "physical_body"
    return ""


def capability_from_tool_descriptor(d: ToolDescriptorV2) -> CapabilityDescriptorV2:
    """单个旧描述符 → V2 能力（诚实标注 adapted_from）。"""
    effect_class = _effect_class_of(d)
    return CapabilityDescriptorV2(
        capability_id=d.tool_id,
        version=d.version,
        source=d.source,
        description=d.description,
        input_schema=dict(d.input_schema),
        output_schema=dict(d.output_schema),
        effect=CapabilityEffectV1(
            **{
                "class": effect_class,
                "domain": _effect_domain_of(d, effect_class),
                "reversible": d.side_effect_class
                is not ToolSideEffectClass.IRREVERSIBLE,
                "risk_tier": d.risk_tier,
                "requires_fresh_real_context": (
                    effect_class is EffectClassV1.PHYSICAL_EFFECT
                    and d.freshness_ms is not None
                ),
            }
        ),
        compatibility=CapabilityCompatibilityV1(
            modes=list(d.supported_modes),
            body_types=list(d.required_body_types),
            runtime_requirements=list(d.required_capabilities),
        ),
        execution=CapabilityExecutionV1(
            long_running=d.typical_latency_ms >= 1000,
            idempotent=d.idempotent,
        ),
        evidence=CapabilityEvidenceV1(
            evidence_class=d.evidence_class.value,
            verifier_ref=d.verifier,
        ),
        metadata={"adapted_from": ToolDescriptorV2.SCHEMA},
    )


def tool_descriptor_from_capability(cap: CapabilityDescriptorV2) -> ToolDescriptorV2:
    """V2 → legacy 视图（过渡期服务未迁移消费者；不做新增事实）。

    注意：这是有损投影（execution/compatibility 细节丢失）——legacy
    视图只供过滤/展示，不作安全判定的唯一来源。
    """
    effect = cap.effect.class_
    if effect is EffectClassV1.PHYSICAL_EFFECT:
        execution_class = ExecutionClass.PHYSICAL_ACTION
    elif effect in (EffectClassV1.PURE_COMPUTE, EffectClassV1.SIMULATED_EFFECT):
        execution_class = ExecutionClass.COMPUTE
    else:
        execution_class = ExecutionClass.OBSERVE
    model_callable = execution_class is not ExecutionClass.PHYSICAL_ACTION
    # legacy 不变量：OBSERVE 必须 side_effect NONE（N5D 实证：
    # HOST_PROCESS 等能力 downgrade 成 OBSERVE+REVERSIBLE 会炸）。
    side_effect = (
        ToolSideEffectClass.NONE
        if execution_class is ExecutionClass.OBSERVE
        else (
            ToolSideEffectClass.REVERSIBLE
            if cap.effect.reversible
            else ToolSideEffectClass.IRREVERSIBLE
        )
    )
    return ToolDescriptorV2(
        tool_id=cap.capability_id,
        version=cap.version,
        source=cap.source,
        execution_class=execution_class,
        side_effect_class=side_effect,
        effect_domain=(
            "SIMULATION_STATE_ONLY"
            if effect is EffectClassV1.SIMULATED_EFFECT
            else ""
        ),
        description=cap.description,
        input_schema=dict(cap.input_schema),
        output_schema=dict(cap.output_schema),
        supported_modes=list(cap.compatibility.modes),
        required_body_types=list(cap.compatibility.body_types),
        risk_tier=cap.effect.risk_tier,
        evidence_class=(
            ToolEvidenceClass.SIMULATED
            if cap.evidence.evidence_class in ("SIMULATED", "SIM_DYN_ROLLOUT")
            else ToolEvidenceClass.DERIVED
            if cap.evidence.evidence_class == "DERIVED"
            else ToolEvidenceClass.CONFIGURED
            if cap.evidence.evidence_class == "CONFIGURED"
            else ToolEvidenceClass.MEASURED
        ),
        verifier=cap.evidence.verifier_ref,
        idempotent=cap.execution.idempotent,
        model_callable=model_callable,
        requires_exact_action_grant=not model_callable,
        required_capabilities=list(cap.compatibility.runtime_requirements),
    )


__all__ = ["capability_from_tool_descriptor", "tool_descriptor_from_capability"]
