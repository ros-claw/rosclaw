"""PR-N5A 红测试（调整方案 §三.N5A）：Capability Registry 组件/接线/变异层。

红测试先行——适配器与 canonical 存储不存在时必须红。

1. 迁移适配器映射表：ToolDescriptorV2（兼容输入）→
   CapabilityDescriptorV2（canonical），映射规则显式可查；
2. ToolCatalog 内部以 CapabilityDescriptorV2 为 canonical 存储——
   register(legacy) 后 capability() 给出映射结果，legacy 视图不变；
3. 投影闸：PHYSICAL_EFFECT 能力拒绝 direct 投影（fail closed），
   propose_only 合法；
4. 变异：effect 变化 → capability digest 变化（审批/并发/Verifier
   依赖冻结 digest）；
5. 产品接线：内置 sim 工具经 register_native_tools 后，
   sim_reach 的 canonical 能力 effect.class = SIMULATED_EFFECT。
"""

from __future__ import annotations

import pytest

from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)
from rosclaw.contracts.common import ValidationError


def _legacy(**overrides) -> ToolDescriptorV2:
    payload = {
        "tool_id": "sim_reach",
        "source": "native:agentd",
        "execution_class": ExecutionClass.COMPUTE,
        "description": "MuJoCo reach",
        "input_schema": {"type": "object", "additionalProperties": False},
        "output_schema": {"type": "object", "additionalProperties": False},
        "supported_modes": ["SIMULATION"],
        "evidence_class": ToolEvidenceClass.SIMULATED,
        "verifier": "sandbox-receipt+task-predicate",
        "risk_tier": "LOW",
    }
    payload.update(overrides)
    return ToolDescriptorV2(**payload)


class TestMigrationAdapter:
    """映射表显式可查——任何改动都会被本表钉住。"""

    @pytest.mark.parametrize(
        ("execution_class", "side_effect", "evidence", "effect_domain",
         "expected_effect", "expected_reversible"),
        [
            (ExecutionClass.OBSERVE, ToolSideEffectClass.NONE,
             ToolEvidenceClass.MEASURED, "", "READ_ONLY", True),
            (ExecutionClass.COMPUTE, ToolSideEffectClass.NONE,
             ToolEvidenceClass.DERIVED, "", "PURE_COMPUTE", True),
            (ExecutionClass.COMPUTE, ToolSideEffectClass.NONE,
             ToolEvidenceClass.SIMULATED, "", "SIMULATED_EFFECT", True),
            (ExecutionClass.COMPUTE, ToolSideEffectClass.REVERSIBLE,
             ToolEvidenceClass.MEASURED, "SIMULATION_STATE_ONLY",
             "SIMULATED_EFFECT", True),
            (ExecutionClass.PHYSICAL_ACTION, ToolSideEffectClass.REVERSIBLE,
             ToolEvidenceClass.MEASURED, "", "PHYSICAL_EFFECT", True),
            (ExecutionClass.PHYSICAL_ACTION, ToolSideEffectClass.IRREVERSIBLE,
             ToolEvidenceClass.MEASURED, "", "PHYSICAL_EFFECT", False),
        ],
    )
    def test_mapping_table(
        self, execution_class, side_effect, evidence, effect_domain,
        expected_effect, expected_reversible,
    ) -> None:
        from rosclaw.agentd.tooling.capability_adapter import (
            capability_from_tool_descriptor,
        )

        legacy = _legacy(
            execution_class=execution_class,
            side_effect_class=side_effect,
            evidence_class=evidence,
            effect_domain=effect_domain,
            model_callable=execution_class is not ExecutionClass.PHYSICAL_ACTION,
            requires_exact_action_grant=(
                execution_class is ExecutionClass.PHYSICAL_ACTION
            ),
        )
        cap = capability_from_tool_descriptor(legacy)
        assert cap.schema_version == "rosclaw.capability.v2"
        assert cap.capability_id == legacy.tool_id
        assert cap.effect.class_.value == expected_effect
        assert cap.effect.reversible is expected_reversible
        assert cap.effect.risk_tier == legacy.risk_tier
        # 兼容块映射
        assert list(cap.compatibility.modes) == list(legacy.supported_modes)
        assert cap.execution.idempotent == legacy.idempotent
        assert cap.evidence.evidence_class == legacy.evidence_class.value
        assert cap.evidence.verifier_ref == legacy.verifier
        # 来源标记：兼容输入必须诚实标注
        assert cap.metadata.get("adapted_from") == "rosclaw.tool_descriptor.v2"


class TestCatalogCanonicalV2:
    def test_register_legacy_stores_capability_v2_canonical(self) -> None:
        from rosclaw.agentd.tooling.catalog import ToolCatalog

        catalog = ToolCatalog()
        catalog.register(_legacy())
        cap = catalog.capability("sim_reach")
        assert cap is not None
        assert cap.capability_id == "sim_reach"
        assert cap.effect.class_.value == "SIMULATED_EFFECT"
        # legacy 视图保持不变（resolver/dispatch 未迁移期）
        legacy = catalog.get("sim_reach")
        assert legacy is not None
        assert legacy.execution_class is ExecutionClass.COMPUTE

    def test_register_capability_v2_directly(self) -> None:
        from rosclaw.agentd.tooling.capability_adapter import (
            capability_from_tool_descriptor,
        )
        from rosclaw.agentd.tooling.catalog import ToolCatalog

        catalog = ToolCatalog()
        cap = capability_from_tool_descriptor(_legacy(tool_id="sim_state"))
        catalog.register_capability(cap)
        assert catalog.capability("sim_state") is not None
        # V2 注册同样给出 legacy 视图（过渡期为旧消费者服务）
        assert catalog.get("sim_state") is not None


class TestProjectionGuard:
    def _physical_capability(self):
        from rosclaw.agentd.tooling.capability_adapter import (
            capability_from_tool_descriptor,
        )

        return capability_from_tool_descriptor(_legacy(
            tool_id="physical_move",
            execution_class=ExecutionClass.PHYSICAL_ACTION,
            side_effect_class=ToolSideEffectClass.REVERSIBLE,
            model_callable=False,
            requires_exact_action_grant=True,
        ))

    def test_physical_effect_rejects_direct_projection(self) -> None:
        """PHYSICAL_EFFECT 原始 executor 永不直接暴露（fail closed）。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.contracts.agent.capability import (
            ProjectionExposure,
            ToolProjectionV1,
        )

        catalog = ToolCatalog()
        cap = self._physical_capability()
        catalog.register_capability(cap)
        direct = ToolProjectionV1(
            tool_name="physical_move",
            capability_id=cap.capability_id,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            exposure=ProjectionExposure.DIRECT,
        )
        with pytest.raises(ValidationError):
            catalog.register_projection(direct)

    def test_physical_effect_propose_only_projection_ok(self) -> None:
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.contracts.agent.capability import (
            ProjectionExposure,
            ToolProjectionV1,
        )

        catalog = ToolCatalog()
        cap = self._physical_capability()
        catalog.register_capability(cap)
        propose = ToolProjectionV1(
            tool_name="propose_physical_move",
            capability_id=cap.capability_id,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            exposure=ProjectionExposure.PROPOSE_ONLY,
        )
        catalog.register_projection(propose)
        assert catalog.projection("propose_physical_move") is not None

    def test_simulated_effect_direct_projection_ok(self) -> None:
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.contracts.agent.capability import (
            ProjectionExposure,
            ToolProjectionV1,
        )

        catalog = ToolCatalog()
        catalog.register(_legacy())
        direct = ToolProjectionV1(
            tool_name="sim_reach",
            capability_id="sim_reach",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            exposure=ProjectionExposure.DIRECT,
        )
        catalog.register_projection(direct)


class TestMutation:
    def test_effect_change_changes_digest(self) -> None:
        """拆掉一根关键连线（effect 篡改）→ digest 必须变化。"""
        from rosclaw.agentd.tooling.capability_adapter import (
            capability_from_tool_descriptor,
        )
        from rosclaw.contracts.agent.capability import EffectClassV1

        cap = capability_from_tool_descriptor(_legacy())
        tampered = cap.model_copy(deep=True)
        tampered.effect.class_ = EffectClassV1.READ_ONLY
        assert tampered.canonical_hash() != cap.canonical_hash()


class TestProductWiring:
    def test_native_sim_tools_have_canonical_capabilities(self) -> None:
        """内置工具经 register_native_tools 注册后，canonical 能力可读
        且 sim_reach effect = SIMULATED_EFFECT。"""
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tools import SIM_REACH_TOOL, BuiltinToolRegistry

        catalog = ToolCatalog()
        register_native_tools(
            catalog,
            BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e"),
            simulation=True,
        )
        cap = catalog.capability(SIM_REACH_TOOL)
        assert cap is not None
        assert cap.effect.class_.value == "SIMULATED_EFFECT"
        assert cap.evidence.evidence_class == "SIMULATED"
