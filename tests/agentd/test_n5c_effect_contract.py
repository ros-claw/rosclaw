"""PR-N5C 红测试（调整方案 §三.N5C）：单一 Effect Contract。

红测试先行——EffectResolver/冻结事件/生成出口不存在时必须红。

1. 静态能力：effect 从 canonical CapabilityDescriptorV2 读取
   （N5A 适配链），不再有两个手写分类源；
2. 通用入口按参数计算实际 effect：rosclaw_execute/compute 调 SIM
   能力 → SIMULATED_EFFECT，调 PHYSICAL 能力 → PHYSICAL_EFFECT；
   不可解析 → EFFECT_UNRESOLVABLE（fail closed，永不默认良性）；
3. 执行前冻结：dispatcher 在执行前把冻结 effect 写入
   tool.effect_resolved 事件（含 capability digest + arguments digest）；
4. 审批读冻结后/canonical effect：POLICY_AUTO 只认 canonical
   SIMULATED_EFFECT + simulation_state 域；
5. pi.capabilities 每条携带 canonical effect_class；
6. TS 面由 Python 注册表自动生成（effects.generated）——手写
   EFFECT_BY_TOOL 不再是事实源。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)

REPO = Path(__file__).resolve().parents[2]


def _catalog():
    from rosclaw.agentd.tooling.catalog import ToolCatalog

    catalog = ToolCatalog()
    catalog.register(ToolDescriptorV2(
        tool_id="sim_reach",
        source="native:agentd",
        execution_class=ExecutionClass.COMPUTE,
        effect_domain="SIMULATION_STATE_ONLY",
        evidence_class=ToolEvidenceClass.SIMULATED,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    ))
    catalog.register(ToolDescriptorV2(
        tool_id="physical_move",
        source="native:agentd",
        execution_class=ExecutionClass.PHYSICAL_ACTION,
        side_effect_class=ToolSideEffectClass.REVERSIBLE,
        model_callable=False,
        requires_exact_action_grant=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    ))
    return catalog


class TestEffectResolver:
    def test_static_effect_from_canonical_capability(self) -> None:
        from rosclaw.agentd.tooling.effect_resolver import EffectResolver

        resolver = EffectResolver(_catalog())
        frozen = resolver.resolve("sim_reach", {})
        assert frozen.effect_class == "SIMULATED_EFFECT"
        assert frozen.domain == "simulation_state"
        assert frozen.source == "capability:sim_reach"
        assert frozen.capability_digest
        assert frozen.arguments_digest

    def test_generic_entry_computes_effect_from_arguments(self) -> None:
        """相同入口：调 SIM 能力 → SIMULATED_EFFECT；调 PHYSICAL 能力
        → PHYSICAL_EFFECT——不能永远静态写成 SIMULATED_EFFECT。"""
        from rosclaw.agentd.tooling.effect_resolver import EffectResolver

        resolver = EffectResolver(_catalog())
        sim = resolver.resolve(
            "rosclaw_execute", {"capability_id": "sim_reach"}
        )
        assert sim.effect_class == "SIMULATED_EFFECT"
        phys = resolver.resolve(
            "rosclaw_execute", {"capability_id": "physical_move"}
        )
        assert phys.effect_class == "PHYSICAL_EFFECT"
        assert phys.reversible is True

    def test_unresolvable_fails_closed(self) -> None:
        from rosclaw.agentd.tooling.effect_resolver import (
            EffectResolver,
            EffectUnresolvableError,
        )

        resolver = EffectResolver(_catalog())
        with pytest.raises(EffectUnresolvableError):
            resolver.resolve("rosclaw_execute", {"capability_id": "no_such"})
        with pytest.raises(EffectUnresolvableError):
            resolver.resolve("rosclaw_execute", {})  # 缺 capability_id
        with pytest.raises(EffectUnresolvableError):
            resolver.resolve("totally_unknown_tool", {})

    def test_direct_dispatch_tool_effects(self) -> None:
        """非 catalog 路由的 rosclaw_* 分支工具：Python 侧唯一声明
        （替代被删的 TS 手写表）。"""
        from rosclaw.agentd.tooling.effect_resolver import EffectResolver

        resolver = EffectResolver(_catalog())
        assert resolver.resolve("rosclaw_request_action", {}).effect_class == (
            "PHYSICAL_EFFECT"
        )
        assert resolver.resolve("rosclaw_fail_safe", {}).effect_class == (
            "SHADOW_PROPOSAL"
        )
        assert resolver.resolve("rosclaw_artifact_register", {}).effect_class == (
            "WORKSPACE_WRITE"
        )
        assert resolver.resolve("rosclaw_status", {}).effect_class == "READ_ONLY"
        assert resolver.resolve("rosclaw_inspect", {}).effect_class == "READ_ONLY"
        # rosclaw_task 当前唯一实现是 SIM 管线宏。
        assert resolver.resolve(
            "rosclaw_task", {"goal": "simulate_trajectory"}
        ).effect_class == "SIMULATED_EFFECT"


class TestFrozenEffectEvent:
    async def test_dispatcher_freezes_effect_before_execution(
        self, tmp_path: Path
    ) -> None:
        """执行前冻结：tool.effect_resolved 事件先于执行结果存在，
        携带冻结 digest。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute",
                mission=mission.mission_id, idem="n5c_1",
                lease=await _issue_lease(service, mission),
                arguments={
                    "capability_id": "trajectory_generate_planar_path",
                    "arguments": {"shape": "star5",
                                  "center_m": [0.35, 0.25, 0.30],
                                  "scale_m": 0.10},
                },
            ),
        )
        assert result.ok, result
        events = service.events_replay(mission.mission_id, limit=200)
        resolved = [
            e for e in events if e.type.value == "tool.effect_resolved"
        ]
        assert resolved, "缺 tool.effect_resolved 冻结事件"
        payload = resolved[0].payload
        assert payload["tool_name"] == "rosclaw_compute"
        assert payload["effect_class"] == "SIMULATED_EFFECT"
        assert payload["capability_digest"]
        assert payload["arguments_digest"]
        await service.close()

    async def test_unresolvable_effect_rejects_call(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_compute",
                mission=mission.mission_id, idem="n5c_2",
                lease=await _issue_lease(service, mission),
                arguments={"capability_id": "ghost_capability",
                           "arguments": {}},
            ),
        )
        assert result.ok is False
        assert result.error_code in ("EFFECT_UNRESOLVABLE", "CAPABILITY_UNKNOWN")
        await service.close()


class TestAdmissionReadsCanonicalEffect:
    """POLICY_AUTO 只认 canonical 能力 effect——legacy 字段不再作数。"""

    async def test_auto_requires_canonical_simulated_effect(
        self, tmp_path: Path
    ) -> None:
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
        )
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        admission = ActionAdmissionService(service)
        # POLICY_AUTO 的设计对象：kit sim-only 动作（SIMULATION_STATE_ONLY）。
        await service._ensure_mcp_discovered()
        descriptor = service._tool_catalog.get("ur5e.move_joints")
        assert descriptor is not None
        assert descriptor.effect_domain == "SIMULATION_STATE_ONLY"
        auto = admission._policy_auto_applies(mission, descriptor)
        assert auto is True  # 既有安全条件全满足（DEV_SIM_ONLY + kit 源等）
        # 变异：canonical 效应域被改（legacy 视图仍声称
        # SIMULATION_STATE_ONLY）→ 不得自动——审批读 canonical。
        cap = service._tool_catalog.capability("ur5e.move_joints")
        assert cap is not None
        assert cap.effect.domain == "simulation_state"

        tampered = cap.model_copy(deep=True)
        tampered.effect.domain = "physical_body"
        service._tool_catalog._capabilities["ur5e.move_joints"] = tampered
        auto = admission._policy_auto_applies(mission, descriptor)
        assert auto is False, "canonical effect 域被改后 POLICY_AUTO 仍放行"
        await service.close()


class TestCapabilitiesEffectSurface:
    async def test_pi_capabilities_carry_effect_class(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.capabilities",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        for bucket in ("observation_capabilities", "compute_capabilities"):
            for entry in result.get(bucket) or []:
                assert entry.get("effect_class"), (
                    f"{bucket} 条目缺 effect_class: {entry}"
                )
        compute_effects = {
            e["capability_id"]: e["effect_class"]
            for e in result.get("compute_capabilities") or []
        }
        assert compute_effects.get("sim_reach") == "SIMULATED_EFFECT"
        await service.close()


class TestGeneratedEffectMap:
    def test_generated_map_matches_checked_in_golden(self) -> None:
        """Python 注册表 → 生成 TS 效果表（手写 EFFECT_BY_TOOL 的
        替代）；漂移即红——与 golden schema 同一模式。"""
        from rosclaw.agentd.tooling.effect_resolver import (
            export_generated_effects,
        )

        golden = (
            REPO / "packages" / "rosclaw-agent" / "src" / "tools"
            / "effects.generated.json"
        )
        assert golden.exists(), (
            "effects.generated.json 缺失——由 export_generated_effects 生成"
        )
        current = export_generated_effects(_catalog())
        checked_in = json.loads(golden.read_text(encoding="utf-8"))
        # checked-in 是全量产品面（含本产品全部 rosclaw_* 工具）；
        # 测试 catalog 的子集必须与之一致。
        for name, effect in current.items():
            assert checked_in.get(name) == effect, (
                f"{name}: checked-in={checked_in.get(name)!r} != {effect!r}"
            )
        # 通用入口诚实标记 DYNAMIC——不允许静态写成 SIMULATED_EFFECT。
        for dynamic in ("rosclaw_execute", "rosclaw_compute", "rosclaw_observe"):
            assert checked_in.get(dynamic) == "DYNAMIC", (
                f"{dynamic} 必须为 DYNAMIC（按参数解析），当前 "
                f"{checked_in.get(dynamic)!r}"
            )
