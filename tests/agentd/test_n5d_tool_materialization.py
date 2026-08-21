"""PR-N5D 红测试（调整方案 §三.N5D）：动态工具物化。

红测试先行——snapshot 构建/命名/曝光规则/桥接端点/摘要校验不存在
时必须红。

核心规则：
1. Capability Registry → 按 body/mode/health 过滤 → CapabilitySnapshot
   （generation + digest + body_id + mode + active + excluded reasons）；
2. READ_ONLY/PURE_COMPUTE/SIMULATED_EFFECT → direct 精确强类型工具；
   PHYSICAL_EFFECT → propose_<slug>（propose_only，进 ActionAdmission）；
   不可曝光的 effect → excluded（fail closed，附机器原因码）；
3. 工具名稳定：capability_id 的 "." → "__"（沿用 wire 约定）；碰撞
   追加 digest 后缀；
4. excluded 能力不进模型工具面，只能经 inspect 查原因；
5. 执行携带 snapshot digest：registry 变了 → CAPABILITY_SNAPSHOT_CHANGED
   （不静默换工具），下一步重新规划一次。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)


def _catalog():
    from rosclaw.agentd.tooling.catalog import ToolCatalog

    catalog = ToolCatalog()
    catalog.register(ToolDescriptorV2(
        tool_id="ur5e.plan_cartesian_path",
        source="mcp:ur5e-sim",
        execution_class=ExecutionClass.COMPUTE,
        evidence_class=ToolEvidenceClass.DERIVED,
        input_schema={
            "type": "object",
            "properties": {"shape": {"type": "string"}},
            "required": ["shape"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        description="规划笛卡尔轨迹",
    ))
    catalog.register(ToolDescriptorV2(
        tool_id="ur5e.move_joints",
        source="mcp:ur5e-sim",
        execution_class=ExecutionClass.PHYSICAL_ACTION,
        side_effect_class=ToolSideEffectClass.REVERSIBLE,
        effect_domain="SIMULATION_STATE_ONLY",
        model_callable=False,
        requires_exact_action_grant=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    ))
    catalog.register(ToolDescriptorV2(
        tool_id="sim_reach",
        source="native:agentd",
        execution_class=ExecutionClass.COMPUTE,
        evidence_class=ToolEvidenceClass.SIMULATED,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    ))
    return catalog


class TestSnapshotBuild:
    def test_active_excluded_split_and_fields(self) -> None:
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot

        snap = build_capability_snapshot(
            _catalog(), body_id="sim/ur5e", mode="SIMULATION"
        )
        assert snap.generation >= 1
        assert snap.digest.startswith("sha256:")
        assert snap.body_id == "sim/ur5e"
        assert snap.mode == "SIMULATION"
        active = {a.capability_id: a for a in snap.active}
        # COMPUTE+DERIVED → PURE_COMPUTE direct；COMPUTE+SIMULATED →
        # SIMULATED_EFFECT direct
        assert "ur5e.plan_cartesian_path" in active
        assert "sim_reach" in active
        assert active["sim_reach"].effect_class == "SIMULATED_EFFECT"
        assert active["sim_reach"].exposure == "direct"
        # PHYSICAL → propose_only 投影（propose_ 前缀），不直接暴露
        phys = active.get("ur5e.move_joints")
        assert phys is not None
        assert phys.exposure == "propose_only"
        assert phys.tool_name.startswith("propose_")

    def test_tool_name_slug_and_collision(self) -> None:
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot

        snap = build_capability_snapshot(
            _catalog(), body_id="sim/ur5e", mode="SIMULATION"
        )
        names = {a.capability_id: a.tool_name for a in snap.active}
        # "." → "__"（沿用 wire 约定，OpenAI 函数名合法）
        assert names["ur5e.plan_cartesian_path"] == "ur5e__plan_cartesian_path"
        assert names["sim_reach"] == "sim_reach"
        # 全部合法且唯一
        import re

        for name in names.values():
            assert re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name), name
        assert len(set(names.values())) == len(names)

    def test_digest_stable_and_changes_with_registry(self) -> None:
        """digest 对同内容稳定；registry 变化（注册/隔离）→ digest 变。"""
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot

        s1 = build_capability_snapshot(
            _catalog(), body_id="sim/ur5e", mode="SIMULATION"
        )
        s2 = build_capability_snapshot(
            _catalog(), body_id="sim/ur5e", mode="SIMULATION"
        )
        assert s1.digest == s2.digest
        catalog = _catalog()
        catalog.quarantine_tool("sim_reach", "health failed")
        s3 = build_capability_snapshot(
            catalog, body_id="sim/ur5e", mode="SIMULATION"
        )
        assert s3.digest != s1.digest
        # 隔离的进 excluded 并附原因
        excluded = {e.capability_id: e for e in s3.excluded}
        assert "sim_reach" in excluded
        assert excluded["sim_reach"].reason

    def test_unexposable_effect_excluded_fail_closed(self) -> None:
        """未知/不可曝光 effect 不进工具面（fail closed + 原因码）。"""
        from rosclaw.agentd.tooling.capability_adapter import (
            capability_from_tool_descriptor,
        )
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot
        from rosclaw.contracts.agent.capability import EffectClassV1

        catalog = _catalog()
        cap = capability_from_tool_descriptor(
            ToolDescriptorV2(
                tool_id="weird", source="mcp:weird",
                execution_class=ExecutionClass.OBSERVE,
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
        )
        cap.effect.class_ = EffectClassV1.HOST_PROCESS  # 能力面不曝光
        catalog.register_capability(cap)
        snap = build_capability_snapshot(
            catalog, body_id="sim/ur5e", mode="SIMULATION"
        )
        assert "weird" not in {a.capability_id for a in snap.active}
        excluded = {e.capability_id: e for e in snap.excluded}
        assert excluded["weird"].reason == "EFFECT_NOT_EXPOSABLE"


class TestSnapshotBridgeAndEnforcement:
    async def test_bridge_snapshot_endpoint(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.capability.snapshot",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        snap = result["snapshot"]
        assert snap["schema_version"] == "rosclaw.capability_snapshot.v1"
        assert snap["digest"].startswith("sha256:")
        assert snap["mode"] == "SIMULATION"
        assert snap["generation"] >= 1
        # 内置 sim 能力在 active 且带精确 schema
        ids = {a["capability_id"] for a in snap["active"]}
        assert "sim_reach" in ids or "sim_get_state" in ids
        await service.close()

    async def test_stale_snapshot_digest_rejected(self, tmp_path: Path) -> None:
        """回合执行中 registry 变了：携带旧 digest 的调用 →
        CAPABILITY_SNAPSHOT_CHANGED（不静默换工具）。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        request = _request(
            "rosclaw_compute",
            mission=mission.mission_id, idem="n5d_1",
            lease=await _issue_lease(service, mission),
            arguments={
                "capability_id": "trajectory_generate_planar_path",
                "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                              "scale_m": 0.10},
            },
        )
        request.arguments["snapshot_digest"] = "sha256:stale"
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000, request=request,
        )
        assert result.ok is False
        assert result.error_code == "CAPABILITY_SNAPSHOT_CHANGED", result
        await service.close()

    async def test_current_snapshot_digest_accepted(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        snap_result = await bridge._dispatch(
            "user:local:1000", 1, "pi.capability.snapshot",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        digest = snap_result["snapshot"]["digest"]
        request = _request(
            "rosclaw_compute",
            mission=mission.mission_id, idem="n5d_2",
            lease=await _issue_lease(service, mission),
            arguments={
                "capability_id": "trajectory_generate_planar_path",
                "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                              "scale_m": 0.10},
            },
        )
        request.arguments["snapshot_digest"] = digest
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000, request=request,
        )
        assert result.ok, result
        await service.close()
