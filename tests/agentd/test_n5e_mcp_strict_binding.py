"""PR-N5E 红测试（调整方案 §三.N5E）：MCP 严格绑定。

红测试先行——严格分类/QUARANTINED_UNCLASSIFIED 不存在时必须红。

1. 生产模式：MCP 工具必须经显式声明分类（安装配置的
   observation/compute/action 列表 + output_schema + effect domain）；
   名称/注解启发式不再上线执行；
2. 含糊工具 → QUARANTINED_UNCLASSIFIED（注册可见、不可执行、
   进 snapshot excluded）；
3. 启发式只在 doctor/inspect 给建议（suggest_classification 纯建议，
   不影响上线分类）；
4. 第一方 kit（显式声明）不受影响；
5. discovery / capabilities / snapshot 共享同一 digest 事实源。
"""

from __future__ import annotations

from pathlib import Path


class _FakeTool:
    def __init__(self, name: str, *, read_only: bool = False) -> None:
        self.name = name
        self.description = f"tool {name}"
        self.inputSchema = {"type": "object"}
        self.annotations = (
            type("A", (), {"readOnlyHint": read_only, "destructiveHint": False})()
        )


class _FakeClient:
    """持久 client 形态的假 MCP server（跳过 stdio 进程）。"""

    def __init__(self, tool_names: list[str], *, read_only: bool = False) -> None:
        self._tools = [_FakeTool(n, read_only=read_only) for n in tool_names]

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name: str, arguments: dict):
        return {"ok": True, "echo": name}


def _adapter(tool_names, *, read_only=False, **cfg_kwargs):
    from rosclaw.agentd.tooling.catalog import ToolCatalog
    from rosclaw.agentd.tooling.mcp_adapter import (
        McpCapabilityAdapter,
        McpServerConfig,
    )

    catalog = ToolCatalog()
    cfg = McpServerConfig(name="ext", command="true", **cfg_kwargs)
    return McpCapabilityAdapter(
        cfg, catalog, client=_FakeClient(tool_names, read_only=read_only)
    ), catalog


class TestStrictBinding:
    async def test_explicitly_declared_tools_classified(self) -> None:
        """显式声明的工具按列表分类且不隔离。"""
        adapter, catalog = _adapter(
            ["ext.get_pose", "ext.plan_path", "ext.move"],
            observation_tools=("ext.get_pose",),
            compute_tools=("ext.plan_path",),
            action_tools=("ext.move",),
            output_schemas={"ext.plan_path": {
                "type": "object", "properties": {"ok": {"type": "boolean"}},
            }},
        )
        report = await adapter.discover()
        assert report.ok
        from rosclaw.contracts.agent.tool import ExecutionClass

        assert catalog.get("ext.get_pose").execution_class is ExecutionClass.OBSERVE
        assert catalog.get("ext.plan_path").execution_class is ExecutionClass.COMPUTE
        assert catalog.get("ext.move").execution_class is ExecutionClass.PHYSICAL_ACTION
        for tid in ("ext.get_pose", "ext.plan_path", "ext.move"):
            assert catalog.quarantine_reason(tid) is None

    async def test_undeclared_tool_quarantined_unclassified(self) -> None:
        """未显式声明的工具 → QUARANTINED_UNCLASSIFIED：注册可见、
        不可执行、原因诚实。"""
        adapter, catalog = _adapter(["ext.mystery"], read_only=True)
        report = await adapter.discover()
        assert report.ok
        reason = catalog.quarantine_reason("ext.mystery")
        assert reason is not None
        assert "QUARANTINED_UNCLASSIFIED" in reason
        # 不可执行（execute_v2 BLOCKED）。
        env = await catalog.execute_v2("c1", "ext.mystery", {})
        assert env.status.value == "BLOCKED"
        assert env.error.code == "CAPABILITY_QUARANTINED"

    async def test_name_verb_heuristic_no_longer_classifies(self) -> None:
        """名称动词启发式不再上线：叫 move_fast 的未声明工具此前会被
        分成 PHYSICAL_ACTION 上线——现在必须 QUARANTINED_UNCLASSIFIED。"""
        adapter, catalog = _adapter(["ext.move_fast"])
        await adapter.discover()
        reason = catalog.quarantine_reason("ext.move_fast")
        assert reason is not None and "QUARANTINED_UNCLASSIFIED" in reason

    async def test_annotations_alone_do_not_classify(self) -> None:
        """第三方自声明注解（readOnlyHint）不是绑定依据——未显式声明
        的 readOnly 工具同样隔离。"""
        adapter, catalog = _adapter(["ext.read_thing"], read_only=True)
        await adapter.discover()
        assert catalog.quarantine_reason("ext.read_thing") is not None


class TestHeuristicIsDoctorOnly:
    def test_suggest_classification_advisory_only(self) -> None:
        """启发式仅存于建议函数（developer doctor 用），不参与上线。"""
        from rosclaw.agentd.tooling.mcp_adapter import suggest_classification

        assert suggest_classification("ext.move_fast", None) == "PHYSICAL_ACTION"
        assert (
            suggest_classification(
                "ext.get_pose",
                type("A", (), {"readOnlyHint": True, "destructiveHint": False})(),
            )
            == "OBSERVE"
        )
        assert suggest_classification("ext.mystery", None) == ""

    async def test_suggestion_carried_in_quarantine_reason(self) -> None:
        """隔离原因里带启发式建议（doctor/inspect 展示），但分类本身
        不据此上线。"""
        adapter, catalog = _adapter(["ext.move_fast"])
        await adapter.discover()
        reason = catalog.quarantine_reason("ext.move_fast") or ""
        assert "PHYSICAL_ACTION" in reason  # 建议在案
        assert "QUARANTINED_UNCLASSIFIED" in reason


class TestSharedSnapshotFacts:
    async def test_quarantined_unclassified_in_snapshot_excluded(
        self, tmp_path: Path
    ) -> None:
        """discovery→catalog→snapshot 同一事实源：隔离工具进 excluded
        且 digest 覆盖它。"""
        from rosclaw.agentd.tooling.snapshot import build_capability_snapshot

        adapter, catalog = _adapter(
            ["ext.get_pose", "ext.mystery"],
            observation_tools=("ext.get_pose",),
        )
        await adapter.discover()
        snap = build_capability_snapshot(
            catalog, body_id="sim/ur5e", mode="SIMULATION"
        )
        excluded = {e.capability_id: e for e in snap.excluded}
        assert "ext.mystery" in excluded
        assert excluded["ext.mystery"].reason == "QUARANTINED_UNCLASSIFIED"
        assert "ext.get_pose" in {a.capability_id for a in snap.active}

    async def test_first_party_kit_unaffected(self, tmp_path: Path) -> None:
        """第一方 kit（显式列表+schema 声明）全部正常分类不隔离——
        产品接线实证。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        for d in service._tool_catalog.list(source="mcp:ur5e-sim"):
            if d.tool_id == "ur5e.execute_cartesian_path":
                # deprecated dev 整轨迹层按设计隔离（八审 P0-3 模型不得
                # 搬运载荷 + N5E 未声明不上线——两契约同时成立）。
                reason = service._tool_catalog.quarantine_reason(d.tool_id)
                assert reason is not None and "QUARANTINED_UNCLASSIFIED" in reason
                continue
            assert service._tool_catalog.quarantine_reason(d.tool_id) is None, (
                f"第一方 kit 工具被误隔离: {d.tool_id}"
            )
        await service.close()


class TestCapabilitiesSnapshotConsistency:
    async def test_capabilities_and_snapshot_share_excluded(
        self, tmp_path: Path
    ) -> None:
        """pi.capabilities 与 pi.capability.snapshot 的 excluded 一致
        （同一 digest 事实源）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        service._tool_catalog.quarantine_tool(
            "ur5e.plan_cartesian_path", "QUARANTINED_UNCLASSIFIED: 未绑定"
        )
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        caps = await bridge._dispatch(
            "user:local:1000", 1, "pi.capabilities",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        snap = await bridge._dispatch(
            "user:local:1000", 1, "pi.capability.snapshot",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        cap_excluded = {
            e["capability_id"] for e in caps.get("excluded") or []
        }
        snap_excluded = {
            e["capability_id"] for e in snap["snapshot"]["excluded"]
        }
        assert "ur5e.plan_cartesian_path" in cap_excluded
        assert "ur5e.plan_cartesian_path" in snap_excluded
        await service.close()
