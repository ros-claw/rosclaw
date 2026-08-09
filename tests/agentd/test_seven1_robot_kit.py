"""PR-SEVEN-1 红测试（七审 §2.3/§6）：Clean-install Robot Kit 闭环。

红测试先行——当前 main 的缺陷：

1. AgentService 只从用户 config.yaml 的 mcp_servers 装配执行能力——
   默认安装的 body 是 sim/ur5e，但 UR5e 能力包不会自动激活
   （用户终端实测 action capabilities 为 0）；
2. Product Journey 手工写 mcp_servers 并引用仓库源码路径——
   证明的是"注入正确配置后能 reach"，不是"发布包开箱即用"。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestFirstPartyKitActivation:
    async def test_clean_config_auto_activates_ur5e_kit(self, tmp_path: Path) -> None:
        """空配置（无 mcp_servers）+ sim/ur5e body → 第一方 UR5e kit
        自动激活：目录含 3 个动作能力 + executor 存在。"""
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService
        from tests.agentd.test_pi_tool_bridge import _turn

        config = load_agent_config(tmp_path / "config.yaml")
        assert not config.mcp_servers, "前置条件：零 mcp_servers 配置"
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
        )
        await service._ensure_mcp_discovered()
        catalog = service._tool_catalog
        for tool_id in ("ur5e.move_joints", "ur5e.move_to_pose", "ur5e.stop"):
            descriptor = catalog.get(tool_id)
            assert descriptor is not None, (
                f"clean install 缺 {tool_id}——第一方 kit 未自动激活"
            )
        # executor 必须按 source 路由存在（不是只有目录条目）。
        assert service._sim_executors.get("mcp:ur5e-sim") is not None, (
            "kit 激活但 executor 未装配"
        )
        await service.close()

    async def test_kit_activation_reports_health(self, tmp_path: Path) -> None:
        """kit 状态可查询：READY 含 health probes；未激活/损坏诚实报
        BROKEN——不能再出现 body 有 identity 但动作数为 0 的假就绪。"""
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService
        from tests.agentd.test_pi_tool_bridge import _turn

        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
        )
        status = await service.robot_kit_status()
        assert status["kit_id"] == "rosclaw/ur5e-sim"
        assert status["state"] in ("READY", "BROKEN")
        assert status.get("action_capability_count", 0) >= 3, (
            f"kit READY 但动作能力数为 {status.get('action_capability_count')}"
        )
        assert status.get("executor") == "READY"
        await service.close()

    def test_kit_manifest_shipped_in_package(self) -> None:
        """RobotKitManifestV1 必须随包发布（不是测试夹具）。"""
        import rosclaw

        pkg = Path(rosclaw.__file__).parent
        candidates = list(pkg.glob("sim/kits/*.json")) + list(pkg.glob("sim/kits/*.yaml"))
        assert candidates, "包内无 robot kit manifest（sim/kits/）"
        content = candidates[0].read_text(encoding="utf-8")
        assert "rosclaw/ur5e-sim" in content
        assert "executor" in content


class TestJourneyFixtureBan:
    """七审 §6 PR-SEVEN-1.8/§7 CI 结构门禁。"""

    def test_journey_has_no_handwritten_mcp_servers(self) -> None:
        source = (REPO / "tests" / "agentd" / "test_product_journey.py").read_text(
            encoding="utf-8"
        )
        assert "mcp_servers:" not in source, (
            "Product Journey 仍手写 mcp_servers——证明的是注入配置不是开箱即用"
        )

    def test_journey_has_no_repo_source_paths(self) -> None:
        source = (REPO / "tests" / "agentd" / "test_product_journey.py").read_text(
            encoding="utf-8"
        )
        assert "REPO / 'src'" not in source and 'REPO / "src"' not in source, (
            "Product Journey 仍引用仓库源码路径作为 executor"
        )
