"""PR-SEVEN-7 红测试（七审 §6 Journey C 单元面）：REAL hard boundary。

红测试先行——当前缺陷：pi.mission.create 把 mode 硬编码为
SIMULATION，调用方请求 REAL 时被静默降级（不是显式拒绝）——
边界应该是显式 MODE_FORBIDDEN，且绝不创建 REAL 工件。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _turn


async def _service(tmp_path: Path):
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService

    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )


class TestRealMissionBoundary:
    async def test_real_mission_create_explicitly_refused(self, tmp_path: Path) -> None:
        """pi.mission.create mode=REAL → 显式拒绝（MODE_FORBIDDEN），
        不是静默降级成 SIMULATION。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await _service(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.mission.create",
            {"token": service.control_token, "goal": "真机抓取", "mode": "REAL"},
        )
        assert not result.get("ok"), f"REAL mission 竟被创建/降级: {result}"
        assert result.get("code") == "MODE_FORBIDDEN"
        await service.close()

    async def test_shadow_mission_create_explicitly_refused(
        self, tmp_path: Path
    ) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await _service(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.mission.create",
            {"token": service.control_token, "goal": "影子模式", "mode": "SHADOW"},
        )
        assert not result.get("ok"), f"SHADOW mission 竟被创建/降级: {result}"
        assert result.get("code") == "MODE_FORBIDDEN"
        await service.close()

    async def test_default_sim_still_works(self, tmp_path: Path) -> None:
        """对照：默认（无 mode 参数）SIM 创建不受影响。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await _service(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.mission.create",
            {"token": service.control_token, "goal": "仿真任务"},
        )
        assert result.get("ok"), result
        assert result.get("mode") == "SIMULATION"
        await service.close()
