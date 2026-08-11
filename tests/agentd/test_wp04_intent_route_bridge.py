"""WP-P0-5 红测试（总纲 §7.1）：pi.intent.route 桥方法 + 输入路由。"""

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


class TestIntentRouteBridge:
    async def test_route_star_text(self, tmp_path: Path) -> None:
        service = await _service(tmp_path)
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.intent.route",
            {"token": service.control_token, "text": "我想用机械臂画个五角星"},
        )
        assert result.get("ok")
        assert result["spec"]["goal"] == "draw_shape"
        miss = await bridge._dispatch(
            "user:local:1000", 1, "pi.intent.route",
            {"token": service.control_token, "text": "分析这段日志"},
        )
        assert miss.get("ok") and miss.get("spec") is None
        await service.close()
