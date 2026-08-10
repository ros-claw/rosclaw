"""PR-SEVEN-6 红测试（七审 §6）：bootstrap 安全边界。

红测试先行——当前缺陷：

1. pi.operator.bootstrap 空 mission_id 不严格拒绝（可能启动带
   --no-human-presence-check 的 operatord）；
2. 未知 mission_id 不拒绝；
3. --no-human-presence-check 启动的 operatord 对 REAL 模式卡片没有
   scope 限制（_decide_sim_card 无 mode 检查——no-presence 实例会签
   REAL 卡）。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _turn


class TestBootstrapHardening:
    async def _service(self, tmp_path: Path):
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService

        config = load_agent_config(tmp_path / "config.yaml")
        return AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
        )

    async def test_empty_mission_id_refused(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await self._service(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.operator.bootstrap",
            {"token": service.control_token},
        )
        assert not result.get("ok"), "空 mission_id 竟允许 bootstrap"
        assert result.get("code") in ("MISSION_REQUIRED", "MISSION_NOT_FOUND", "MODE_FORBIDDEN")
        assert not (tmp_path / "operatord" / "operator-identity.json").exists()
        await service.close()

    async def test_unknown_mission_refused(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service = await self._service(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.operator.bootstrap",
            {"token": service.control_token, "mission_id": "mis_nonexistent"},
        )
        assert not result.get("ok"), "未知 mission 竟允许 bootstrap"
        assert not (tmp_path / "operatord" / "operator-identity.json").exists()
        await service.close()


class TestNoPresenceSimScope:
    async def test_no_presence_operatord_refuses_real_card(self, tmp_path: Path) -> None:
        """no-presence 实例只有 SIM scope——REAL 模式卡决定必须拒绝。"""
        from rosclaw.operatord.enrollment import enroll
        from rosclaw.operatord.server import OperatorDaemon

        identity = enroll(tmp_path / "operatord")
        daemon = OperatorDaemon(
            identity=identity,
            socket_path=tmp_path / "run" / "operatord.sock",
            agent_socket=None,
            daemon_client=None,
            require_human_presence=False,
        )

        # 直接调 _decide（stub _find_card 返回 REAL 卡）。
        async def _fake_find_card(request_id: str):
            return {
                "request_id": request_id,
                "display_hash": "h",
                "mode": "REAL",
            }

        daemon._find_card = _fake_find_card  # type: ignore[method-assign]
        result = await daemon._decide(
            "user:local:1000",
            {"request_id": "appr_x", "display_hash": "h", "approve": True},
            peer_pid=1,
        )
        assert not result.get("ok"), f"no-presence operatord 竟受理 REAL 卡: {result}"
        assert "SIM" in result.get("error", "") or "REAL" in result.get("error", "")

    async def test_no_presence_operatord_still_signs_sim(self, tmp_path: Path) -> None:
        """对照：no-presence 实例对 SIM 卡仍工作（不被误伤）。"""
        from rosclaw.operatord.enrollment import enroll
        from rosclaw.operatord.server import OperatorDaemon

        identity = enroll(tmp_path / "operatord")
        daemon = OperatorDaemon(
            identity=identity,
            socket_path=tmp_path / "run" / "operatord.sock",
            agent_socket=None,
            daemon_client=None,
            require_human_presence=False,
        )

        async def _fake_find_card(request_id: str):
            return {
                "request_id": request_id,
                "display_hash": "h",
                "mode": "SIMULATION",
            }

        daemon._find_card = _fake_find_card  # type: ignore[method-assign]
        result = await daemon._decide(
            "user:local:1000",
            {"request_id": "appr_x", "display_hash": "h", "approve": True},
            peer_pid=1,
        )
        # SIM 卡走到 agent 投影失败（本测试无 agent socket）——而不是被
        # scope 拒绝。
        assert "scope" not in str(result.get("error", "")).lower()
        assert "SIM-scope" not in str(result.get("error", ""))
