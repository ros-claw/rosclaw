"""PR-SIX-5 红测试（六审 §7）：Operator 默认 SIM 闭环。

红测试先行——当前 main journey 在 chat 前手工 enroll + 启动
operatord，真实用户直接 chat 只会看到 Operator OFFLINE 且无从下手：

1. pi.operator.status 报告 enrollment/running 真实状态；
2. pi.operator.bootstrap（SIM developer）：单键初始化——enroll +
   启动独立 operatord 进程， approvals.list 可用；
3. REAL/SHADOW 模式 bootstrap 一律拒绝（不自动 enrollment、不用
   --no-human-presence-check）；
4. 主 journey 不得再预启动 operatord（黑盒验收：clean install →
   chat → TUI 内单键初始化 → 完成 SIM approval）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.agentd.test_pi_approval import _setup_with_operatord  # noqa: F401  (参照)
from tests.agentd.test_pi_tool_bridge import _setup as _base_setup  # noqa: F401
from tests.agentd.test_pi_tool_bridge import _turn


async def _setup_no_operator(tmp_path: Path):
    """只起 agentd 侧（无 operatord、无 enrollment）。"""
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService

    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )
    # 生产路径里 operator.sock 投影由 app lifespan 启动——单测无 app，
    # 显式起（与 _setup_with_operatord 同构）。
    from rosclaw.agentd.operator_socket import OperatorSocketServer

    agent_server = OperatorSocketServer(service, tmp_path / "run" / "operator.sock")
    await agent_server.start()
    service._operator_projection_for_test = agent_server  # 防 GC
    mission = service.create_mission("six-5 operator bootstrap")
    return service, mission


class TestOperatorBootstrap:
    async def test_status_reports_enrollment_state(self, tmp_path: Path) -> None:
        """未 enrollment 时 pi.operator.status 必须如实报告。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission = await _setup_no_operator(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.operator.status",
            {"token": service.control_token},
        )
        assert result.get("ok"), result
        assert result.get("enrolled") is False
        assert result.get("running") is False
        await service.close()

    async def test_bootstrap_enrolls_and_starts_operatord(
        self, tmp_path: Path
    ) -> None:
        """SIM developer：bootstrap 单键初始化——enroll + 独立进程启动，
        operatord.sock 可用（真实 approvals.list 通过）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission = await _setup_no_operator(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.operator.bootstrap",
            {"token": service.control_token},
        )
        assert result.get("ok"), f"bootstrap 失败: {result}"
        # operatord.sock 在合理时间内出现且 approvals.list 可用。
        from rosclaw.agentd.operator_socket import operator_call

        sock = tmp_path / "run" / "operatord.sock"
        deadline = asyncio.get_event_loop().time() + 20
        while asyncio.get_event_loop().time() < deadline and not sock.exists():
            await asyncio.sleep(0.2)
        assert sock.exists(), "bootstrap 后 operatord.sock 未出现"
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        assert listed.get("ok"), listed
        status = await bridge._dispatch(
            "user:local:1000", 1, "pi.operator.status",
            {"token": service.control_token},
        )
        assert status.get("enrolled") is True
        assert status.get("running") is True
        await service.close()

    async def test_bootstrap_refused_for_real_mode(self, tmp_path: Path) -> None:
        """REAL/SHADOW 模式不得自动 enrollment/启动（人工在场策略）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission = await _setup_no_operator(tmp_path)
        # SIM 环境无法创建 SHADOW/REAL mission（create_mission 自身
        # fail closed——这也是证据）。用 stub 验证 bootstrap 的模式拒绝分支。
        from types import SimpleNamespace

        shadow_mission = SimpleNamespace(
            mission_id="mis_shadow", mode=SimpleNamespace(value="SHADOW")
        )
        original_get = service.get_mission
        service.get_mission = lambda mid: (  # type: ignore[method-assign]
            shadow_mission if mid == "mis_shadow" else original_get(mid)
        )
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.operator.bootstrap",
            {"token": service.control_token,
             "mission_id": "mis_shadow"},
        )
        assert not result.get("ok"), "SHADOW 模式竟允许 bootstrap"
        assert result.get("code") in ("MODE_FORBIDDEN", "OPERATOR_POLICY")
        # 且未产生 enrollment。
        assert not (tmp_path / "operatord" / "operator-identity.json").exists()
        await service.close()


class TestJourneyNoPrestartedOperator:
    """黑盒约束：主 journey 不得再预启动 operatord。"""

    def test_journey_does_not_prestart_operatord(self) -> None:
        """test_full_journey_pty 不得包含手工 enroll/start operatord——
        否则默认体验缺口永远测不到。"""
        source = (
            Path(__file__).resolve().parent / "test_product_journey.py"
        ).read_text(encoding="utf-8")
        # 旅程体（非其他测试文件）里不得有 operatord enroll/start 夹具。
        import re

        body = re.search(
            r"def test_full_journey_pty.*?def _run_journey", source, re.DOTALL
        )
        assert body, "找不到 test_full_journey_pty"
        assert "operatord\", \"enroll" not in body.group(0), (
            "旅程仍手工 enroll operatord——默认体验缺口被夹具掩盖"
        )
        assert "operatord\", \"start" not in body.group(0), (
            "旅程仍手工启动 operatord——默认体验缺口被夹具掩盖"
        )
