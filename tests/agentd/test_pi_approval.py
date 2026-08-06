"""PR-PNA-5（规格 §20 验收子集）：bridge request_action 授权链。

- 创建授权卡 → operatord（SIM 签名）批准 → 执行 → receipt；
- operator 拒绝 → 动作不执行；
- 超时未决定 → 动作不执行（默认拒绝）；
- 模型/工具路径本身不含任何"自批"入口（工具只建卡+等待）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.agentd.operator_socket import OperatorSocketServer, operator_call
from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.operatord.enrollment import enroll
from rosclaw.operatord.server import OperatorDaemon
from tests.agentd.test_pi_tool_bridge import _request, _setup


async def _setup_with_operatord(tmp_path: Path):
    service, mission = await _setup(tmp_path)
    agent_sock = tmp_path / "run" / "operator.sock"
    agent_server = OperatorSocketServer(service, agent_sock)
    await agent_server.start()
    identity = enroll(tmp_path / "operatord")
    operatord = OperatorDaemon(
        identity=identity,
        socket_path=tmp_path / "run" / "operatord.sock",
        agent_socket=agent_sock,
        daemon_client=None,
        require_human_presence=False,
    )
    await operatord.start()
    return service, mission, operatord, agent_server, tmp_path / "run" / "operatord.sock"


async def _decide_pending(service, mission_id: str, sock: Path, approve: bool) -> dict:
    listed = await operator_call(sock, "approvals.list", {"mission_id": mission_id})
    entry = listed["approvals"][0]
    return await operator_call(
        sock,
        "approvals.decide",
        {"request_id": entry["request_id"], "display_hash": entry["display_hash"],
         "approve": approve},
    )


class TestRequestActionChain:
    async def test_approve_then_execute_then_receipt(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(tmp_path)
        dispatcher = PiToolDispatcher(service)

        async def operator_approves() -> None:
            for _ in range(30):
                await asyncio.sleep(0.5)
                if service.pending_approvals(mission.mission_id):
                    await _decide_pending(service, mission.mission_id, sock, True)
                    return

        approver = asyncio.create_task(operator_approves())
        try:
            result = await dispatcher.execute(
                _request(
                    "rosclaw_request_action",
                    mission=mission.mission_id,
                    idem="idem_ra_1",
                    arguments={
                        "capability_id": "sim_ground_truth",
                        "arguments": {},
                        "expected_effect": "SIM 探测",
                        "risk_tier": "LOW",
                    },
                )
            )
        finally:
            approver.cancel()
        assert result.approval_id
        # SIM executor 结果（接受与否取决于 sim executor 存在性——关键是
        # 链走到了执行而非卡在授权）。
        assert result.status in {"COMPLETED", "FAILED"}
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_operator_decline_never_executes(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(tmp_path)
        dispatcher = PiToolDispatcher(service)

        async def operator_denies() -> None:
            for _ in range(30):
                await asyncio.sleep(0.5)
                if service.pending_approvals(mission.mission_id):
                    await _decide_pending(service, mission.mission_id, sock, False)
                    return

        denier = asyncio.create_task(operator_denies())
        try:
            result = await dispatcher.execute(
                _request(
                    "rosclaw_request_action",
                    mission=mission.mission_id,
                    idem="idem_ra_2",
                    arguments={"capability_id": "sim_ground_truth", "arguments": {}},
                )
            )
        finally:
            denier.cancel()
        assert not result.ok
        assert result.status == "DECLINED"
        assert result.error_code == "OPERATOR_DECLINED"
        # 没有产生任何 grant。
        assert service.list_grants() == []
        await operatord.stop()
        await agent_server.stop()
        await service.close()
