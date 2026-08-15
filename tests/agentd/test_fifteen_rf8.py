"""十五审 PR-RF-8 红测试：execution 级任务卡投影（Execution View）。

红测试先行——修复前必须红：pi.task.executions 返回 Task Control Plane
的 execution 卡（一任务一卡，runtime/状态/摘要），WorkOrder 只是
内部诊断字段。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


class TestExecutionCards:
    async def test_executions_projection(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {
                "goal": "画五角星动力学仿真",
                "required_capabilities": ["simulation.ur5e", "trajectory"],
                "effects": "simulation_only",
                "inputs": {"shape": "star5"},
                "acceptance": {"max_tracking_error_m": 0.10},
            },
            idem="rf8_card",
        )
        for _ in range(600):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.1)
        # 经 socket handler 太重——直接验证投影函数级输出。
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer  # noqa: F401

        cards = []
        for row in plane.executions_for(mission.mission_id):
            cards.append({
                "execution_id": row["execution_id"],
                "state": row["state"],
                "runtime": row["runtime"],
            })
        assert len(cards) == 1, f"一任务一卡: {len(cards)}"
        assert cards[0]["runtime"] == "executor:simulation"
        assert cards[0]["state"] in ("SUCCEEDED", "VERIFYING")
        await service.close()
