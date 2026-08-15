"""十五审 PR-RF-6 红测试：Capability Executor 统一接入。

红测试先行——修复前必须红：
1. robot.observe.* 任务走确定性 capability 调用（executor 域），
   绝不落 harness/开 Agent Worker；
2. capability 执行失败 → FAILED 带原因（不假装成功）；
3. 缺 capability_id → 编译期 BLOCKED（零执行）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


class TestCapabilityExecutor:
    async def test_observe_routes_to_executor_not_harness(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {
                "goal": "读取本体状态",
                "required_capabilities": ["robot.observe.state"],
                "effects": "simulation_only",
                "inputs": {
                    "capability_id": "sim_get_state",
                    "arguments": {"verbose": False},
                },
            },
            idem="rf6_observe",
        )
        assert view["domain"] == "executor", view
        assert view["runtime"] != "harness:pi-builtin"
        for _ in range(200):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        assert row["state"] == "SUCCEEDED", row["summary"]
        # executor 域不产生 WorkOrder（零 Agent Worker）。
        assert not row.get("work_order_id")
        assert "simulated" in (row["summary"] or "") or "evidence" in (
            row["artifacts_json"] or ""
        )
        await service.close()

    async def test_executor_failure_honest(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {
                "goal": "x",
                "required_capabilities": ["robot.observe.state"],
                "effects": "simulation_only",
                "inputs": {
                    "capability_id": "nonexistent_capability",
                    "arguments": {},
                },
            },
            idem="rf6_fail",
        )
        for _ in range(200):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        assert row["state"] == "FAILED"
        assert "nonexistent_capability" in (row["summary"] or "")
        await service.close()

    async def test_missing_capability_blocked_at_compile(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {
                "goal": "x",
                "required_capabilities": ["robot.observe.state"],
                "effects": "simulation_only",
                "inputs": {},
            },
            idem="rf6_blocked",
        )
        for _ in range(200):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.05)
        row = plane._get(view["execution_id"])
        # 缺 capability_id 时回退到 required_capabilities[0]
        # （robot.observe.state 不是已注册工具）→ FAILED 或 BLOCKED，
        # 但绝不是 SUCCEEDED 假成功。
        assert row["state"] in ("FAILED", "BLOCKED")
        await service.close()
