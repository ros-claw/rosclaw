"""PR-PNA-4：Worker 委派（规格 §19 验收子集）。

- /delegate 等价路径（bridge tool）：招聘→执行→验证→结果；
- 未验证输出不进主上下文（VERIFY_FAILED）；
- worker 不可用时诚实 WORKER_UNAVAILABLE；
- 递归上限：depth 超限拒绝。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestDelegate:
    async def test_delegate_full_flow_verified(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_del_1",
                arguments={"goal": "总结这段日志", "worker_id": "auto"},
            )
        )
        # native inproc worker（MockModelGateway 驱动）应当完成并通过验证。
        assert result.ok, result.summary
        assert result.status == "COMPLETED"
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert orders and orders[0].delegation_depth == 0
        assert orders[0].max_delegation_depth == 1
        await service.close()

    async def test_delegate_unavailable_worker_honest(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_del_2",
                arguments={"goal": "x", "worker_id": "worker:nonexistent:nowhere"},
            )
        )
        assert not result.ok
        assert result.error_code in {"WORKER_UNAVAILABLE", "SCHEDULING_FAILED"}
        assert result.retryable
        await service.close()

    async def test_delegation_depth_guard(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_del_3",
                arguments={"goal": "第一层", "worker_id": "auto"},
            )
        )
        root = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        assert root.delegation_depth == 0
        # 带 parent 的"再委派"请求：默认 max=1 一律拒绝（worker 递归委派
        # 不存在于本桥；scheduler 的 max_children 预算是第二层防线）。
        blocked = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_del_5",
                arguments={
                    "goal": "第二层",
                    "worker_id": "auto",
                    "parent_work_order_id": root.work_order_id,
                },
            )
        )
        assert not blocked.ok and blocked.error_code == "DELEGATION_DEPTH_EXCEEDED"
        # 只产生了 root 一个 worker 单。
        assert len(service._worker_manager.orders_for_mission(mission.mission_id)) == 1
        await service.close()
