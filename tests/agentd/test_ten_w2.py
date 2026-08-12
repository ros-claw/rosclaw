"""十审 Gate W2 红测试：五工具协议 + 终态投影。

红测试先行——修复前必须红：rosclaw_list_work / rosclaw_update_work
不在工具表（TOOL_UNKNOWN）；pi.worker.status 终态不带 summary/accepted。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestListWork:
    async def test_list_empty_mission(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request("rosclaw_list_work", mission=mission.mission_id, idem="idem_w2_l0")
        )
        assert result.ok
        assert "没有" in result.summary
        await service.close()

    async def test_list_shows_orders_with_exact_ids(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        done = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w2_l1",
                arguments={"goal": "快任务", "worker_id": "auto"},
            )
        )
        assert done.status == "COMPLETED"
        listed = await dispatcher.execute(
            _request("rosclaw_list_work", mission=mission.mission_id, idem="idem_w2_l2")
        )
        assert listed.ok
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        assert order.work_order_id in listed.summary
        assert "ACCEPTED" in listed.summary
        await service.close()


class TestUpdateWork:
    async def test_update_records_steer_note(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w2_u1",
                arguments={"goal": "x", "worker_id": "auto"},
            )
        )
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        # 已 ACCEPTED（快任务）——update 诚实拒绝。
        terminal = await dispatcher.execute(
            _request(
                "rosclaw_update_work",
                mission=mission.mission_id,
                idem="idem_w2_u2",
                arguments={"work_order_id": order.work_order_id, "note": "请加限制"},
            )
        )
        assert not terminal.ok
        assert terminal.error_code == "ALREADY_TERMINAL"
        await service.close()

    async def test_update_unknown_order(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_update_work",
                mission=mission.mission_id,
                idem="idem_w2_u3",
                arguments={"work_order_id": "wo_ghost", "note": "x"},
            )
        )
        assert not result.ok
        assert result.error_code == "WORK_ORDER_NOT_FOUND"
        await service.close()

    async def test_update_running_order_records_note(self, tmp_path: Path) -> None:
        """运行中的单：备注落账（retry/后续 attempt 生效），回复诚实说明
        不能实时转向。"""
        service, mission = await _setup(tmp_path)
        from tests.agentd.test_ten_w0 import _register_stub, _slow_adapter_module

        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w2_u4",
                arguments={"goal": "长任务", "worker_id": "worker:stub:slow"},
            )
        )
        assert started.status == "STARTED"
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        updated = await dispatcher.execute(
            _request(
                "rosclaw_update_work",
                mission=mission.mission_id,
                idem="idem_w2_u5",
                arguments={"work_order_id": order.work_order_id, "note": "只看 src/"},
            )
        )
        assert updated.ok
        assert "不能实时" in updated.summary or "retry" in updated.summary
        current = service._worker_manager.order(order.work_order_id)
        notes = current.inputs.get("steer_notes")
        assert notes and notes[0]["note"] == "只看 src/"
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w2_u6",
                arguments={"work_order_id": order.work_order_id},
            )
        )
        await service.close()
