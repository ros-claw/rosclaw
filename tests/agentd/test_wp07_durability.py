"""WP-P0-7 红测试（总纲 §7.4/WP-P0-7）：持久 PlanStore + 取消传播 +
崩溃 attach。

红测试先行：

1. PlanStore 进程内存——executor 重启即丢（crash 后 execute_plan
   只能猜）；必须落盘：重启后 plan 可见、已消费状态不复活；
2. /cancel 目前只改 task 状态——待批准卡仍可被批准并执行（取消
   不传播）；取消必须撤销未消费审批；
3. crash 后同一任务再提交（新 idempotency_key）会二次执行——
   恢复运行中任务只能 attach，不能重新下发。
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


class TestPersistentPlanStore:
    def test_plans_survive_store_reload(self, tmp_path: Path) -> None:
        """PlanStore 落盘：新实例（模拟进程重启）仍能看到 plan；
        已消费状态不复活。"""
        from rosclaw.sim.plan_store import PersistentPlanStore

        store = PersistentPlanStore(tmp_path / "plans")
        record = store.put(
            {"hash": "ab" * 32, "points": [{"x": 1, "y": 2, "z": 3}]},
            "test plan",
        )
        # 模拟进程重启：新实例同目录。
        reloaded = PersistentPlanStore(tmp_path / "plans")
        again = reloaded.get_for_execute(record["plan_id"])
        assert again["digest"] == record["digest"]
        # 消费后重启不可再用。
        reloaded.consume(record["plan_id"])
        third = PersistentPlanStore(tmp_path / "plans")
        try:
            third.get_for_execute(record["plan_id"])
            raise AssertionError("已消费 plan 在重启后复活")
        except ValueError as exc:
            assert "consumed" in str(exc)


class TestCancelPropagates:
    async def test_cancel_revokes_pending_approval(self, tmp_path: Path) -> None:
        """取消 WAITING_APPROVAL 任务 → 待批准卡被撤销（再决定即拒，
        绝不产生 grant/执行）。"""
        service, mission = await _setup(tmp_path)
        (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agent" / "safety.json").write_text(
            json.dumps({"sim_policy": "ask"}), encoding="utf-8"
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_task", mission=mission.mission_id, idem="idem_cp_1",
                lease=await _issue_lease(service, mission),
                arguments={
                    "goal": "draw_shape",
                    "parameters": {"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 0.10},
                },
            ),
        )
        payload = json.loads(result.summary)
        assert payload["state"] == "WAITING_APPROVAL"
        approval_id = payload["approval_id"]
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        cancel = await bridge._dispatch(
            "user:local:1000", 1, "pi.task.cancel",
            {"token": service.control_token, "task_id": payload["task_id"]},
        )
        assert cancel.get("ok")
        # 取消后再批准必须失败（不产 grant）。
        from rosclaw.contracts.common import ValidationError

        try:
            grant = service._broker.decide(
                approval_id, principal="user:local:1000", approve=True
            )
            assert grant is None, "已取消的卡竟产出 grant"
        except ValidationError as exc:
            assert "CANCELLED" in str(exc) or "already" in str(exc)
        row = service._store.connection.execute(
            "SELECT status FROM operator_requests WHERE request_id = ?",
            (approval_id,),
        ).fetchone()
        assert row[0] == "CANCELLED", row
        await service.close()


class TestCrashAttachOnly:
    async def test_resubmit_same_task_attaches(self, tmp_path: Path) -> None:
        """同一 mission+goal+params 的非终态任务再提交（新幂等键）
        → 返回既有 task（attach），不产生第二个提案。"""
        service, mission = await _setup(tmp_path)
        (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agent" / "safety.json").write_text(
            json.dumps({"sim_policy": "ask"}), encoding="utf-8"
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        params = {"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 0.10}
        first = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_task", mission=mission.mission_id, idem="idem_at_1",
                lease=await _issue_lease(service, mission),
                arguments={"goal": "draw_shape", "parameters": params},
            ),
        )
        first_payload = json.loads(first.summary)
        assert first_payload["state"] == "WAITING_APPROVAL"
        # 崩溃后重提（新的 idempotency key，同参数）。
        second = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_task", mission=mission.mission_id, idem="idem_at_2",
                lease=await _issue_lease(service, mission),
                arguments={"goal": "draw_shape", "parameters": params},
            ),
        )
        second_payload = json.loads(second.summary)
        assert second_payload["task_id"] == first_payload["task_id"], (
            f"同一任务被重复提交: {first_payload['task_id']} vs {second_payload['task_id']}"
        )
        assert second_payload.get("attached") is True
        cards = service._store.connection.execute(
            "SELECT COUNT(*) FROM operator_requests"
        ).fetchone()[0]
        assert cards == 1, f"attach 竟产生第二提案: {cards}"
        await service.close()
