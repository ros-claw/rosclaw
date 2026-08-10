"""PR-EIGHT-5 红测试（八审 §1.6/P0-5）：Task Compiler + Task Runner。

红测试先行——当前模型要完成五角星必须自己拼 capabilities→plan→
execute→trace→verify 工具链（真实会话 16 次提案/逐点降级）。
任务应是一个确定性编译器入口：rosclaw_task(goal=draw_shape) 一次
调用，内核完成规划→策略→单动作执行→自动验证，模型只拿 TaskResult。

验收锚点：恰好 1 次 action proposal；1 个 COMPLETED txn；verifier
PASS 才 VERIFIED；幂等重放不产生第二动作。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


async def _run_task(service, mission, *, idem: str, params: dict | None = None):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    dispatcher = PiToolDispatcher(service)
    return await dispatcher.execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_task",
            mission=mission.mission_id,
            idem=idem,
            lease=await _issue_lease(service, mission),
            arguments={
                "goal": "draw_shape",
                "parameters": params
                or {
                    "shape": "star5",
                    "center_m": [0.35, 0.25, 0.30],
                    "radius_m": 0.10,
                },
            },
        ),
    )


class TestDrawShapeTask:
    async def test_auto_sim_end_to_end_single_action(self, tmp_path: Path) -> None:
        """默认 SIM（POLICY_AUTO）：一次调用 → VERIFIED；恰好 1 次
        提案、1 个 COMPLETED txn、verifier PASS。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_task_1")
        assert result.ok, f"task 失败: {result.summary} {result.error_code}"
        import json

        payload = json.loads(result.summary)
        assert payload["state"] == "VERIFIED", payload
        assert payload["verification"]["verdict"] == "PASS"
        assert payload["policy"] == "AUTO_SIM"
        assert payload["task_id"]
        # 恰好一次提案 + 单 txn（execute_plan）。
        db = service._store.connection
        cards = db.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
        txns = db.execute(
            "SELECT capability_id, state FROM action_txns"
        ).fetchall()
        assert cards == 1, f"任务应恰好一次提案: {cards}"
        assert len(txns) == 1 and txns[0][0] == "ur5e.execute_plan", txns
        assert txns[0][1] == "COMPLETED"
        await service.close()

    async def test_invalid_params_fail_before_any_proposal(
        self, tmp_path: Path
    ) -> None:
        """参数越界（半径超工作区）→ FAILED + 零提案（编译期拦截）。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(
            service, mission, idem="idem_task_2",
            params={"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 5.0},
        )
        assert not result.ok
        db = service._store.connection
        cards = db.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
        assert cards == 0, f"非法参数竟产生提案: {cards}"
        await service.close()

    async def test_idempotent_replay_no_second_action(self, tmp_path: Path) -> None:
        """同一 idempotency_key 重放 → 同一 task，无第二动作。"""
        service, mission = await _setup(tmp_path)
        first = await _run_task(service, mission, idem="idem_task_3")
        assert first.ok
        second = await _run_task(service, mission, idem="idem_task_3")
        assert second.ok
        db = service._store.connection
        txns = db.execute("SELECT COUNT(*) FROM action_txns").fetchone()[0]
        assert txns == 1, f"重放产生第二动作: {txns}"
        await service.close()

    async def test_verifier_unavailable_is_honest_inconclusive(
        self, tmp_path: Path
    ) -> None:
        """verifier 不可用（被隔离）→ 任务诚实 INCONCLUSIVE/FAILED——
        不得 VERIFIED，不得让模型声称完成。"""
        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        service._tool_catalog.quarantine_tool("ur5e.verify_drawing", "test_quarantine")
        result = await _run_task(service, mission, idem="idem_task_4")
        import json

        payload = json.loads(result.summary)
        assert payload["state"] in ("INCONCLUSIVE", "FAILED"), payload
        assert payload["state"] != "VERIFIED"
        await service.close()

    async def test_task_record_persisted_with_phases(self, tmp_path: Path) -> None:
        """任务记录持久化（崩溃可恢复）：终态 + plan/txn 引用齐全。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_task_5")
        assert result.ok
        import json

        payload = json.loads(result.summary)
        row = service._store.connection.execute(
            "SELECT state, plan_id, txn_id, goal FROM task_records WHERE task_id = ?",
            (payload["task_id"],),
        ).fetchone()
        assert row, "task_records 无记录"
        assert row[0] == "VERIFIED"
        assert row[1], "缺 plan_id 引用"
        assert row[2], "缺 txn_id 引用"
        assert row[3] == "draw_shape"
        await service.close()
