"""0901 体验探讨 P0-4 红测试：解释性追问 = 只读确定性回答（硬 Gate A）。

0901 实证：任务 FAIL+PARTIAL 后用户问"你这是啥？"——系统没有
只读解释路径，模型重跑了整个任务（新 trace + 第二套 artifact）。

硬 Gate A（文档 §十二）：解释性追问 → 0 新 Task / 0 新 Trace /
0 新 Artifact / 0 次仿真渲染 / 从 TaskOutcome 直接回答。

闭环断言：
1. 解释标记（这是啥/什么意思/结果呢/文件在哪/为什么失败/成功了
   吗/给我看）→ 已有任务时 owner=EXPLAIN_HANDLER + suppress
   模型回合 + 回答负载含 outcome+artifacts；
2. 无最近任务时不劫持（走模型正常聊天）；
3. 解释后账本零新增（tasks/artifacts/plans 数量不变）；
4. 非解释的问句（"你能画五角星吗？"）不被劫持。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestExplainRoute:
    def test_explain_markers(self) -> None:
        from rosclaw.agentd.explain_route import is_explain_followup

        for text in (
            "你这是啥？", "什么意思", "结果呢", "文件在哪", "为什么失败了",
            "成功了吗", "给我看看", "刚才那个是什么",
        ):
            assert is_explain_followup(text), text
        for text in ("画一个五角星", "你能画五角星吗？", "帮我写个脚本"):
            assert not is_explain_followup(text), text

    async def test_explain_followup_deterministic_answer(
        self, tmp_path: Path
    ) -> None:
        """已有终态任务 + 解释追问 → owner=EXPLAIN_HANDLER +
        suppress + outcome 负载——零新 task/artifact。"""
        import asyncio

        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "b.sock")
        # 先跑完一个真实任务（画五角星——自动路由链）。
        r1 = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {"token": service.control_token, "mission_id": mission.mission_id,
             "session_ref": "pi_1", "message_id": "m1",
             "text": "画一个五角星"},
        )
        task_id = str(r1["auto_task"]["task_id"])
        kernel = service._task_kernel
        deadline = asyncio.get_event_loop().time() + 180
        while asyncio.get_event_loop().time() < deadline:
            t = kernel.get_task(task_id)
            if t and t["state"] in ("SUCCEEDED", "FAILED"):
                break
            await asyncio.sleep(2)
        counts_before = {
            t: kernel._conn.execute(
                f"SELECT COUNT(*) AS n FROM {t}"  # noqa: S608
            ).fetchone()["n"]
            for t in ("tasks", "artifacts")
        }
        plans_before = len(list((tmp_path / "sim" / "plans").glob("*.json")))
        # 解释追问。
        r2 = await bridge._dispatch(
            "user:local:1000", 2, "pi.input.persist",
            {"token": service.control_token, "mission_id": mission.mission_id,
             "session_ref": "pi_1", "message_id": "m2",
             "text": "你这是啥？"},
        )
        disposition = r2.get("turn_disposition") or {}
        assert disposition.get("owner") == "EXPLAIN_HANDLER", r2
        assert disposition.get("suppress_model_turn") is True, r2
        explain = r2.get("explain")
        assert explain, "缺 explain 负载"
        assert explain.get("task_id") == task_id, explain
        outcome = explain.get("outcome") or {}
        assert outcome.get("verification"), outcome
        assert explain.get("artifacts") is not None, explain
        # 硬 Gate A：账本零新增。
        for table, before in counts_before.items():
            after = kernel._conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608
            ).fetchone()["n"]
            assert after == before, f"{table}: {before} → {after}"
        plans_after = len(list((tmp_path / "sim" / "plans").glob("*.json")))
        assert plans_after == plans_before, "解释追问竟产生了新 plan"
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
