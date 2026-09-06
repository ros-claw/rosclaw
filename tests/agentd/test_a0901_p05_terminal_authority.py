"""0901 体验探讨 P0-5 红测试：具身任务终态权威——手拼低层
capability 不得发布终态。

0901 实证（用户日志）：第二轮模型手拼低层 capability——
simulate 产出 trace artifact 后 turn_end 的 consider→finish_task
直接把任务收成"验收 PASS · 交付 DELIVERED"，渲染和验证还在
后面跑。后续视频/验证 artifact 挂在了一个已终态的任务上。

机制实证（本文件复现）：具身任务（body_id 非空）+ 裸 trace
artifact + 无 PlanGraph 事件 + 无 Operator 链 txn →
consider→finish_task 竟 COMPLETED/SUCCEEDED。

闭环断言：
1. 裸手拼产物不发布终态（task 保持 ACTIVE，无 task.terminal，
   outcome 不 COMPLETED）；
2. 有 PlanGraph 执行证据（plan.node 事件）→ 正常完成（自动路由
   链回归护栏）；
3. 无 body 的任务（hello.txt 类编码任务）不受影响（h4 语义不变）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _register_trace(kernel, task_id: str, tmp_path: Path) -> None:
    trace_dir = tmp_path / "sim" / "traces" / "trace_hand"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "trace.json").write_text('{"ok": true}', encoding="utf-8")
    kernel.register_artifact(
        task_id=task_id, path=str(trace_dir / "trace.json"),
        media_type="application/json",
        producer="kernel:capability:ur5e.simulate",
    )


class TestTerminalAuthority:
    # 大道至简 R0-2b：plan.node 证据的回归护栏随 PlanExecutor
    # （recipe 引擎）一起删除——受信执行证据的活形态是
    # kernel:capability 仿真 receipt（test_ddzj_r02b_recipe_purge.py
    # 与 test_wp03_resume_report.py 覆盖）。

    def test_bare_hand_chain_must_not_complete_embodied_task(
        self, tmp_path: Path
    ) -> None:
        """裸手拼（无 plan.node、无 Operator txn）→ 不得终态。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from tests.agentd.test_pi_tool_bridge import _kernel
        from tests.agentd.test_r02_task_spec_deliverables import _draw_task

        kernel, _conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "让机械臂动一下")
        kernel.note_tool_use(task_id, "rosclaw_compute")
        _register_trace(kernel, task_id, tmp_path)
        outcome = TaskCoordinator(kernel).consider(task_id)
        task = kernel.get_task(task_id)
        assert task["state"] not in ("SUCCEEDED",), (
            f"手拼裸产物竟发布终态：{task['state']}"
        )
        if outcome is not None:
            assert outcome["lifecycle"] != "COMPLETED", outcome
        terminal = kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND event_type = 'task.terminal'",
            (task_id,),
        ).fetchone()
        assert int(terminal["n"]) == 0, "裸手拼竟产生 task.terminal"


    def test_non_body_task_unaffected(self, tmp_path: Path) -> None:
        """无 body 的编码任务（h4 语义）不受影响——hand write+deliver
        仍由 consider 收尾。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from tests.agentd.test_pi_tool_bridge import _kernel

        kernel, _conn = _kernel(tmp_path)
        kernel.persist_input(
            mission_id="m1", session_ref="s1",
            message_id="msg_1", text="写一个 hello.txt 并交付",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), body_id="",  # 无 body——编码任务
        )
        task_id = str(bound["task_id"])
        hello = tmp_path / "hello.txt"
        hello.write_text("hello\n", encoding="utf-8")
        kernel.register_artifact(
            task_id=task_id, path=str(hello),
            media_type="text/plain", producer="model:rosclaw_deliver",
        )
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        assert outcome["lifecycle"] == "COMPLETED", outcome


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
