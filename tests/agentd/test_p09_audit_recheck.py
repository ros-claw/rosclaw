"""0827 复核（对抗自审）：断链不沉默 + 重启恢复 + HOME 导出。

大道至简 R0-1 后：聊天自动路由已删除，本文件直接驱动
TaskExecutionService（执行链纪律不变——失败是数据、重启恢复
重新驱动同一 task/revision；驱动方纪律由 R0-2 demo 驱动面继承）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


class TestBrokenChainNotSilent:
    async def test_failed_chain_transitions_terminal(
        self, tmp_path: Path
    ) -> None:
        """渲染链打坏 → 自动路由执行失败 → 任务必须到 FAILED 终态
        （task.terminal 事件存在——watcher 才能呈现诚实失败）。"""
        from rosclaw.agentd import sim_render
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        # 打坏场景渲染（spec 要求 scene_video → render_scene 节点失败）。
        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_BACKEND_UNAVAILABLE: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        try:
            kernel = service._task_kernel
            text = "画一个五角星，给我仿真视频"
            kernel.persist_input(
                mission_id=mission.mission_id, session_ref="pi_1",
                message_id="msg_broken", text=text,
            )
            bound = kernel.ensure_task_for_effect(
                mission_id=mission.mission_id, session_ref="pi_1",
                backend_native_id="pi_1", cwd="", mode="SIMULATION",
                body_id=str(mission.body_binding.body_id),
                explicit_goal=text,
            )
            task_id = str(bound["task_id"])
            from rosclaw.task_kernel.task_router import compile_recipe_inputs

            outcome = await asyncio.to_thread(
                service._task_execution.execute, task_id,
                recipe_inputs=compile_recipe_inputs(text),
            )
            assert not outcome.ok, "渲染打坏竟执行成功"
            # 断链不沉默纪律（原 auto_route._run，现由驱动面继承）：
            # 执行失败 → FAILED 终态 + task.terminal。
            kernel.transition(
                task_id, "FAILED",
                reason=f"{outcome.error_code or 'EXECUTION_FAILED'}: "
                       f"{(outcome.failure or '')[:200]}",
            )
            task = kernel.get_task(task_id)
            assert task["state"] == "FAILED", (
                f"断链必须到 FAILED 终态（不能 RUNNING 沉默空转）：{task['state']}"
            )
            terminal = kernel._conn.execute(
                "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
                "AND event_type = 'task.terminal'",
                (task_id,),
            ).fetchone()
            assert int(terminal["n"]) == 1, "缺 task.terminal 事件"
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
            await service.close()


class TestResumeInterruptedExecution:
    async def test_startup_redrives_interrupted_chain(
        self, tmp_path: Path
    ) -> None:
        """模拟崩溃现场（RUNNING + plan.node 事件 + 无 task.terminal）
        → 服务启动恢复钩子重新驱动 → 同一 task/revision 到终态。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        kernel = service._task_kernel
        text = "画一个五角星"
        kernel.persist_input(
            mission_id=mission.mission_id, session_ref="pi_1",
            message_id="msg_resume", text=text,
        )
        bound = kernel.ensure_task_for_effect(
            mission_id=mission.mission_id, session_ref="pi_1",
            backend_native_id="pi_1", cwd="", mode="SIMULATION",
            body_id=str(mission.body_binding.body_id),
            explicit_goal=text,
        )
        task_id = str(bound["task_id"])
        from rosclaw.task_kernel.task_router import compile_recipe_inputs

        outcome = await asyncio.to_thread(
            service._task_execution.execute, task_id,
            recipe_inputs=compile_recipe_inputs(text),
        )
        assert outcome.ok, f"{outcome.error_code}: {outcome.failure}"
        # 首轮执行完（SUCCEEDED），然后**伪造崩溃现场**：把任务
        # 状态拨回 RUNNING（等价于 rollout/render 之间进程被杀）。
        assert kernel.get_task(task_id)["state"] == "SUCCEEDED"
        revision_before = int(kernel.get_task(task_id)["active_revision"])
        kernel._conn.execute(
            "UPDATE tasks SET state = 'RUNNING' WHERE task_id = ?",
            (task_id,),
        )
        kernel._conn.commit()
        # 中途崩溃的真实事件形态：最后一条事件是 plan.node_*（不是
        # task.terminal——旧终态事件不算中断现场）。
        kernel._emit(task_id, "plan.node_started",
                     {"node_id": "render", "op": "simulation.render"})
        kernel._conn.commit()
        # 调用启动恢复钩子——必须把中断的链重新驱动到终态（同一
        # task、同一 revision）。
        resumed = await service.resume_interrupted_executions()
        assert task_id in resumed, f"中断任务未被恢复驱动：{resumed}"
        deadline = asyncio.get_event_loop().time() + 180
        while asyncio.get_event_loop().time() < deadline:
            task = kernel.get_task(task_id)
            if task and task["state"] in ("SUCCEEDED", "FAILED"):
                break
            await asyncio.sleep(2)
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", task["state"]
        assert int(task["active_revision"]) == revision_before, (
            f"恢复不得新建 revision：{task['active_revision']} ≠ {revision_before}"
        )
        await service.close()


class TestHomeEnvExported:
    def test_chat_bootstrap_exports_rosclaw_home(self, tmp_path: Path) -> None:
        """agentd 进程必须导出 ROSCLAW_HOME（否则 PlanRef 生产/消费
        分裂或 conformance 误杀工具对——用户不会手工 export）。"""
        import os

        from rosclaw.agentd.cli import _ensure_home_env

        old = os.environ.pop("ROSCLAW_HOME", None)
        try:
            _ensure_home_env(tmp_path)
            assert os.environ.get("ROSCLAW_HOME") == str(tmp_path)
        finally:
            if old is not None:
                os.environ["ROSCLAW_HOME"] = old
            else:
                os.environ.pop("ROSCLAW_HOME", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
