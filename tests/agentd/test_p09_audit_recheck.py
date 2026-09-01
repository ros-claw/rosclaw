"""0827 复核（对抗自审）红测试：断链不沉默 + 重启恢复 + HOME 导出。

复核发现的三处真实缺陷（文档 §十 故障注入 Gate 的未实证项）：
1. 断链沉默：确定性链失败（渲染坏掉）→ execute() ok=False，任务
   停在 RUNNING 无 task.terminal——watcher 无呈现点，用户永远
   等不到回复（进度 widget 永久空转）。
2. 重启不恢复：rollout 与 render 之间 agentd 重启 → auto_route
   后台协程死掉，任务停在 RUNNING 无人重新驱动（Gate 明确要求
   "恢复同一 task/revision"）。
3. ROSCLAW_HOME 未在 agentd 进程导出 → PlanRef 生产/消费分裂
   （或 conformance 误杀工具对）——cli chat 必须 setdefault 导出。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


async def _persist(bridge, service, mission_id: str, message_id: str, text: str):
    return await bridge._dispatch(
        "user:local:1000", 1, "pi.input.persist",
        {
            "token": service.control_token,
            "mission_id": mission_id,
            "session_ref": "pi_1",
            "message_id": message_id,
            "text": text,
        },
    )


class TestBrokenChainNotSilent:
    async def test_failed_chain_transitions_terminal(
        self, tmp_path: Path
    ) -> None:
        """渲染链打坏 → 自动路由执行失败 → 任务必须到 FAILED 终态
        （task.terminal 事件存在——watcher 才能呈现诚实失败）。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        # 打坏场景渲染（spec 要求 scene_video → render_scene 节点失败）。
        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_BACKEND_UNAVAILABLE: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        try:
            bridge = PiBridgeServer(
                service, tmp_path / "run" / "pi-bridge.sock"
            )
            result = await _persist(
                bridge, service, mission.mission_id, "msg_broken",
                "画一个五角星，给我仿真视频",
            )
            assert result.get("auto_task"), result
            task_id = str(result["auto_task"]["task_id"])
            kernel = service._task_kernel
            deadline = asyncio.get_event_loop().time() + 180
            while asyncio.get_event_loop().time() < deadline:
                task = kernel.get_task(task_id)
                if task and task["state"] in ("SUCCEEDED", "FAILED", "REPAIR_REQUIRED"):
                    break
                await asyncio.sleep(2)
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
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await _persist(
            bridge, service, mission.mission_id, "msg_resume",
            "画一个五角星",
        )
        assert result.get("auto_task"), result
        task_id = str(result["auto_task"]["task_id"])
        kernel = service._task_kernel
        # 等首轮执行完（SUCCEEDED），然后**伪造崩溃现场**：把任务
        # 状态拨回 RUNNING（等价于 rollout/render 之间进程被杀）。
        deadline = asyncio.get_event_loop().time() + 180
        while asyncio.get_event_loop().time() < deadline:
            task = kernel.get_task(task_id)
            if task and task["state"] == "SUCCEEDED":
                break
            await asyncio.sleep(2)
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
