"""PR-EIGHT-8（H9 重接）：Embodied Checkpoint + 命令面（TaskKernel 权威）。

1. pi.context.checkpoint——从权威存储（missions.db/TaskKernel）重建
   结构化具身检查点：goal/body/mode/非终态 task/pending approval/
   最新 receipt 引用/安全策略。LLM compaction 摘要永远不是安全状
   态权威；
2. pi.task.list——当前 mission 的任务清单（kernel）；
3. pi.task.trace——任务全审计链（task/revision/事件/验证/产物）。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup

SESSION = "pi_1"


async def _run_task(service, mission, *, idem: str):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    return await PiToolDispatcher(service).execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_task",
            mission=mission.mission_id,
            idem=idem,
            lease=await _issue_lease(service, mission),
            arguments={
                "goal": "draw_shape",
                "parameters": {"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 0.10},
            },
        ),
    )


def _bind_kernel_task(service, mission, *, goal: str = "画五角星") -> str:
    """输入事务绑定 kernel 任务（与 InputController 同一入口）。"""
    bound = service._task_kernel.bind_message(
        mission_id=mission.mission_id,
        session_ref=SESSION,
        backend_native_id=SESSION,
        message_id="msg_ckpt_1",
        text=goal,
        cwd=str(service._home),
    )
    return str(bound["task_id"])


class TestContextCheckpoint:
    async def test_checkpoint_from_authoritative_store(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        task_id = _bind_kernel_task(service, mission)
        result = await _run_task(service, mission, idem="idem_ckpt_1")
        assert result.ok
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        checkpoint = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.context.checkpoint",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert checkpoint.get("ok"), checkpoint
        cp = checkpoint.get("checkpoint") or {}
        assert cp.get("schema_version") == "rosclaw.embodied_checkpoint.v1"
        assert cp.get("mission_id") == mission.mission_id
        assert cp.get("mode") == "SIMULATION"
        assert cp.get("body_id") == "sim/ur5e"
        # 活跃 kernel 任务在非终态列表。
        nonterminal = cp.get("nonterminal_tasks") or []
        assert any(t.get("task_id") == task_id for t in nonterminal), cp
        assert cp.get("pending_approvals") == []
        assert cp.get("sim_policy") in ("auto", "ask")
        await service.close()

    async def test_checkpoint_after_kernel_finish(self, tmp_path: Path) -> None:
        """kernel 终态（Verifier 验收）后：非终态列表为空、recent 含
        SUCCEEDED。"""
        service, mission = await _setup(tmp_path)
        task_id = _bind_kernel_task(service, mission)
        result = await _run_task(service, mission, idem="idem_ckpt_2")
        assert result.ok
        # rosclaw_task 已把 gif/trace 登记进 kernel 产物账本——用
        # Verifier 路径收尾（finish_task 真验收）。
        kernel = service._task_kernel
        artifacts = [
            str(r["artifact_id"])
            for r in service._store.connection.execute(
                "SELECT artifact_id FROM artifacts WHERE task_id = ?", (task_id,)
            ).fetchall()
        ]
        finish = kernel.finish_task(
            task_id=task_id, summary="五角星仿真完成", artifact_ids=artifacts
        )
        assert finish["status"] == "SUCCEEDED", finish
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        checkpoint = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.context.checkpoint",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        cp = checkpoint.get("checkpoint") or {}
        assert cp.get("nonterminal_tasks") == []
        recent = cp.get("recent_tasks") or []
        assert recent and recent[0].get("state") == "SUCCEEDED"
        await service.close()


class TestTaskListAndTrace:
    async def test_task_list(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        task_id = _bind_kernel_task(service, mission)
        await _run_task(service, mission, idem="idem_tl_1")
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.task.list",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        tasks = result.get("tasks") or []
        assert any(t.get("task_id") == task_id for t in tasks), tasks
        await service.close()

    async def test_task_trace(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        task_id = _bind_kernel_task(service, mission)
        result = await _run_task(service, mission, idem="idem_tt_1")
        assert result.ok
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        trace = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.task.trace",
            {"token": service.control_token, "task_id": task_id},
        )
        assert trace.get("ok"), trace
        chain = trace.get("trace") or {}
        assert (chain.get("task") or {}).get("task_id") == task_id
        # 审计链：task.started 与 artifact.created 事件在列；产物带 sha。
        event_types = [e.get("event_type") for e in chain.get("events") or []]
        assert "task.started" in event_types, event_types
        assert "artifact.created" in event_types, event_types
        artifacts = chain.get("artifacts") or []
        assert artifacts and all(a.get("sha256") for a in artifacts)
        await service.close()
