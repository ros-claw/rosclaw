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


async def _run_task(service, mission, *, idem: str) -> str:
    """大道至简 R0-2b：通用能力链（plan→simulate→render）驱动——
    capability 产物自动登记；返回产物归属的 task_id（admission
    建的任务——不再先 bind 再跑 recipe 的双段式）。
    """
    import json as _json

    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    lease = await _issue_lease(service, mission)
    dispatcher = PiToolDispatcher(service)
    plan = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem=idem + "_plan", lease=lease,
            arguments={
                "capability_id": "trajectory_generate_planar_path",
                "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                              "scale_m": 0.10, "plane": "xy",
                              "max_segment_m": 0.02},
            },
        ),
    )
    assert plan.ok, plan.summary
    plan_id = _json.loads(plan.summary)["value"]["plan_id"]
    sim = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem=idem + "_sim", lease=lease,
            arguments={
                "capability_id": "ur5e_simulate_cartesian_trajectory",
                "arguments": {"plan_id": plan_id},
            },
        ),
    )
    assert sim.ok, sim.summary
    trace_id = _json.loads(sim.summary)["value"]["trace_id"]
    render = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem=idem + "_render", lease=lease,
            arguments={
                "capability_id": "simulation_render_trace",
                "arguments": {"trace_id": trace_id, "format": "gif"},
            },
        ),
    )
    assert render.ok, render.summary
    row = service._store.connection.execute(
        "SELECT task_id FROM artifacts ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert row, "能力链无产物登记"
    return str(row["task_id"])



def _bind_kernel_task(service, mission, *, goal: str = "画五角星") -> str:
    """输入事务绑定 kernel 任务（与 InputController 同一入口）。"""
    bound = service._task_kernel.bind_message(
        mission_id=mission.mission_id,
        session_ref=SESSION,
        backend_native_id=SESSION,
        message_id="msg_ckpt_1",
        text=goal,
        cwd=str(service._home),
        body_id=mission.body_binding.body_id,
    )
    return str(bound["task_id"])


class TestContextCheckpoint:
    async def test_checkpoint_from_authoritative_store(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        # 大道至简 R0-2b：任务由首个 effectful 调用的 admission 创建
        # （聊天输入不建任务）——先驱动能力链建任务，再取"执行后
        # 未收尾"的 checkpoint（任务在非终态列表）。
        task_id = await _run_task(service, mission, idem="idem_ckpt_1")
        before = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.context.checkpoint",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert before.get("ok"), before
        cp = before.get("checkpoint") or {}
        assert cp.get("schema_version") == "rosclaw.embodied_checkpoint.v1"
        assert cp.get("mission_id") == mission.mission_id
        assert cp.get("mode") == "SIMULATION"
        assert cp.get("body_id") == "sim/ur5e"
        nonterminal = cp.get("nonterminal_tasks") or []
        assert any(t.get("task_id") == task_id for t in nonterminal), cp
        assert cp.get("pending_approvals") == []
        assert cp.get("sim_policy") in ("auto", "ask")
        # kernel 收尾（capability 仿真 receipt = 受信执行证据）后
        # 任务进入终态，不再停留在非终态列表。
        kernel = service._task_kernel
        artifacts = [
            str(r["artifact_id"])
            for r in service._store.connection.execute(
                "SELECT artifact_id FROM artifacts WHERE task_id = ?", (task_id,)
            ).fetchall()
        ]
        finish = kernel.finish_task(
            task_id=task_id, summary="仿真完成", artifact_ids=artifacts,
        )
        assert finish["status"] == "SUCCEEDED", finish
        after = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.context.checkpoint",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        cp_after = after.get("checkpoint") or {}
        assert not any(
            t.get("task_id") == task_id
            for t in (cp_after.get("nonterminal_tasks") or [])
        ), cp_after
        recent = cp_after.get("recent_tasks") or []
        assert any(
            t.get("task_id") == task_id and t.get("state") == "SUCCEEDED"
            for t in recent
        ), cp_after
        await service.close()

    async def test_checkpoint_after_kernel_finish(self, tmp_path: Path) -> None:
        """kernel 终态（recipe 内 Verifier 验收）后：非终态列表为空、
        recent 含 SUCCEEDED；finish_task 重放幂等（不覆盖原 receipt）。"""
        service, mission = await _setup(tmp_path)
        task_id = await _run_task(service, mission, idem="idem_ckpt_2")
        # 大道至简 R0-2b：能力链不自动收尾——finish_task 由模型面
        # 调用（rosclaw_task_finish）；重放幂等返回原终态。
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
        replay = kernel.finish_task(
            task_id=task_id, summary="五角星仿真完成", artifact_ids=artifacts
        )
        assert replay.get("already_terminal") is True, replay
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
        task_id = await _run_task(service, mission, idem="idem_tl_1")
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
        task_id = await _run_task(service, mission, idem="idem_tt_1")
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
