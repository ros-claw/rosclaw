"""PR-EIGHT-8 红测试（八审 §5 P1 Embodied Compaction + 命令面）。

红测试先行：

1. pi.context.checkpoint——从权威存储（missions.db/task_records）
   重建结构化具身检查点：goal/body/mode/非终态 task/pending
   approval/最新 receipt 引用/安全策略。LLM compaction 摘要永远不
   是安全状态权威；
2. pi.task.list——当前 mission 的任务清单（/task）；
3. pi.task.trace——任务全审计链（/trace task_id）：状态迁移 +
   approval/grant/txn/receipt 引用。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


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


class TestContextCheckpoint:
    async def test_checkpoint_from_authoritative_store(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
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
        # 完成的任务在 recent_tasks（非终态列表为空但键存在）。
        assert cp.get("nonterminal_tasks") == []
        recent = cp.get("recent_tasks") or []
        assert recent and recent[0].get("state") == "VERIFIED"
        assert cp.get("pending_approvals") == []
        assert cp.get("sim_policy") in ("auto", "ask")
        await service.close()


class TestTaskListAndTrace:
    async def test_task_list(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
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
        assert len(tasks) == 1
        assert tasks[0]["state"] == "VERIFIED"
        assert tasks[0]["goal"] == "draw_shape"
        await service.close()

    async def test_task_trace_full_chain(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_tr_1")
        import json

        task_id = json.loads(result.summary)["task_id"]
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        trace = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.task.trace",
            {"token": service.control_token, "task_id": task_id},
        )
        assert trace.get("ok"), trace
        tr = trace.get("trace") or {}
        assert tr.get("task", {}).get("state") == "VERIFIED"
        assert tr.get("approval", {}).get("status") == "APPROVED"
        assert tr.get("grant", {}).get("consumed") in (1, True)
        assert tr.get("txn", {}).get("state") == "COMPLETED"
        assert tr.get("receipt"), "缺 receipt 引用"
        await service.close()
