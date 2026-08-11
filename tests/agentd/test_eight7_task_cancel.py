"""PR-EIGHT-7 红测试（八审 §4 P0-9）：任务取消 + 状态投影。

红测试先行：

1. /cancel 必须能取消真实 task（不只是当前 LLM 回合）——
   WAITING_APPROVAL 的任务取消后 CANCELLED，resume 拒绝；
2. 终态任务取消是诚实 no-op（不改变状态）；
3. kit BROKEN 提示在 reason 为空时不得渲染悬空冒号（审计实测
   "Robot kit incomplete:  — One-key repair"）。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


async def _waiting_task(service, mission, tmp_path: Path) -> str:
    """ask 策略 + 无 operatord → 任务停在 WAITING_APPROVAL。"""
    (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
    import json

    (tmp_path / "agent" / "safety.json").write_text(
        json.dumps({"sim_policy": "ask"}), encoding="utf-8"
    )
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    dispatcher = PiToolDispatcher(service)
    result = await dispatcher.execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_task",
            mission=mission.mission_id,
            idem="idem_cancel_1",
            lease=await _issue_lease(service, mission),
            arguments={
                "goal": "draw_shape",
                "parameters": {"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 0.10},
            },
        ),
    )
    payload = json.loads(result.summary)
    assert payload["state"] == "WAITING_APPROVAL", payload
    return payload["task_id"]


class TestTaskCancel:
    async def test_cancel_waiting_task(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        task_id = await _waiting_task(service, mission, tmp_path)
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.task.cancel",
            {"token": service.control_token, "task_id": task_id},
        )
        assert result.get("ok"), result
        assert result.get("state") == "CANCELLED"
        # resume 被拒（CANCELLED 是终态——结果是 REJECTED）。
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        resumed = await PiToolDispatcher(service).execute(
            caller_pid=1,
            caller_uid=1000,
            request=_request(
                "rosclaw_task",
                mission=mission.mission_id,
                idem="idem_cancel_resume",
                lease=await _issue_lease(service, mission),
                arguments={"task_id": task_id},
            ),
        )
        assert not resumed.ok, "取消的任务竟可 resume"
        import json as _json

        assert _json.loads(resumed.summary)["state"] == "CANCELLED"
        await service.close()

    async def test_cancel_terminal_task_is_honest_noop(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        result = await PiToolDispatcher(service).execute(
            caller_pid=1,
            caller_uid=1000,
            request=_request(
                "rosclaw_task",
                mission=mission.mission_id,
                idem="idem_cancel_2",
                lease=await _issue_lease(service, mission),
                arguments={
                    "goal": "draw_shape",
                    "parameters": {"shape": "star5", "center_m": [0.35, 0.25, 0.30], "radius_m": 0.10},
                },
            ),
        )
        import json

        payload = json.loads(result.summary)
        assert payload["state"] == "VERIFIED"
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        cancel = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.task.cancel",
            {"token": service.control_token, "task_id": payload["task_id"]},
        )
        assert cancel.get("ok")
        assert cancel.get("state") == "VERIFIED", "终态任务被 cancel 改写"
        assert cancel.get("changed") is False
        await service.close()
