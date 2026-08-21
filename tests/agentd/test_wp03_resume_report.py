"""WP-P0-3 红测试（总纲 §5.4）：Resume Reconciliation V2。

红测试先行——当前恢复只重接 binding/lease，用户不知道"恢复了
什么、重新验证了什么、哪些权限失效了"。Resume Report 必须说明：
对话/任务（已完成绝不重放、运行中只 attach）/机器人（重新观测）/
权限（旧授权失效规则）。
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


def _bridge(service, tmp_path: Path):
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    return PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")




def _bind_and_finish(service, mission, *, idem: str) -> str:
    """PR-H9：kernel 任务 + rosclaw_task 产物 + Verifier 收尾
    （SUCCEEDED 的唯一合法路径）。"""
    import asyncio as _asyncio  # noqa: F401

    bound = service._task_kernel.bind_message(
        mission_id=mission.mission_id, session_ref="pi_1",
        backend_native_id="pi_1", message_id=f"msg_{idem}",
        text="画五角星", cwd=str(service._home),
        body_id=mission.body_binding.body_id,
    )
    task_id = str(bound["task_id"])
    return task_id


def _finish_kernel(service, task_id: str) -> None:
    artifacts = [
        str(r["artifact_id"])
        for r in service._store.connection.execute(
            "SELECT artifact_id FROM artifacts WHERE task_id = ?", (task_id,)
        ).fetchall()
    ]
    service._task_kernel.finish_task(
        task_id=task_id, summary="五角星仿真完成", artifact_ids=artifacts
    )

class TestResumeReport:
    async def test_completed_task_session_report(self, tmp_path: Path) -> None:
        """已验收任务（kernel SUCCEEDED）：报告说明"已验收、不会重新
        执行"。"""
        service, mission = await _setup(tmp_path)
        task_id = _bind_and_finish(service, mission, idem="rr_1")
        result = await _run_task(service, mission, idem="idem_rr_1")
        assert result.ok
        _finish_kernel(service, task_id)
        bridge = _bridge(service, tmp_path)
        report = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.session.resume_report",
            {"token": service.control_token, "pi_session_id": "pi_1"},
        )
        assert report.get("ok"), report
        r = report["report"]
        assert r["verdict"] == "RESUMED"
        assert r["mode"] == "SIMULATION"
        assert r["body_id"]
        task_line = next(
            (line for line in r["lines"] if "task_" in line), ""
        )
        assert "不会重新执行" in task_line or "已验收" in task_line, r["lines"]
        assert any("授权" in line or "策略" in line for line in r["lines"])
        await service.close()

    async def test_missing_mission_is_read_only(self, tmp_path: Path) -> None:
        """Mission 不存在：只读恢复判定，不伪装成原 Mission。"""
        service, mission = await _setup(tmp_path)
        # 破坏：删除 mission 行。
        service._store.connection.execute(
            "DELETE FROM missions WHERE mission_id = ?", (mission.mission_id,)
        )
        bridge = _bridge(service, tmp_path)
        report = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.session.resume_report",
            {"token": service.control_token, "pi_session_id": "pi_1"},
        )
        assert report.get("ok")
        assert report["report"]["verdict"] == "READ_ONLY"
        await service.close()

    async def test_waiting_approval_expired_flagged(self, tmp_path: Path) -> None:
        """过期 PENDING 授权卡（broker 侧）→ 报告 REAUTH_NEEDED，
        不自动恢复执行权。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.action_dispatch import request_approval
        from rosclaw.contracts.agent.decision import DecisionV1

        decision = DecisionV1.model_validate_contract(
            {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "dec_rr_expired",
                "mission_id": mission.mission_id,
                "context_id": f"ctx_{mission.mission_id}",
                "context_revision": 1,
                "next_intent": "REQUEST_APPROVAL",
                "summary": "请求授权",
                "proposed_operation": {
                    "type": "approval_request",
                    "payload": {
                        "capability_id": "sim.hold_position",
                        "arguments": {},
                        "risk_tier": "LOW",
                    },
                },
            }
        )
        await request_approval(
            service, decision, mode="SIMULATION", principal="user:local:1000"
        )
        # 让卡过期。
        service._store.connection.execute(
            "UPDATE operator_requests SET request_json = "
            "json_set(request_json, '$.expires_at', '2000-01-01T00:00:00+00:00')"
        )
        bridge = _bridge(service, tmp_path)
        report = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.session.resume_report",
            {"token": service.control_token, "pi_session_id": "pi_1"},
        )
        r = report["report"]
        assert r["verdict"] == "REAUTH_NEEDED", r
        assert any(
            "过期" in line or "重新确认" in line for line in r["lines"]
        ), r["lines"]
        await service.close()
