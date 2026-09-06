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
    """大道至简 R0-2b：无 recipe 链——任务产物经通用能力工具产生
    （rosclaw_compute：plan → simulate，capability 产物自动登记）。
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
    # 渲染交付物（lineage+render receipt 血缘产物——终态权威门禁
    # 认可的受信执行证据）。
    result = await dispatcher.execute(
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
    return result


def _bridge(service, tmp_path: Path):
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    return PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")





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
        result = await _run_task(service, mission, idem="idem_rr_1")
        assert result.ok, result.summary
        # R0-2b：capability 产物落在 admission 建的任务上——以产物
        # 归属为准（不再是 bind+recipe 双段式）。
        row = service._store.connection.execute(
            "SELECT task_id FROM artifacts ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert row, "能力执行无产物登记"
        _finish_kernel(service, str(row["task_id"]))
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
