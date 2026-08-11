"""WP-P0-3 红测试（总纲 §5.4）：Resume Reconciliation V2。

红测试先行——当前恢复只重接 binding/lease，用户不知道"恢复了
什么、重新验证了什么、哪些权限失效了"。Resume Report 必须说明：
对话/任务（已完成绝不重放、运行中只 attach）/机器人（重新观测）/
权限（旧授权失效规则）。
"""

from __future__ import annotations

import json
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


class TestResumeReport:
    async def test_completed_task_session_report(self, tmp_path: Path) -> None:
        """已完成任务：报告说明"已验证、不会重新执行"。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_rr_1")
        assert result.ok
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
        # 任务行：VERIFIED + 不重放。
        task_line = next(
            (line for line in r["lines"] if "task_" in line), ""
        )
        assert "不会重新执行" in task_line or "已验证" in task_line, r["lines"]
        # 权限行：旧授权失效/当前策略说明。
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
        """WAITING_APPROVAL + 卡已过期 → 报告标记需重新确认（不自动
        恢复执行权）。"""
        service, mission = await _setup(tmp_path)
        (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agent" / "safety.json").write_text(
            json.dumps({"sim_policy": "ask"}), encoding="utf-8"
        )
        result = await _run_task(service, mission, idem="idem_rr_2")
        payload = json.loads(result.summary)
        assert payload["state"] == "WAITING_APPROVAL"
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
        assert r["verdict"] in ("RESUMED", "REAUTH_NEEDED")
        assert any(
            "过期" in line or "重新确认" in line for line in r["lines"]
        ), r["lines"]
        await service.close()
