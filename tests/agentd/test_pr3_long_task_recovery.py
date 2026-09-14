"""0914 PR-3：长任务等待与恢复——计时域拆分 + 完成可发现性。

实证（0914 体验日志 + 代码核验）：
1. ProviderStallWatchdog 度量"pi 事件静默"而非"Provider 停滞"——
   90s 渲染/sleep 工具期间无任何事件，45s streamIdle 误杀模型请求，
   后台 Operation 却 SUCCEEDED（"Operation 成功与 Provider idle
   取消并存"）；
2. process_start 的 summary 只说"不要死等"却不告诉模型该做什么
   ——模型 sleep25/sleep60 轮询（撞上 45s 误杀）；
3. resume_report 只报任务终态——不含已完成 Operation 与其产物
   路径，恢复后模型不知道 r1 渲染已成功，复制视频再登记（恢复
   语义腐烂）。

闭环断言：工具阶段 Provider 时钟暂停；process_start 指引"结束
回合、完成自动推送、禁止轮询"；resume_report 列出终态 Operation
与产物路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agentd.test_wp03_resume_report import (
    _bridge,
    _run_task,
    _setup,
)


class TestProcessStartGuidance:
    async def test_summary_guides_turn_end_no_polling(self, tmp_path: Path) -> None:
        """process_start 的 summary 必须告诉模型**做什么**（结束回合、
        完成自动推送）而不只是"不要死等"——且明确禁止 sleep/轮询
        （0914 实证：模型 sleep25/sleep60 撞上 45s 误杀）。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_pr3_guide")
        assert result.ok, result.summary
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from rosclaw.contracts.pi.tool_request import PiToolRequestV1

        disp = PiToolDispatcher(service)
        req = PiToolRequestV1(
            request_id="ptr_pr3",
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            tool_name="rosclaw_process_start",
            arguments={"command": "sleep 0.1"},
            requested_at="2026-09-14T00:00:00+00:00",
            idempotency_key="idem_pr3_start",
        )
        res = await disp._process_start(req)
        summary = res.summary
        assert "结束" in summary and "回合" in summary, (
            f"summary 未指引结束回合: {summary}"
        )
        assert "推送" in summary or "通知" in summary
        assert "轮询" in summary or "sleep" in summary.lower(), (
            f"summary 未明确禁止轮询/sleep: {summary}"
        )
        await service.close()


class TestResumeReportOperations:
    async def test_report_lists_terminal_operations_with_artifacts(
        self, tmp_path: Path
    ) -> None:
        """resume_report 必须列出终态 Operation 与产物路径——恢复后
        模型直接复用成功产物（0914 实证：模型不知 r1 渲染已成功，
        复制视频再登记）。"""
        service, mission = await _setup(tmp_path)
        result = await _run_task(service, mission, idem="idem_pr3_ops")
        assert result.ok, result.summary
        row = service._store.connection.execute(
            "SELECT task_id, path FROM artifacts ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert row, "能力执行无产物登记"
        # 种一条终态 Operation（渲染类能力路径不走 OperationManager
        # ——报告契约对 operations 表负责，与来源无关）。
        service._store.connection.execute(
            "INSERT INTO operations (operation_id, task_id, attempt_id, kind,"
            " state, resumable, started_at, ended_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (
                "op_pr3_terminal", str(row["task_id"]), "main", "render",
                "SUCCEEDED", "2026-09-14T00:00:00+00:00",
                "2026-09-14T00:01:30+00:00",
            ),
        )
        bridge = _bridge(service, tmp_path)
        report = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.session.resume_report",
            {"token": service.control_token, "pi_session_id": "pi_1"},
        )
        assert report.get("ok"), report
        lines = report["report"]["lines"]
        joined = "\n".join(lines)
        # 产物路径必须出现在报告里（模型可复用——不再复制/重渲染）。
        assert row["path"] in joined or str(row["path"]).split("/")[-1] in joined, (
            f"报告未含产物路径: {lines}"
        )
        # 终态 Operation 行（SUCCEEDED/FAILED 可见）。
        assert any(
            ("SUCCEEDED" in line or "已完成" in line) and "op" in line.lower()
            or "Operation" in line
            for line in lines
        ), f"报告未列终态 Operation: {lines}"
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
