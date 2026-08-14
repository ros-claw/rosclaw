"""十三审 PR-13.5/13.6 红测试：诊断工具 + 真实 preflight。

1. rosclaw_read_work_failure 返回 verifier 原因/最后事件/stderr/
   checkpoint（模型据此如实诊断）；
2. read_work_events 分页（cursor）；
3. preflight 探测走 Worker 真实执行面（PATH python3 无 PIL 即
   BLOCKED_PREFLIGHT）——十三审修复点：此前探测的是 agentd venv。
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestDiagnosticTools:
    async def test_failure_diagnosis_has_real_content(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = tmp_path / "fake-fail"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "echo 'boom: disk almost full' >&2\n"
            'echo \'{"kind":"attempt_failed","error_code":"WORKER_CRASH","message":"killed"}\'\n'
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        dispatcher = PiToolDispatcher(service)
        done = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_135_fail",
                arguments={"goal": "x", "worker_id": "worker:rosclaw:pi"},
            )
        )
        assert not done.ok
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        diag = await dispatcher.execute(
            _request(
                "rosclaw_read_work_failure",
                mission=mission.mission_id,
                idem="idem_135_diag",
                arguments={"work_order_id": wo},
            )
        )
        assert diag.ok
        assert "诊断" in diag.summary
        assert "boom: disk almost full" in diag.summary
        assert "最后事件" in diag.summary
        events = await dispatcher.execute(
            _request(
                "rosclaw_read_work_events",
                mission=mission.mission_id,
                idem="idem_135_ev",
                arguments={"work_order_id": wo},
            )
        )
        assert events.ok and "attempt_started" in events.summary
        await service.close()


class TestRealPreflight:
    async def test_preflight_uses_worker_python(self, tmp_path: Path, monkeypatch) -> None:
        """agentd venv 有 PIL 不代表 Worker PATH python3 有——探测必须
        走子进程真实执行面（本机系统 python3 若无 PIL 即拒绝）。"""
        import shutil
        import subprocess

        from rosclaw.agentd.workers import pi_managed

        # 本机系统 python3 有没有 PIL 决定预期（真实探测——不再假）。
        sys_python_has_pil = (
            subprocess.run(
                ["python3", "-c", "import PIL"], capture_output=True
            ).returncode
            == 0
        )
        monkeypatch.setattr(shutil, "which", lambda _x: None)  # 无 ffmpeg
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by="test",
            capability="code.develop",
            goal="渲染",
            inputs={"task_type": "artifact_build", "worker_profile": "sim-builder"},
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["image/gif"]),
            side_effect_policy=SideEffectPolicy(**{"class": "sandbox_process"}),
        )
        from rosclaw.agentd.workers.adapter import AdapterError

        if sys_python_has_pil:
            await adapter._preflight(order)  # 不抛
        else:
            with pytest.raises(AdapterError, match="BLOCKED_PREFLIGHT"):
                await adapter._preflight(order)
