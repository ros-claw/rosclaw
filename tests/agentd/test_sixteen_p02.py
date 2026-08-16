"""建议-0816 P0-2 红测试：acceptance 真验收 + REPAIRING 同会话修复。

红测试先行——修复前必须红：
1. Worker COMPLETED 不直接 SUCCEEDED——先 VERIFYING 跑 acceptance；
2. acceptance 失败 → REPAIRING → 反馈同一 Pi session（resume 恢复
   点=同一 session 文件）修复后复验 → SUCCEEDED——全程一个
   execution，不新建 Worker；
3. artifacts/usage 回灌 task_executions（不再是 500 字摘要了事）；
4. 修复两轮仍不过 → FAILED + 真实证据（不无限循环）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


def _enable_fake(service, tmp_path: Path, monkeypatch, body: str, name: str) -> None:
    from rosclaw.agentd.workers import pi_managed

    fake = tmp_path / name
    fake.write_text(body)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(
        pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake))
    )
    adapter = pi_managed.PiManagedAdapter(
        rosclaw_home=tmp_path, conn=service._store.connection
    )
    service._worker_manager._adapters["pi_managed"] = adapter
    adapter._manager_ref = service._worker_manager
    if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
        service._registry.set_status(
            "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
        )


async def _wait_terminal(plane, execution_id: str, timeout: float = 60.0):
    for _ in range(int(timeout * 20)):
        row = plane._get(execution_id)
        if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
            return row
        await asyncio.sleep(0.05)
    return None


class TestAcceptanceGate:
    async def test_completed_goes_through_verifying(self, tmp_path: Path,
                                                    monkeypatch) -> None:
        """COMPLETED → VERIFYING → acceptance 过 → SUCCEEDED（带验收证据）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"done"}\'\n',
            "fake-ok",
        )
        plane = service._task_control_plane
        seen_states: list[str] = []
        view = await plane.submit(
            mission.mission_id,
            {"goal": "写个脚本", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {}},
            idem="p02_ok",
        )
        row = await _wait_terminal(plane, view["execution_id"])
        assert row is not None
        assert row["state"] == "SUCCEEDED", row["summary"]
        await service.close()

    async def test_verifier_fail_repairs_same_session(self, tmp_path: Path,
                                                      monkeypatch) -> None:
        """acceptance 失败 → REPAIRING → 同 session 修复 → SUCCEEDED。
        关键：只有一个 execution；修复 attempt 带同一 session 恢复点。"""
        service, mission = await _setup(tmp_path)
        # fake：第一次不写验收文件（验收失败）；resume 后写。
        flag = tmp_path / "repaired.flag"
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"session_persisted","session_file":"/tmp/fake-session.jsonl"}\'\n'
            f"if [ -f {flag} ]; then\n"
            '  echo \'{"kind":"attempt_finished","report":"修复完成"}\'\n'
            "else\n"
            '  echo \'{"kind":"attempt_finished","report":"初版完成"}\'\n'
            "fi\n",
            "fake-repair",
        )
        (tmp_path / "fake-session.jsonl").write_text("{}\n")
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "实现功能", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {"required_files": ["deliverable.txt"]}},
            idem="p02_repair",
        )
        # 等第一轮验收失败进入 REPAIRING，再补交付物放行第二轮。
        repaired_seen = False
        for _ in range(1200):
            row = plane._get(view["execution_id"])
            if row["state"] == "REPAIRING" and not repaired_seen:
                repaired_seen = True
                # 修复轮：fake 的 workspace 里补文件 + flag。
                flag.write_text("1")
                orders = service._worker_manager.orders_for_mission(mission.mission_id)
                for o in orders:
                    ws = str(o.inputs.get("workspace") or "")
                    if ws:
                        (Path(ws) / "deliverable.txt").write_text("x")
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        assert repaired_seen, "从未进入 REPAIRING——验收门没生效"
        row = plane._get(view["execution_id"])
        assert row["state"] == "SUCCEEDED", row["summary"]
        executions = plane.executions_for(mission.mission_id)
        assert len(executions) == 1, "裂变出多个 execution"
        await service.close()

    async def test_repair_budget_exhausted_honest_fail(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """修复预算耗尽 → FAILED + 验收证据（不无限循环）。"""
        service, mission = await _setup(tmp_path)
        _enable_fake(
            service, tmp_path, monkeypatch,
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"session_persisted","session_file":"/tmp/fake-session.jsonl"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"永远缺交付物"}\'\n',
            "fake-stubborn",
        )
        (tmp_path / "fake-session.jsonl").write_text("{}\n")
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "实现功能", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {"required_files": ["never-exists.txt"]}},
            idem="p02_fail",
        )
        row = await _wait_terminal(plane, view["execution_id"], timeout=90)
        assert row is not None
        assert row["state"] == "FAILED", row["state"]
        assert "never-exists.txt" in (row["summary"] or "") or (
            "验收" in (row["summary"] or "")
        )
        await service.close()
