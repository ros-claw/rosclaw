"""十四审 PR-14.5 红测试：agentd 重启恢复（总纲 §1.7/PR-14.5）。

红测试先行——修复前必须红：
1. RUNNING 的 WorkOrder 在重启对账后必须 INTERRUPTED_RESUMABLE
   （禁止 FAILED——合并红线），session/workspace 保留可 resume；
2. 孤儿子进程按 SIGTERM→SIGKILL 清理（不留计费孤儿）；
3. OFFERED/CLAIMED → CANCELLED（从未启动，诚实取消）；
4. INTERRUPTED_RESUMABLE 不被二次对账（幂等）；
5. 对账后 rosclaw_resume_work 能从同一 session 开新 attempt。
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _order(mission_id: str, wo: str) -> WorkOrderV1:
    return WorkOrderV1(
        work_order_id=wo,
        mission_id=mission_id,
        issued_by="test",
        capability="analysis.text",
        goal="长任务",
        inputs={"instructions": "x"},
        budgets=BudgetEnvelope(wall_time_sec=600, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )


def _seed_running(service, mission_id: str, wo: str, *, with_child: bool = True,
                  with_session: bool = True) -> subprocess.Popen | None:
    """直接落库一个 RUNNING 单 + 可选孤儿进程/session（模拟重启前状态）。"""
    conn = service._store.connection
    order = _order(mission_id, wo).model_copy(update={"status": "RUNNING"})
    conn.execute(
        "INSERT INTO work_orders (work_order_id, mission_id, capability, status, "
        "order_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (wo, mission_id, "analysis.text", "RUNNING", order.model_dump_json(),
         "2026-08-14T00:00:00+00:00", "2026-08-14T00:00:00+00:00"),
    )
    work_dir = service._home / "work" / wo
    session_dir = work_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = ""
    if with_session:
        session_file = str(session_dir / "sess-1.jsonl")
        Path(session_file).write_text('{"type":"session"}\n', encoding="utf-8")
    (work_dir / "state.json").write_text(
        json.dumps({"status": "RUNNING", "session_file": session_file}),
        encoding="utf-8",
    )
    proc = None
    if with_child:
        proc = subprocess.Popen(
            ["sleep", "300"], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        (work_dir / "child.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc


class TestRestartReconcile:
    async def test_running_becomes_interrupted_resumable_not_failed(
        self, tmp_path: Path
    ) -> None:
        """合并红线：agentd 重启绝不把运行任务标 FAILED。"""
        service, mission = await _setup(tmp_path)
        wo = new_id("wo")
        proc = _seed_running(service, mission.mission_id, wo)
        assert proc is not None
        reconciled = await service.reconcile_workers_on_start()
        assert wo in reconciled
        order = service._worker_manager.order(wo)
        assert order.status == "INTERRUPTED_RESUMABLE", order.status
        assert order.status != "FAILED"
        # 孤儿进程被清理（不留计费孤儿）。
        time.sleep(0.2)
        assert proc.poll() is not None, "孤儿子进程未被清理"
        # session_file 保留（resume 的恢复点）。
        state = json.loads(
            (tmp_path / "work" / wo / "state.json").read_text(encoding="utf-8")
        )
        assert state["session_file"].endswith("sess-1.jsonl")
        await service.close()

    async def test_offered_claimed_become_cancelled(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        conn = service._store.connection
        wo = new_id("wo")
        order = _order(mission.mission_id, wo).model_copy(update={"status": "CLAIMED"})
        conn.execute(
            "INSERT INTO work_orders (work_order_id, mission_id, capability, status, "
            "order_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (wo, mission.mission_id, "analysis.text", "CLAIMED",
             order.model_dump_json(),
             "2026-08-14T00:00:00+00:00", "2026-08-14T00:00:00+00:00"),
        )
        await service.reconcile_workers_on_start()
        assert service._worker_manager.order(wo).status == "CANCELLED"
        await service.close()

    async def test_interrupted_resumable_not_re_reconciled(
        self, tmp_path: Path
    ) -> None:
        """二次重启对账幂等：INTERRUPTED_RESUMABLE 是等待恢复的诚实
        状态，不被反复改写。"""
        service, mission = await _setup(tmp_path)
        wo = new_id("wo")
        _seed_running(service, mission.mission_id, wo, with_child=False)
        await service.reconcile_workers_on_start()
        first = service._worker_manager.order(wo)
        assert first.status == "INTERRUPTED_RESUMABLE"
        reconciled2 = await service.reconcile_workers_on_start()
        assert wo not in reconciled2
        assert service._worker_manager.order(wo).status == "INTERRUPTED_RESUMABLE"
        await service.close()

    async def test_resume_after_restart_opens_new_attempt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """对账 → INTERRUPTED_RESUMABLE → rosclaw_resume_work 开新
        attempt（同一 session 恢复——retry ≠ resume）。"""
        from rosclaw.agentd.workers import pi_managed

        service, mission = await _setup(tmp_path)
        wo = new_id("wo")
        _seed_running(service, mission.mission_id, wo, with_child=False)
        await service.reconcile_workers_on_start()
        assert service._worker_manager.order(wo).status == "INTERRUPTED_RESUMABLE"
        # resume：fake worker 立即完成（新 attempt 携带 _resume_session）。
        fake = tmp_path / "fake-resume"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"session_resumed","from":"x"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"重启后同会话完成"}\'\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
            )
        dispatcher = PiToolDispatcher(service)
        resumed = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_145_resume",
                arguments={"work_order_id": wo},
            )
        )
        assert resumed.ok, resumed.summary
        view = service._worker_manager.job_view(wo)
        assert view is not None and len(view["attempts"]) == 2
        new_wo = view["attempts"][1]["attempt_id"]
        for _ in range(200):
            current = service._worker_manager.order(new_wo)
            if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED"):
                break
            import asyncio

            await asyncio.sleep(0.05)
        current = service._worker_manager.order(new_wo)
        assert current.status in ("ACCEPTED", "SUBMITTED", "VERIFYING"), current.status
        await service.close()


class TestOrphanCleanup:
    async def test_sigterm_grace_then_sigkill(self, tmp_path: Path) -> None:
        """SIGTERM 给 worker 落 termination.json 的机会；顽固进程
        SIGKILL 兜底。"""
        service, mission = await _setup(tmp_path)
        wo = new_id("wo")
        # 忽略 SIGTERM 且自身存活的顽固孤儿——必须 SIGKILL 收场。
        proc = subprocess.Popen(
            ["sh", "-c", "trap '' TERM; while true; do sleep 1; done"],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        work_dir = tmp_path / "work" / wo
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "child.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        (work_dir / "state.json").write_text(
            json.dumps({"status": "RUNNING", "session_file": ""}), encoding="utf-8"
        )
        conn = service._store.connection
        order = _order(mission.mission_id, wo).model_copy(update={"status": "RUNNING"})
        conn.execute(
            "INSERT INTO work_orders (work_order_id, mission_id, capability, status, "
            "order_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (wo, mission.mission_id, "analysis.text", "RUNNING",
             order.model_dump_json(),
             "2026-08-14T00:00:00+00:00", "2026-08-14T00:00:00+00:00"),
        )
        await service.reconcile_workers_on_start()
        time.sleep(0.2)
        assert proc.poll() is not None, "顽固孤儿未被 SIGKILL"
        assert proc.returncode == -signal.SIGKILL
        await service.close()
