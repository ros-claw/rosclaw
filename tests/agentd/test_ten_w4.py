"""十审 Gate W4 红测试：steer 通道 / 崩溃对账 / retry lineage / token 预算。

红测试先行——修复前必须红：
1. _update_work 对运行中的内置 Worker 实时送达 steer（stdin 通道），
   不再只是落账备注；
2. agentd 重启对账：非终态单标 FAILED/CANCELLED + pid 文件的孤儿
   进程组被杀；
3. rosclaw_retry_work：终态单新 attempt 携带 steer 备注 + parent/root
   lineage；运行中单拒绝；
4. mission token 预算：超过 ROSCLAW_WORKER_TOKEN_BUDGET 诚实拒绝。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestSteerChannel:
    async def test_update_delivers_steer_to_running_pi_worker(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        received = tmp_path / "steer.txt"
        fake = tmp_path / "fake-entry"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "while IFS= read -r line; do\n"
            f"  echo \"$line\" >> {received}\n"
            "done\n"
        )
        fake.chmod(0o755)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        service._worker_manager._adapters["pi_managed"] = adapter
        # CI 无 node：service 初始化时按探测诚实 DISABLED——本测试自带
        # fake entry，强制 ENABLED（产品探测路径由 W1 测试覆盖）。
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="test fake entry"
            )

        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w4_steer",
                arguments={
                    "goal": "长任务",
                    "worker_id": "worker:rosclaw:pi",
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.status == "STARTED", started.summary
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        # 等子进程真正起来（steer 通道注册）。
        for _ in range(200):
            if wo in adapter._procs:
                break
            await asyncio.sleep(0.05)
        assert wo in adapter._procs, "worker 子进程未注册"
        updated = await dispatcher.execute(
            _request(
                "rosclaw_update_work",
                mission=mission.mission_id,
                idem="idem_w4_steer2",
                arguments={"work_order_id": wo, "note": "只看 src/ 目录"},
            )
        )
        assert updated.ok
        assert "实时送达" in updated.summary, updated.summary
        for _ in range(100):
            if received.exists() and "只看 src" in received.read_text():
                break
            await asyncio.sleep(0.05)
        assert received.exists() and "steer" in received.read_text()
        assert "只看 src" in received.read_text()
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w4_steer_c",
                arguments={"work_order_id": wo},
            )
        )
        await service.close()


class TestCrashReconciliation:
    async def test_restart_marks_orphans_and_kills_children(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        card = service._registry.get("worker:native:basic")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="analysis.text",
            goal="x",
            inputs={"instructions": "x"},
            budgets=BudgetEnvelope(wall_time_sec=300, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        assert scheduled.status == "RUNNING"
        # 伪造 child.pid（真实起个 sleep 进程组当孤儿）。
        work_dir = tmp_path / "work" / scheduled.work_order_id
        work_dir.mkdir(parents=True)
        sleeper = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", "sleep 300 & sleep 300",
            start_new_session=True,
        )
        (work_dir / "child.pid").write_text(f"{sleeper.pid}\n")
        pgid = os.getpgid(sleeper.pid)
        # 模拟重启：内存 run 注册表清空，直接对账。
        reconciled = await service.reconcile_workers_on_start()
        assert scheduled.work_order_id in reconciled
        current = service._worker_manager.order(scheduled.work_order_id)
        # 十四审 PR-14.5：RUNNING 重启不再 FAILED——INTERRUPTED_RESUMABLE
        # （会话/工作区保留，可 resume；孤儿进程组仍必须杀）。
        assert current is not None and current.status == "INTERRUPTED_RESUMABLE"
        # 孤儿进程组被杀。
        for _ in range(70):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                break
            await asyncio.sleep(0.1)
        with pytest.raises((ProcessLookupError, PermissionError)):
            os.killpg(pgid, 0)
        await service.close()


class TestRetryWork:
    async def test_retry_terminal_order_carries_notes_and_lineage(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        done = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w4_r1",
                arguments={"goal": "快任务", "worker_id": "auto"},
            )
        )
        assert done.status == "COMPLETED"
        old = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        # 运行中拒绝：先造一个 RUNNING 单？（直接断言终态路径 + 非终态拒绝）
        retry = await dispatcher.execute(
            _request(
                "rosclaw_retry_work",
                mission=mission.mission_id,
                idem="idem_w4_r2",
                arguments={"work_order_id": old.work_order_id},
            )
        )
        assert retry.ok, retry.summary
        assert retry.status == "STARTED"
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 2
        child = orders[1]
        assert child.parent_work_order_id == old.work_order_id
        assert child.root_work_order_id == old.work_order_id
        assert old.work_order_id in retry.summary
        # 非终态单拒绝 retry。
        from tests.agentd.test_ten_w0 import _register_stub, _slow_adapter_module

        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w4_r3",
                arguments={"goal": "慢任务", "worker_id": "worker:stub:slow"},
            )
        )
        assert started.status == "STARTED"
        running = service._worker_manager.orders_for_mission(mission.mission_id)[-1]
        refused = await dispatcher.execute(
            _request(
                "rosclaw_retry_work",
                mission=mission.mission_id,
                idem="idem_w4_r4",
                arguments={"work_order_id": running.work_order_id},
            )
        )
        assert not refused.ok and refused.error_code == "NOT_TERMINAL"
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w4_r5",
                arguments={"work_order_id": running.work_order_id},
            )
        )
        await service.close()


class TestTokenBudget:
    async def test_mission_token_budget_exhausted_honest(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # MockModelGateway usage：首单花费 2 tokens——cap=2 时第二单必须被拒。
        monkeypatch.setenv("ROSCLAW_WORKER_TOKEN_BUDGET", "2")
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        done = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w4_b1",
                arguments={"goal": "快任务", "worker_id": "auto"},
            )
        )
        assert done.status == "COMPLETED"
        refused = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w4_b2",
                arguments={"goal": "再来一个", "worker_id": "auto"},
            )
        )
        assert not refused.ok
        assert refused.error_code == "WORKER_BUDGET_EXHAUSTED"
        await service.close()
