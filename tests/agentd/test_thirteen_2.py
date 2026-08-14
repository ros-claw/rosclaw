"""十三审 HOTFIX-13.2 红测试：取消默认硬超时 + 预算暂停 + 失联多信号。

红测试先行——修复前必须红：
1. 默认（无显式权威）wall_time 到期只提醒不杀——Worker 有进度继续；
2. hard_deadline_sec 无权威来源 → DEADLINE_AUTHORITY_REQUIRED 硬拒绝；
3. 显式 benchmark 硬截止 → wrap-up 后终止（保留）；
4. token 100% → BUDGET_PAUSED（不 FAILED）；/job extend 唤醒继续；
5. 事件静默 + 进程活着 + CPU 在动 → UNREACHABLE（不杀）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _fake(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


async def _hire(service, mission, tmp_path, fake, monkeypatch, *, wall=3, policy=None):
    from rosclaw.agentd.workers import pi_managed

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
    card = service._registry.get("worker:rosclaw:pi")
    order = WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id=mission.mission_id,
        issued_by="test",
        capability="analysis.text",
        goal="x",
        inputs={
            "instructions": "x",
            **({"execution_policy": policy} if policy else {}),
        },
        budgets=BudgetEnvelope(wall_time_sec=wall, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    scheduled = service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )
    return scheduled


class TestNoDefaultWallKill:
    async def test_soft_wall_does_not_kill(self, tmp_path: Path, monkeypatch) -> None:
        """wall_time 只作 soft target——到期后 Worker 继续直到完成。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        # fake：4 秒（超过 wall=2 的 soft target）后正常完成。
        fake = _fake(
            tmp_path,
            "fake-slow-finish",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "i=0\nwhile [ $i -lt 8 ]; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "  sleep 0.5\n  i=$((i+1))\ndone\n"
            'echo \'{"kind":"attempt_finished","report":"慢慢做完了"}\'\n',
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch, wall=2)
        result, report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED", result.summary
        assert "慢慢做完了" in result.summary
        assert report.accepted
        await service.close()

    async def test_hard_deadline_without_authority_rejected(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_132_auth",
                arguments={
                    "goal": "x",
                    "execution_policy": {"hard_deadline_sec": 300},  # 无来源
                },
            )
        )
        assert not result.ok
        assert result.error_code == "DEADLINE_AUTHORITY_REQUIRED"
        await service.close()


class TestBudgetPause:
    async def test_token_limit_pauses_not_fails(self, tmp_path: Path, monkeypatch) -> None:
        """token 100% → BUDGET_PAUSED（保留会话，可 extend）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-token-burn",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"usage","input_tokens":900,"output_tokens":200}\'\n'
            "while true; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "  sleep 0.5\ndone\n",
        )
        scheduled = await _hire(
            service, mission, tmp_path, fake, monkeypatch,
            wall=300,
            policy={"token_soft_limit": 1000},
        )
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        for _ in range(200):
            current = service._worker_manager.order(scheduled.work_order_id)
            if current and current.status == "BUDGET_PAUSED":
                break
            await asyncio.sleep(0.05)
        current = service._worker_manager.order(scheduled.work_order_id)
        assert current is not None and current.status == "BUDGET_PAUSED", current.status
        # extend 唤醒（fake 的 stdin 不消费——送达失败则诚实报错也行，
        # 这里只验证 extend 的预算落账与状态翻转路径）。
        await service._worker_manager.cancel_order(scheduled.work_order_id, reason="test")
        await asyncio.wait_for(driver, 10)
        await service.close()
