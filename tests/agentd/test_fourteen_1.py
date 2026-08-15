"""十四审 PR-14.1 红测试：终止 exit130 循环 + 控制协议 ACK + 终止原因权威。

红测试先行——修复前必须红：
1. token soft target 只告警，绝不暂停进程（总纲 §3.1）；
2. 控制协议必须 ACK：request_pause 在 control.ack PAUSED 前不得声称暂停；
3. pause 后进程必须存活、resume 后同会话完成（总纲 Gate 0）；
4. 无 termination.json 的 exit 130 = SIGNAL_UNKNOWN → INTERRUPTED_RESUMABLE，
   不是 FAILED（exit code 不得直接当语义）；
5. termination.json 的 PROVIDER_FATAL → FAILED 且摘要带权威 cause；
6. 显式 hard cost limit（user 权威）才允许预算暂停，且先 ACK 再落
   BUDGET_PAUSED。
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


async def _hire(service, mission, tmp_path, fake, monkeypatch, *, wall=60, policy=None,
                tokens=1000):
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
        budgets=BudgetEnvelope(wall_time_sec=wall, model_tokens=tokens),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    return service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )


#: 新控制协议的 fake worker：pause → ACK PAUSED 并空转等待；resume →
#: ACK RUNNING 并完成。关键：pause 后进程必须活着（Gate 0）。
#: 注意：stdin 读取者必须在前台（POSIX sh 后台任务 stdin 是 /dev/null）。
_FAKE_CONTROL = """#!/bin/sh
echo '{"kind":"attempt_started"}'
echo '{"kind":"usage","input_tokens":600,"output_tokens":200}'
(
  i=0
  while [ $i -lt 20 ]; do
    echo '{"kind":"liveness","phase":"RUNNING_MODEL"}'
    sleep 0.5
    i=$((i+1))
  done
) &
while IFS= read -r line; do
  case "$line" in
    *'"action": "pause"'*)
      cid=$(echo "$line" | sed -n 's/.*"control_id": "\\([^"]*\\)".*/\\1/p')
      echo "{\\"kind\\":\\"control.ack\\",\\"control_id\\":\\"$cid\\",\\"state\\":\\"PAUSED\\"}"
      ;;
    *'"action": "resume"'*)
      cid=$(echo "$line" | sed -n 's/.*"control_id": "\\([^"]*\\)".*/\\1/p')
      echo "{\\"kind\\":\\"control.ack\\",\\"control_id\\":\\"$cid\\",\\"state\\":\\"RUNNING\\"}"
      echo '{"kind":"attempt_finished","report":"同会话继续并完成"}'
      exit 0
      ;;
  esac
done
"""


class TestSoftTokenNeverPauses:
    async def test_soft_token_limit_only_warns(self, tmp_path: Path, monkeypatch) -> None:
        """token 超 soft target → 只 budget_warning；绝不 BUDGET_PAUSED，
        进程继续到完成（总纲 §3.1：soft target 不能控制进程状态）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-token-over",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"usage","input_tokens":900,"output_tokens":200}\'\n'
            "i=0\nwhile [ $i -lt 4 ]; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "  sleep 0.5\n  i=$((i+1))\ndone\n"
            'echo \'{"kind":"attempt_finished","report":"做完了"}\'\n',
        )
        scheduled = await _hire(
            service, mission, tmp_path, fake, monkeypatch,
            policy={"token_soft_limit": 1000},
        )
        result, report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED", result.summary
        # 警告存在、暂停不存在。
        adapter = service._worker_manager._adapters["pi_managed"]
        events = adapter._events.tail(scheduled.work_order_id, limit=500)
        kinds = [e["kind"] for e in events]
        assert "budget_warning" in kinds
        assert "budget_paused" not in kinds
        assert service._worker_manager.order(scheduled.work_order_id).status in (
            "ACCEPTED", "SUBMITTED", "VERIFYING",
        )
        await service.close()


class TestControlProtocolAck:
    async def test_pause_ack_then_resume_same_process(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """request_pause 必须等 control.ack PAUSED 才返回 True；pause 后
        进程存活；request_resume 后同进程完成（Gate 0 核心）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(tmp_path, "fake-control", _FAKE_CONTROL)
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        # 等 attempt_started 出现（进程起来了）。
        for _ in range(100):
            if service._worker_manager._adapters["pi_managed"]._events.tail(scheduled.work_order_id):
                break
            await asyncio.sleep(0.05)
        # pause：ACK 语义——返回 True 时 worker 必须已确认 PAUSED。
        assert await adapter.request_pause(scheduled.work_order_id, reason="user")
        proc = adapter._procs[scheduled.work_order_id]
        assert proc.returncode is None, "pause 后进程退出——exit130 缺陷回归"
        # resume：同进程继续并完成。
        assert await adapter.request_resume(scheduled.work_order_id)
        result, _report = await asyncio.wait_for(driver, 30)
        assert result.status == "COMPLETED", result.summary
        assert "同会话继续并完成" in result.summary
        await service.close()

    async def test_pause_without_ack_returns_false(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """worker 不 ACK（旧协议/异常）→ request_pause 诚实 False，
        不得乐观声称已暂停。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        monkeypatch.setattr(pi_managed, "CONTROL_ACK_TIMEOUT_SEC", 1.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-no-ack",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "i=0\nwhile [ $i -lt 6 ]; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "  sleep 0.5\n  i=$((i+1))\ndone\n"
            'echo \'{"kind":"attempt_finished","report":"done"}\'\n',
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        adapter = service._worker_manager._adapters["pi_managed"]
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        for _ in range(100):
            if service._worker_manager._adapters["pi_managed"]._events.tail(scheduled.work_order_id):
                break
            await asyncio.sleep(0.05)
        assert not await adapter.request_pause(scheduled.work_order_id, reason="user")
        result, _report = await asyncio.wait_for(driver, 30)
        assert result.status == "COMPLETED"
        await service.close()


class TestTerminationCause:
    async def test_bare_exit130_is_interrupted_not_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """无 termination.json 的 exit 130 = SIGNAL_UNKNOWN →
        INTERRUPTED_RESUMABLE（可恢复），绝不是 FAILED（§3.4）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-exit130",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "sleep 0.5\n"
            "exit 130\n",
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        result, _report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "INTERRUPTED", result.summary
        order = service._worker_manager.order(scheduled.work_order_id)
        assert order.status == "INTERRUPTED_RESUMABLE", order.status
        await service.close()

    async def test_termination_json_provider_fatal_failed_with_cause(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """termination.json 是终态原因唯一权威——FAILED 摘要必须带
        结构化 cause（模型不得再猜日志归因，§1.4/§3.4）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-provider-fatal",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_failed","error_code":"MODEL_ERROR",'
            '"message":"provider 401 unauthorized"}\'\n'
            # 原子写 termination.json（work/<wo>/termination.json）。
            "TERM_DIR=$(dirname \"$0\")/work\n"
            "for d in \"$TERM_DIR\"/wo_*; do\n"
            '  echo \'{"cause":"PROVIDER_FATAL","detail":"provider 401"}\''
            ' > "$d/termination.json.tmp"\n'
            '  mv "$d/termination.json.tmp" "$d/termination.json"\n'
            "done\n"
            "exit 1\n",
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        result, _report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        assert "PROVIDER_FATAL" in result.summary, result.summary
        assert "worker exited" not in result.summary
        await service.close()


class TestHardCostLimit:
    async def test_hard_cost_pause_ack_then_extend_completes(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """显式 hard cost limit（user 权威）：到限 → 控制暂停（ACK 后
        BUDGET_PAUSED）→ extend 追加并 resume → 同进程完成。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(tmp_path, "fake-hard-cost", _FAKE_CONTROL)
        scheduled = await _hire(
            service, mission, tmp_path, fake, monkeypatch,
            policy={
                "cost_hard_limit_tokens": 500,
                "cost_hard_limit_source": "user",
            },
        )
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        # 到限后应进入 BUDGET_PAUSED（ACK 已回）。
        paused = False
        for _ in range(200):
            current = service._worker_manager.order(scheduled.work_order_id)
            if current and current.status == "BUDGET_PAUSED":
                paused = True
                break
            await asyncio.sleep(0.05)
        assert paused, "hard cost limit 未触发 BUDGET_PAUSED"
        # 进程必须仍然存活（Gate 0：pause ≠ 终止）。
        adapter = service._worker_manager._adapters["pi_managed"]
        proc = adapter._procs[scheduled.work_order_id]
        assert proc.returncode is None, "budget pause 后进程退出——exit130 回归"
        # extend = 追加预算 + resume。
        dispatcher = PiToolDispatcher(service)
        extended = await dispatcher.execute(
            _request(
                "rosclaw_extend_work",
                mission=mission.mission_id,
                idem="idem_141_extend",
                arguments={"work_order_id": scheduled.work_order_id,
                           "add_tokens": 5000},
            )
        )
        assert extended.ok, extended.summary
        result, _report = await asyncio.wait_for(driver, 30)
        assert result.status == "COMPLETED", result.summary
        await service.close()

    async def test_hard_cost_without_authority_rejected(self, tmp_path: Path) -> None:
        """cost_hard_limit 无 user/admin_policy 权威 → 硬拒绝（同硬截止
        权威规则——模型自定数字不能控制进程，§3.1）。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_141_cost_auth",
                arguments={
                    "goal": "x",
                    "execution_policy": {"cost_hard_limit_tokens": 100},
                },
            )
        )
        assert not result.ok
        assert result.error_code == "COST_LIMIT_AUTHORITY_REQUIRED"
        await service.close()
