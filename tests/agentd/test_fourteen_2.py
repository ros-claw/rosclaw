"""十四审 PR-14.2 红测试（Gate 1）：重试一致性——唯一 RetryCoordinator。

红测试先行——修复前必须红：
1. transient failure 的 auto retry 与手动 rosclaw_retry_work 并发/相继
   触发：只有一个 root Job、同时最多一个 active attempt、手动返回的
   是同一个 attempt（不再裂变成三个 WorkOrder）；
2. 用户重复 retry → 返回同一 attempt；
3. USER_CANCELLED / PROVIDER_FATAL / DELIVERABLE_REJECTED 不自动重试；
4. coordinator CAS：两个并发 request_retry 只创建一个 attempt；
5. "worker exited" 不再是任何自动重试依据（markers 已删除）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.agentd.workers.retry import (
    parse_cause,
    should_auto_retry,
)
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


async def _hire(service, mission, tmp_path, fake, monkeypatch, *, wall=60):
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
        inputs={"instructions": "x"},
        budgets=BudgetEnvelope(wall_time_sec=wall, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    return service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )


#: 第一次运行失败（PROVIDER_TRANSIENT），被 retry 后成功——用标志位
#: 文件区分（retry 复用同一 tmp_path 下的不同 wo 目录，标志放 tmp_path）。
def _flaky_fake(tmp_path: Path) -> Path:
    return _fake(
        tmp_path,
        "fake-flaky",
        "#!/bin/sh\n"
        'echo \'{"kind":"attempt_started"}\'\n'
        "FLAG=$(dirname \"$0\")/flaky-ok\n"
        "if [ -f \"$FLAG\" ]; then\n"
        '  echo \'{"kind":"attempt_finished","report":"retry 后完成"}\'\n'
        "  exit 0\n"
        "fi\n"
        "touch \"$FLAG\"\n"
        'echo \'{"kind":"attempt_failed","error_code":"PROVIDER_TIMEOUT",'
        '"message":"provider 429 rate limit"}\'\n'
        "for d in $(dirname \"$0\")/work/wo_*; do\n"
        '  echo \'{"cause":"PROVIDER_TRANSIENT","detail":"429"}\''
        ' > "$d/termination.json"\n'
        "done\n"
        "exit 1\n",
    )


class TestRetryConsistency:
    async def test_auto_and_manual_retry_converge_to_one_attempt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Gate 1 核心：auto retry（transient）+ 手动 retry 不得裂变——
        只有一个 root Job、两个 attempt（原始+一次 retry）、手动请求
        返回同一个 retry attempt。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _flaky_fake(tmp_path)
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        root = scheduled.work_order_id
        # 驱动到终态（auto retry 应在 _drive_worker 内触发）。
        await service._drive_worker(scheduled)
        order = service._worker_manager.order(root)
        assert order.status == "FAILED", order.status
        # auto retry 已创建（同 root 的第二个 attempt）。
        view = service._worker_manager.job_view(root)
        assert view is not None, "稳定 Job 账本不存在"
        assert len(view["attempts"]) == 2, (
            f"attempts={len(view['attempts'])}——auto retry 未触发或重复创建"
        )
        auto_attempt_id = view["attempts"][1]["attempt_id"]
        assert view["attempts"][1]["actor"] == "auto"
        # 手动 retry（Native Agent）——必须返回同一个 attempt，不新建。
        dispatcher = PiToolDispatcher(service)
        manual = await dispatcher.execute(
            _request(
                "rosclaw_retry_work",
                mission=mission.mission_id,
                idem="idem_142_manual",
                arguments={"work_order_id": root},
            )
        )
        assert manual.ok, manual.summary
        assert auto_attempt_id in manual.summary, manual.summary
        view = service._worker_manager.job_view(root)
        assert len(view["attempts"]) == 2, "手动 retry 创建了第二个 attempt——裂变回归"
        # 清理：等 retry attempt 收尾（flaky fake 第二次会成功）。
        for _ in range(200):
            current = service._worker_manager.order(auto_attempt_id)
            if current and current.status in (
                "ACCEPTED", "FAILED", "CANCELLED", "SUBMITTED", "VERIFYING",
            ):
                break
            await asyncio.sleep(0.05)
        await service.close()

    async def test_duplicate_manual_retry_returns_existing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """用户重复 retry → 同一 attempt（活跃去重）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        # 永久运行 fake——retry 后 attempt 保持 ACTIVE。
        fake = _fake(
            tmp_path,
            "fake-forever",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "FLAG=$(dirname \"$0\")/forever-ok\n"
            "if [ -f \"$FLAG\" ]; then\n"
            "  while true; do echo '{\"kind\":\"liveness\"}'; sleep 0.5; done\n"
            "fi\n"
            "touch \"$FLAG\"\n"
            'echo \'{"kind":"attempt_failed","error_code":"MODEL_ERROR",'
            '"message":"provider 401"}\'\n'
            "for d in $(dirname \"$0\")/work/wo_*; do\n"
            '  echo \'{"cause":"PROVIDER_FATAL","detail":"401"}\''
            ' > "$d/termination.json"\n'
            "done\n"
            "exit 1\n",
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        root = scheduled.work_order_id
        result, _ = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        # PROVIDER_FATAL 不得自动重试——attempts 仍只有 1 个。
        view = service._worker_manager.job_view(root)
        assert len(view["attempts"]) == 1, "PROVIDER_FATAL 被自动重试——违反 §3.5"
        dispatcher = PiToolDispatcher(service)
        first = await dispatcher.execute(
            _request(
                "rosclaw_retry_work",
                mission=mission.mission_id,
                idem="idem_142_dup1",
                arguments={"work_order_id": root},
            )
        )
        assert first.ok, first.summary
        second = await dispatcher.execute(
            _request(
                "rosclaw_retry_work",
                mission=mission.mission_id,
                idem="idem_142_dup2",
                arguments={"work_order_id": root},
            )
        )
        assert second.ok, second.summary
        view = service._worker_manager.job_view(root)
        assert len(view["attempts"]) == 2, "重复 retry 创建了第三个 attempt"
        retry_id = view["attempts"][1]["attempt_id"]
        assert retry_id in first.summary and retry_id in second.summary
        await service._worker_manager.cancel_order(retry_id, reason="test cleanup")
        await service.close()

    async def test_concurrent_request_retry_cas(self, tmp_path: Path, monkeypatch) -> None:
        """coordinator CAS：auto+manual 并发 request_retry 只创建一个。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-fatal2",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "FLAG=$(dirname \"$0\")/cas-ok\n"
            "if [ -f \"$FLAG\" ]; then\n"
            "  while true; do echo '{\"kind\":\"liveness\"}'; sleep 0.5; done\n"
            "fi\n"
            "touch \"$FLAG\"\n"
            'echo \'{"kind":"attempt_failed","error_code":"PROVIDER_TIMEOUT",'
            '"message":"429"}\'\n'
            "for d in $(dirname \"$0\")/work/wo_*; do\n"
            '  echo \'{"cause":"PROVIDER_TRANSIENT","detail":"429"}\''
            ' > "$d/termination.json"\n'
            "done\n"
            "exit 1\n",
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        root = scheduled.work_order_id
        result, _ = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        order = service._worker_manager.order(root)
        coordinator = service._retry_coordinator
        # 真并发：auto + manual 同时请求。
        r_auto, r_manual = await asyncio.gather(
            coordinator.request_retry(order, cause="PROVIDER_TRANSIENT", actor="auto"),
            coordinator.request_retry(order, cause="PROVIDER_TRANSIENT",
                                      actor="native_agent"),
        )
        created = [r for r in (r_auto, r_manual) if r[1]]
        assert len(created) == 1, f"并发创建 {len(created)} 个 attempt——CAS 失效"
        assert r_auto[0].work_order_id == r_manual[0].work_order_id
        view = service._worker_manager.job_view(root)
        assert len(view["attempts"]) == 2
        retry_id = view["attempts"][1]["attempt_id"]
        await service._worker_manager.cancel_order(retry_id, reason="test cleanup")
        await service.close()


class TestRetryPolicy:
    def test_cause_policy(self) -> None:
        """§3.5 白名单：只有结构化 transient/crash 可自动重试。"""
        assert should_auto_retry("PROVIDER_TRANSIENT")
        assert should_auto_retry("WORKER_CRASH")
        assert should_auto_retry("EVENT_PIPE_BROKEN")
        for cause in (
            "USER_CANCELLED", "USER_PAUSED", "BUDGET_HARD_PAUSED",
            "DELIVERABLE_REJECTED", "TOOL_FAILED", "PROVIDER_FATAL",
            "COMPLETED", "SIGNAL_UNKNOWN", "AGENTD_SHUTDOWN",
        ):
            assert not should_auto_retry(cause), cause

    def test_parse_cause_structured_and_legacy(self) -> None:
        assert parse_cause("worker attempt failed [PROVIDER_FATAL]: 401") == (
            "PROVIDER_FATAL"
        )
        assert parse_cause("AdapterError: worker startup timeout") == "WORKER_CRASH"
        assert parse_cause("AdapterError: PROVIDER_TIMEOUT turn") == "PROVIDER_TRANSIENT"
        assert parse_cause("some other failure") is None

    def test_worker_exited_not_a_retry_marker(self) -> None:
        """'worker exited' 永不再是自动重试依据（§3.5：130 可能是取消/
        暂停/信号/重启）。"""
        from rosclaw.agentd import service as svc

        assert not hasattr(svc.AgentService, "_INFRA_FAILURE_MARKERS") or (
            "worker exited"
            not in getattr(svc.AgentService, "_INFRA_FAILURE_MARKERS", ())
        )
        # 裸 exit 摘要不带结构化 cause 时 parse 结果必须不含可重试项。
        assert parse_cause("worker exited 130 without a final report") is None
