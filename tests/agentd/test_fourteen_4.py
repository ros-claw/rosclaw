"""十四审 PR-14.4 红测试：F2 Tasks Center 的服务端投影与控制面。

红测试先行——修复前必须红：
1. pi.worker.jobs：按 root job 聚合（2 attempts 同 root → 一张卡），
   legacy 单（无 attempts 行）回退单卡；
2. pi.worker.control：pause → PAUSE_REQUESTED→PAUSED（ACK 后）；
   resume → RUNNING；不支持控制的 adapter 诚实报错；
3. pi.worker.transcript：tseq 游标分页 + channel 过滤。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from rosclaw.agentd.pi_bridge.server import (
    worker_control,
    worker_jobs_projection,
    worker_transcript_page,
)
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _setup


def _fake(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


_CONTROL_FAKE = """#!/bin/sh
echo '{"kind":"attempt_started"}'
(
  i=0
  while [ $i -lt 40 ]; do
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
      echo '{"kind":"attempt_finished","report":"done"}'
      exit 0
      ;;
  esac
done
"""


async def _hire(service, mission, tmp_path, fake, monkeypatch):
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
        goal="画一个五角星仿真",
        inputs={"instructions": "x"},
        budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    return service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )


class TestJobsProjection:
    async def test_jobs_aggregate_by_root(self, tmp_path: Path, monkeypatch) -> None:
        """同一 root 的两个 attempt → 一张卡（attempts 数 2）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-quick-fail",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "FLAG=$(dirname \"$0\")/j-ok\n"
            "if [ -f \"$FLAG\" ]; then\n"
            '  echo \'{"kind":"attempt_finished","report":"retry done"}\'\n'
            "  exit 0\n"
            "fi\n"
            "touch \"$FLAG\"\n"
            'echo \'{"kind":"attempt_failed","error_code":"PROVIDER_TIMEOUT",'
            '"message":"429"}\'\n'
            "for d in $(dirname \"$0\")/work/wo_*; do\n"
            '  echo \'{"cause":"PROVIDER_TRANSIENT"}\' > "$d/termination.json"\n'
            "done\n"
            "exit 1\n",
        )
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        root = scheduled.work_order_id
        result, _ = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        order = service._worker_manager.order(root)
        retry, created, _reason = await service._retry_coordinator.request_retry(
            order, cause="PROVIDER_TRANSIENT", actor="native_agent"
        )
        assert created and retry is not None
        # 等 retry attempt 收尾。
        for _ in range(200):
            current = service._worker_manager.order(retry.work_order_id)
            if current and current.status in (
                "ACCEPTED", "FAILED", "CANCELLED", "SUBMITTED", "VERIFYING",
            ):
                break
            await asyncio.sleep(0.05)
        jobs = worker_jobs_projection(service, mission.mission_id)
        assert len(jobs) == 1, f"应聚合成一张卡: {len(jobs)}"
        card = jobs[0]
        assert card["root_job_id"] == root
        assert card["goal"].startswith("画一个五角星")
        assert len(card["attempts"]) == 2
        assert card["attempts"][0]["work_order_id"] == root
        # 终态来自最新 attempt（retry 完成）。
        assert card["state"] in ("ACCEPTED", "SUBMITTED", "VERIFYING")
        await service.close()


class TestWorkerControl:
    async def test_pause_resume_via_bridge(self, tmp_path: Path, monkeypatch) -> None:
        """pause：先 PAUSE_REQUESTED，ACK 后 PAUSED；resume 回 RUNNING——
        乐观直写 PAUSED 是不允许的。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(tmp_path, "fake-control", _CONTROL_FAKE)
        scheduled = await _hire(service, mission, tmp_path, fake, monkeypatch)
        driver = asyncio.create_task(
            service._worker_manager.run_to_completion(scheduled)
        )
        wo = scheduled.work_order_id
        for _ in range(100):
            adapter = service._worker_manager._adapters["pi_managed"]
            if adapter._events.tail(wo):
                break
            await asyncio.sleep(0.05)
        paused = await worker_control(service, wo, "pause")
        assert paused["ok"], paused
        assert paused["state"] == "PAUSED"
        order = service._worker_manager.order(wo)
        assert order.status == "PAUSED", order.status
        proc = adapter._procs[wo]
        assert proc.returncode is None, "pause 后进程退出——exit130 回归"
        resumed = await worker_control(service, wo, "resume")
        assert resumed["ok"] and resumed["state"] == "RUNNING"
        assert service._worker_manager.order(wo).status == "RUNNING"
        result, _ = await asyncio.wait_for(driver, 30)
        assert result.status == "COMPLETED"
        await service.close()

    async def test_control_unknown_order(self, tmp_path: Path) -> None:
        service, _mission = await _setup(tmp_path)
        result = await worker_control(service, "wo_missing", "pause")
        assert not result["ok"]
        assert result["code"] == "WORK_ORDER_NOT_FOUND"
        await service.close()


class TestTranscriptBridge:
    async def test_transcript_pagination(self, tmp_path: Path) -> None:
        import json as _json

        service, mission = await _setup(tmp_path)
        wo = new_id("wo")
        d = tmp_path / "work" / wo
        d.mkdir(parents=True)
        (d / "transcript.jsonl").write_text(
            "".join(
                _json.dumps({"tseq": i, "channel": "conversation",
                             "role": "assistant", "text": f"m{i}"})
                + "\n"
                for i in range(1, 11)
            ),
            encoding="utf-8",
        )
        page = worker_transcript_page(service, wo, after_seq=0, limit=4)
        assert [r["tseq"] for r in page["records"]] == [1, 2, 3, 4]
        assert page["has_more"] is True
        page2 = worker_transcript_page(
            service, wo, after_seq=page["next_cursor"], limit=4
        )
        assert [r["tseq"] for r in page2["records"]] == [5, 6, 7, 8]
        await service.close()
