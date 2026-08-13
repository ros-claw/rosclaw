"""十一审 PR-B 红测试：WorkerEventStore 持久化 + tail/subscribe + stderr 脱敏。

红测试先行——修复前必须红：
1. 事件边运行边落 events.jsonl（含 liveness），seq 连续；
2. stderr 落盘且 sk-/api_key 被脱敏；
3. state.json 终态可读；
4. tail(after_seq) 增量读取（subscribe 语义）；
5. 新实例（重启语义）仍能读到全部事件（文件权威）；
6. 大文本字段截断为 preview。
"""

from __future__ import annotations

import stat
from pathlib import Path

from rosclaw.agentd.workers.event_store import WorkerEventStore, redact
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _setup


class TestEventStoreUnit:
    def test_redact_secret_patterns(self) -> None:
        assert "sk-" + "x" * 20 not in redact("key is sk-" + "x" * 20)
        assert "REDACTED" in redact("api_key: abcdefgh12345678")
        assert redact("normal text") == "normal text"

    def test_append_and_tail_incremental(self, tmp_path: Path) -> None:
        store = WorkerEventStore(tmp_path)
        store.append_event("wo_abcdef01", "att_1", "attempt_started", {"worker": "pi"})
        store.append_event("wo_abcdef01", "att_1", "tool_started", {"tool": "read"})
        store.append_event("wo_abcdef01", "att_1", "liveness", {"phase": "RUNNING_TOOL"})
        all_events = store.tail("wo_abcdef01")
        assert [e["kind"] for e in all_events] == [
            "attempt_started",
            "tool_started",
            "liveness",
        ]
        assert [e["seq"] for e in all_events] == [1, 2, 3]
        assert all(e["attempt_id"] == "att_1" for e in all_events)
        # 增量（subscribe 语义）。
        fresh = store.tail("wo_abcdef01", after_seq=2)
        assert len(fresh) == 1 and fresh[0]["kind"] == "liveness"
        # 新实例（重启语义）读同一账本。
        store2 = WorkerEventStore(tmp_path)
        assert len(store2.tail("wo_abcdef01")) == 3
        store2.append_event("wo_abcdef01", "att_1", "attempt_finished", {"report": "done"})
        assert store2.tail("wo_abcdef01")[-1]["seq"] == 4

    def test_stderr_redacted_and_large_text_preview(self, tmp_path: Path) -> None:
        store = WorkerEventStore(tmp_path)
        store.append_stderr("wo_abcdef02", "error: key sk-" + "k" * 30 + " boom\n")
        content = store.tail_stderr("wo_abcdef02")
        assert "sk-" + "k" * 30 not in content
        assert "REDACTED" in content
        store.append_event("wo_abcdef02", "", "big", {"text": "x" * 5000})
        event = store.tail("wo_abcdef02")[-1]
        assert "truncated" in event["text"]
        assert len(event["text"]) < 2200

    def test_state_roundtrip(self, tmp_path: Path) -> None:
        store = WorkerEventStore(tmp_path)
        assert store.read_state("wo_abcdef03") is None
        store.write_state("wo_abcdef03", {"status": "RUNNING", "phase": "RUNNING_TOOL"})
        state = store.read_state("wo_abcdef03")
        assert state is not None and state["status"] == "RUNNING"
        assert "updated_at" in state


class TestAdapterPersists:
    async def test_run_persists_events_stderr_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = tmp_path / "fake-entry"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\' \n'
            "echo 'warning: something odd' >&2\n"
            'echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"完成"}\'\n'
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        service._worker_manager._adapters["pi_managed"] = adapter
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
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        result, report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED"
        store = WorkerEventStore(tmp_path)
        events = store.tail(scheduled.work_order_id)
        kinds = [e["kind"] for e in events]
        assert "attempt_started" in kinds
        assert "liveness" in kinds
        assert "attempt_finished" in kinds
        assert "something odd" in store.tail_stderr(scheduled.work_order_id)
        state = store.read_state(scheduled.work_order_id)
        assert state is not None and state["status"] == "COMPLETED"
        assert report.accepted
        await service.close()
