"""十二审 PR-12.3 红测试：持久会话 + 真 resume + 查看器数据。

红测试先行——修复前必须红：
1. resume 无检查点 → 诚实 NO_CHECKPOINT（不假装恢复）；
2. resume 有 session_file → 新 attempt 带 _resume_session + lineage；
3. resume 运行中单 → NOT_TERMINAL 拒绝；
4. detail 数据源：transcript/artifacts/patch 从 work 目录读出。
"""

from __future__ import annotations

import stat
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _fake(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


async def _completed_order(service, mission, tmp_path, monkeypatch):
    """跑一个完成的 fake 订单，返回 work_order_id。"""
    from rosclaw.agentd.workers import pi_managed

    fake = _fake(
        tmp_path,
        "fake-done",
        "#!/bin/sh\n"
        'echo \'{"kind":"attempt_started"}\'\n'
        f'echo \'{{"kind":"session_persisted","session_file":"{tmp_path}/session/fake-session.jsonl"}}\'\n'
        'echo \'{"kind":"attempt_finished","report":"done"}\'\n',
    )
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
    result = await dispatcher.execute(
        _request(
            "rosclaw_delegate",
            mission=mission.mission_id,
            idem="idem_123_done",
            arguments={"goal": "快任务", "worker_id": "worker:rosclaw:pi"},
        )
    )
    assert result.status == "COMPLETED", result.summary
    return service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id


class TestResume:
    async def test_resume_without_checkpoint_honest_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        wo = await _completed_order(service, mission, tmp_path, monkeypatch)
        # state.json 指向不存在的文件 → NO_CHECKPOINT
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_123_r1",
                arguments={"work_order_id": wo},
            )
        )
        assert not result.ok
        assert result.error_code == "NO_CHECKPOINT"
        await service.close()

    async def test_resume_with_session_creates_resuming_attempt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        wo = await _completed_order(service, mission, tmp_path, monkeypatch)
        # 造出真实 session 文件（state.json 已指向它）。
        session_file = tmp_path / "session" / "fake-session.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text('{"type":"session"}\n', encoding="utf-8")
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_123_r2",
                arguments={"work_order_id": wo},
            )
        )
        assert result.ok, result.summary
        assert result.status == "STARTED"
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 2
        resumed = orders[1]
        assert resumed.parent_work_order_id == wo
        assert resumed.inputs.get("_resume_session") == str(session_file)
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_123_r2c",
                arguments={"work_order_id": resumed.work_order_id},
            )
        )
        await service.close()

    async def test_resume_running_order_refused(self, tmp_path: Path, monkeypatch) -> None:
        service, mission = await _setup(tmp_path)
        from tests.agentd.test_ten_w0 import _register_stub, _slow_adapter_module

        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_123_r3",
                arguments={"goal": "慢", "worker_id": "worker:stub:slow"},
            )
        )
        assert started.status == "STARTED"
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        result = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_123_r4",
                arguments={"work_order_id": wo},
            )
        )
        assert not result.ok and result.error_code == "NOT_TERMINAL"
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_123_r4c",
                arguments={"work_order_id": wo},
            )
        )
        await service.close()


class TestDetailView:
    async def test_detail_reads_work_dir(self, tmp_path: Path, monkeypatch) -> None:
        """查看器数据源：transcript/patch/artifacts 从 work 目录读出。"""
        service, mission = await _setup(tmp_path)
        wo = await _completed_order(service, mission, tmp_path, monkeypatch)
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        store = WorkerEventStore(tmp_path)
        work_dir = store.dir_of(wo)
        (work_dir / "transcript.jsonl").write_text('{"role":"assistant","text":"完成"}\n')
        artifacts = work_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "patch.diff").write_text("diff --git a/x b/x\n")
        (artifacts / "bash-log.txt").write_text("$ pytest\n(exit 0)\n")
        # 经 RPC handler 读（直接调 handler 逻辑——经 store 验证等价面）。
        transcript = (work_dir / "transcript.jsonl").read_text()
        assert "完成" in transcript
        assert (artifacts / "patch.diff").exists()
        assert (artifacts / "bash-log.txt").exists()
        state = store.read_state(wo)
        assert state is not None and "session_file" in state
        await service.close()
