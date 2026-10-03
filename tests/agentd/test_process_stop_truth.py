"""Stopping a terminal operation must report its durable state truthfully."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _task
from tests.agentd.test_pi_tool_bridge import _request


@pytest.mark.parametrize("exit_code,terminal", [(0, "SUCCEEDED"), (17, "FAILED")])
async def test_completed_process_stop_preserves_ledger_and_reports_already_terminal(
    tmp_path, exit_code, terminal
):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)
    service = SimpleNamespace(_store=SimpleNamespace(connection=conn), _operation_manager=manager)
    dispatcher = PiToolDispatcher(service)
    try:
        operation = await manager.start(
            task_id="task_1", attempt_id="", kind="process", argv=["sh", "-c", f"exit {exit_code}"]
        )
        operation_id = operation["operation_id"]
        for _ in range(100):
            if manager.get(operation_id)["state"] == terminal:
                break
            await asyncio.sleep(0.01)
        await manager.wait(operation_id)
        before = manager.get(operation_id)
        assert before["state"] == terminal
        events_before = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        result = await dispatcher._process_stop(
            _request("rosclaw_process_stop", arguments={"operation_id": operation_id})
        )
        assert result.ok
        assert result.status == terminal
        assert "无需取消" in result.summary
        assert "已取消（账本先行）" not in result.summary
        assert manager.get(operation_id) == before
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events_before
    finally:
        for driver in list(manager._drivers.values()):
            if not driver.done():
                driver.cancel()
        await asyncio.gather(*manager._drivers.values(), return_exceptions=True)
        for process in list(manager._procs.values()):
            if process.returncode is None:
                process.kill()
            await process.wait()
        conn.close()


@pytest.mark.parametrize(
    "initial,after",
    [
        ("RUNNING", "SUCCEEDED"),
        ("RUNNING", "CANCELING"),
        ("CANCELLED", "CANCELLED"),
        ("LOST", "LOST"),
        ("RUNNING", "RUNNING"),
    ],
)
async def test_stop_uses_actual_post_request_state_or_existing_terminal(tmp_path, initial, after):
    conn = _conn(tmp_path)
    state, calls = initial, []

    def get(operation_id):
        return {"operation_id": operation_id, "state": state}

    async def cancel(operation_id, *, reason):
        nonlocal state
        calls.append(operation_id)
        state = after

    service = SimpleNamespace(
        _store=SimpleNamespace(connection=conn),
        _operation_manager=SimpleNamespace(get=get, cancel=cancel),
    )
    try:
        result = await PiToolDispatcher(service)._process_stop(
            _request("rosclaw_process_stop", arguments={"operation_id": "fixture"})
        )
        assert result.status == after
        assert "已取消（账本先行）" not in result.summary
        assert bool(calls) == (initial == "RUNNING")
        assert result.ok == (after != "RUNNING")
        if after == "CANCELING":
            assert "等待终态确认" in result.summary
    finally:
        conn.close()
