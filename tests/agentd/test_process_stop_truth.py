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
        owner = {key: before[key] for key in ("operation_id", "task_id", "revision", "state")}
        status = await dispatcher._process_status(
            _request("rosclaw_process_status", arguments={"operation_id": operation_id})
        )
        assert status.ok
        assert status.status == status.operation["state"] == terminal
        assert {key: status.operation[key] for key in owner} == owner
        assert set(status.operation) == {*owner, "failure_code", "observation"}
        assert status.operation["failure_code"] == before["failure_code"]
        assert status.operation["failure_code"] == (f"exit_{exit_code}" if exit_code else "")
        observation = status.operation["observation"]
        assert isinstance(observation, dict)
        assert set(observation) == {
            "observed_at",
            "producer_freshness",
            "process_identity",
            "warnings",
            "limits",
        }
        assert isinstance(observation["observed_at"], str)
        assert observation["observed_at"].endswith("+00:00")
        assert observation["observed_at"] in status.summary
        freshness = observation["producer_freshness"]
        assert isinstance(freshness, dict)
        assert set(freshness) == {"heartbeat_at", "age_s", "meaning"}
        assert freshness["heartbeat_at"] == before["heartbeat_at"]
        assert freshness["age_s"] is None or (
            type(freshness["age_s"]) in (int, float) and freshness["age_s"] >= 0
        )
        assert freshness["meaning"] == (
            "Recorded producer freshness only; not work or physics progress."
        )
        # The child has exited; a successful read is not live identity evidence.
        assert observation["process_identity"]["status"] == "UNKNOWN"
        assert observation["process_identity"]["reason"] in {
            "STORED_IDENTITY_INVALID_OR_ABSENT",
            "CURRENT_IDENTITY_UNAVAILABLE",
            "CURRENT_IDENTITY_MISMATCH",
        }
        assert observation["process_identity"]["reason"] in status.summary
        assert observation["warnings"] == []
        assert observation["limits"] == (
            "Instantaneous identity only; UNKNOWN does not prove absence. "
            "No progress, readiness, safety or stop confirmation; ledger unchanged."
        )
        assert observation["limits"] in status.summary
        read = await dispatcher._process_output(
            _request("rosclaw_process_output", arguments={"operation_id": operation_id})
        )
        assert read.operation == owner
        assert read.status == terminal
        result = await dispatcher._process_stop(
            _request("rosclaw_process_stop", arguments={"operation_id": operation_id})
        )
        assert result.ok
        assert result.status == terminal
        assert result.operation == read.operation
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
