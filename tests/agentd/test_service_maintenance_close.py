"""Maintenance cancellation must finish while its SQLite store remains open.

This does not assert operation-driver/process detach or global cancellation.
"""

from __future__ import annotations

import asyncio

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.service import AgentService


async def test_close_awaits_maintenance_finally_before_database_close(tmp_path):
    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    conn = service._store.connection
    started = asyncio.Event()
    finalized = asyncio.Event()
    ordering = []

    async def maintenance():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)  # cancellation cleanup is a real await boundary
            conn.execute("SELECT 1").fetchone()
            ordering.append("maintenance_finally_db_open")
            finalized.set()

    task = asyncio.create_task(maintenance())
    service._op_maintenance_task = task
    original_close = service._store.close

    def close_store():
        ordering.append("store_close")
        original_close()

    service._store.close = close_store
    await started.wait()
    try:
        await service.close()
        assert task.done(), "close returned before maintenance cancellation settled"
        assert finalized.is_set(), "maintenance finally did not finish with DB open"
        assert ordering == ["maintenance_finally_db_open", "store_close"]
        assert service._op_maintenance_task is None
        # Production lifespan and CLI finally both invoke close.
        await service.close()
        assert ordering == ["maintenance_finally_db_open", "store_close"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        original_close()


async def test_concurrent_repeated_close_waits_for_same_maintenance(tmp_path):
    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    conn = service._store.connection
    started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_started = asyncio.Event()
    closed = []

    async def maintenance():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            conn.execute("SELECT 1").fetchone()

    task = asyncio.create_task(maintenance())
    service._op_maintenance_task = task
    original_close = service._store.close

    def close_store():
        closed.append(task.done())
        original_close()

    service._store.close = close_store
    await started.wait()
    first = asyncio.create_task(service.close())
    await cleanup_started.wait()
    second = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    try:
        assert not closed, "a concurrent close skipped pending maintenance cleanup"
        release_cleanup.set()
        await asyncio.gather(first, second)
        assert closed == [True]
        assert service._op_maintenance_task is None
    finally:
        release_cleanup.set()
        for item in (first, second, task):
            if not item.done():
                item.cancel()
        await asyncio.gather(first, second, task, return_exceptions=True)
        original_close()


async def test_cancelled_close_caller_does_not_cancel_shared_cleanup(tmp_path):
    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    cleanup_started = asyncio.Event()
    release = asyncio.Event()

    async def maintenance():
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release.wait()
            service._store.connection.execute("SELECT 1")

    service._op_maintenance_task = asyncio.create_task(maintenance())
    await asyncio.sleep(0)
    first = asyncio.create_task(service.close())
    await cleanup_started.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert not service._close_task.cancelled()
    second = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    await second
    assert service._close_task.done()
    await service.close()


async def test_real_service_close_preserves_producer_and_reopen_output(tmp_path):
    import sys

    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    manager = service._operation_manager
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[
            sys.executable,
            "-c",
            'import time,os,sys; time.sleep(.2); os.write(1,"中🎾TAIL".encode());sys.exit(7)',
        ],
    )
    proc = manager._procs[op["operation_id"]]
    await asyncio.gather(service.close(), service.close())
    assert proc.returncode is None
    reopened = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    try:
        recovery = await reopened._operation_manager.recover_on_boot()
        assert recovery["reattached"] == 1
        result = await reopened._operation_manager.wait(op["operation_id"], timeout=5)
        assert result["state"] == "FAILED" and result["failure_code"] == "exit_7"
        outputs = [
            e["payload"]["text"]
            for e in reopened._operation_manager.events_since("t", 0)
            if e["event_type"] == "operation.output"
        ]
        assert "".join(outputs) == "中🎾TAIL"
    finally:
        await proc.wait()
        await reopened.close()


async def test_unresolved_handoff_preflight_leaves_maintenance_and_store_retryable(tmp_path):
    from rosclaw.connectors.ros.action_client import STATUS_CANCELED
    from rosclaw.task_kernel.operation_manager import OperationHandoffUnresolvedError

    service = AgentService(load_agent_config(tmp_path / "config.yaml"), tmp_path)
    conn = service._store.connection
    maintenance = asyncio.create_task(asyncio.Event().wait())
    service._op_maintenance_task = maintenance

    class MockClient:
        def send_goal(self, **kwargs):
            self.callbacks = kwargs

    client = MockClient()
    await service._operation_manager.start_action(
        task_id="t",
        attempt_id="",
        action="/private-mock",
        action_type="FixtureAction",
        args={},
        client=client,
    )
    try:
        try:
            await service.close()
        except OperationHandoffUnresolvedError:
            pass
        else:
            raise AssertionError("active Action was incorrectly handed off")
        assert not maintenance.cancelling()
        assert service._op_maintenance_task is maintenance
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        client.callbacks["on_result"](STATUS_CANCELED, {})
        await asyncio.sleep(0)
        await service.close()
        assert maintenance.done()
    finally:
        if not maintenance.done():
            maintenance.cancel()
        await asyncio.gather(maintenance, return_exceptions=True)
