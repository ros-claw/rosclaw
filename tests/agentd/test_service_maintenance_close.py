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
        assert ordering == ["maintenance_finally_db_open", "store_close", "store_close"]
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
        assert closed == [True, True]
        assert service._op_maintenance_task is None
    finally:
        release_cleanup.set()
        for item in (first, second, task):
            if not item.done():
                item.cancel()
        await asyncio.gather(first, second, task, return_exceptions=True)
        original_close()
