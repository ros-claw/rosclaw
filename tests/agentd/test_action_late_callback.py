"""Production Action marshal boundaries; mock listener, no ROS/DDS execution."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from rosclaw.connectors.ros.action_client import STATUS_SUCCEEDED
from rosclaw.task_kernel.operation_manager import OperationHandoffUnresolvedError, OperationManager
from tests.agentd.test_operation_durable_handoff import database


class Listener:
    def send_goal(self, **kwargs):
        self.callbacks = kwargs


def snapshot(conn, oid):
    return (
        tuple(conn.execute("SELECT * FROM operations WHERE operation_id=?", (oid,)).fetchone()),
        [tuple(row) for row in conn.execute("SELECT * FROM task_events ORDER BY seq")],
    )


async def completed(manager, client):
    op = await manager.start_action(
        task_id="t",
        attempt_id="",
        action="/pure-fixture",
        action_type="fixture/Action",
        args={},
        client=client,
    )
    client.callbacks["on_result"](STATUS_SUCCEEDED, {"result": "initial"})
    await asyncio.sleep(0)
    assert manager.get(op["operation_id"])["state"] == "SUCCEEDED"
    return op["operation_id"]


async def test_completed_action_callbacks_after_close_never_schedule_or_touch_store(
    tmp_path, monkeypatch
):
    path = tmp_path / "ledger.db"
    conn = database(path)
    manager = OperationManager(None, conn)
    client = Listener()
    oid = await completed(manager, client)
    before = snapshot(conn, oid)
    await manager.close()
    conn.close()
    loop = asyncio.get_running_loop()

    def forbidden(*args):
        raise AssertionError("closed manager scheduled callback")

    with monkeypatch.context() as patch:
        patch.setattr(loop, "call_soon_threadsafe", forbidden)
        client.callbacks["on_feedback"]({"late": True})
        client.callbacks["on_result"](STATUS_SUCCEEDED, {"result": "late"})
    conn = sqlite3.connect(path)
    assert snapshot(conn, oid) == before
    conn.close()


async def test_real_queued_callback_rechecks_close_before_apply(tmp_path):
    path = tmp_path / "ledger.db"
    conn = database(path)
    manager = OperationManager(None, conn)
    client = Listener()
    oid = await completed(manager, client)
    before = snapshot(conn, oid)
    loop = asyncio.get_running_loop()
    caught = []
    old = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: caught.append(context))
    try:
        client.callbacks["on_feedback"]({"queued_before_close": True})
        client.callbacks["on_result"](STATUS_SUCCEEDED, {"result": "queued"})
        # Actual close body has no suspended driver in this completed fixture:
        # finish it before yielding to the callbacks already queued on the loop.
        await manager._close_observers()
        conn.close()
        await asyncio.sleep(0)
        assert not caught
        conn = sqlite3.connect(path)
        assert snapshot(conn, oid) == before
        conn.close()
    finally:
        loop.set_exception_handler(old)


async def test_active_action_failed_preflight_preserves_normal_callback_updates(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    client = Listener()
    op = await manager.start_action(
        task_id="t",
        attempt_id="",
        action="/pure-fixture",
        action_type="fixture/Action",
        args={},
        client=client,
    )
    with pytest.raises(OperationHandoffUnresolvedError):
        await manager.close()
    assert not manager._closing
    client.callbacks["on_feedback"]({"normal": 1})
    await asyncio.sleep(0)
    assert "normal" in manager.get(op["operation_id"])["progress_json"]
    client.callbacks["on_result"](STATUS_SUCCEEDED, {"result": "normal"})
    await asyncio.sleep(0)
    assert manager.get(op["operation_id"])["state"] == "SUCCEEDED"
    await manager.close()
    conn.close()


def test_closed_loop_callback_never_falls_back_to_listener_thread_database_write(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    client = Listener()
    loop = asyncio.new_event_loop()
    op = loop.run_until_complete(
        manager.start_action(
            task_id="t",
            attempt_id="",
            action="/pure-fixture",
            action_type="fixture/Action",
            args={},
            client=client,
        )
    )
    before = snapshot(conn, op["operation_id"])
    loop.close()
    errors = []

    def late_listener():
        try:
            client.callbacks["on_feedback"]({"late": True})
            client.callbacks["on_result"](STATUS_SUCCEEDED, {"late": True})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=late_listener)
    thread.start()
    thread.join(1)
    assert not thread.is_alive() and not errors
    assert snapshot(conn, op["operation_id"]) == before
    assert manager.get(op["operation_id"])["state"] == "RUNNING"  # no false terminal/stop evidence
    conn.close()
