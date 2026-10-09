"""Repeated cancellation cannot acknowledge another caller's pending cleanup."""

import asyncio
import contextlib
import os
import signal
import sys
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _events, _task
from tests.agentd.test_p1b3_ros2_action import ABORTED, CANCELED, SUCCEEDED, FakeActionClient
from tests.agentd.test_pi_tool_bridge import _request


@pytest.mark.parametrize("wrapper_exited", [False, True])
def test_repeated_cancel_preserves_pending_cleanup_and_first_reason(tmp_path, wrapper_exited):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)

    async def run():
        operation = await manager.start(
            task_id="task_1",
            attempt_id="",
            kind="process",
            argv=[
                sys.executable,
                "-c",
                ("import os; os.fork() and os._exit(0); " if wrapper_exited else "")
                + "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "print('concurrent-cancel-ready',flush=True); time.sleep(30)",
            ],
        )
        operation_id = operation["operation_id"]
        process = manager._procs[operation_id]
        first = None
        try:
            async with asyncio.timeout(5):
                while not any(
                    event["type"] == "operation.output"
                    and "concurrent-cancel-ready" in event["payload"].get("text", "")
                    for event in _events(conn)
                ):
                    await asyncio.sleep(0.01)
                if wrapper_exited:
                    while process.returncode is None:
                        await asyncio.sleep(0.01)
                first = asyncio.create_task(manager.cancel(operation_id, reason="first-owner"))
                while manager.get(operation_id)["state"] != "CANCELING":
                    await asyncio.sleep(0.01)
                assert not first.done()
                assert process.stdout is not None and not process.stdout.at_eof()
                await manager.cancel(operation_id, reason="second-caller")
                assert manager.get(operation_id)["state"] == "CANCELING"
                assert manager.get(operation_id)["cancel_reason"] == "first-owner"
                assert not any(event["type"] == "operation.cancelled" for event in _events(conn))
        finally:
            if first is not None:
                await asyncio.wait_for(first, timeout=7)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await manager.wait(operation_id, timeout=5)
        assert process.stdout is not None and process.stdout.at_eof()
        assert manager.get(operation_id)["state"] == "CANCELLED"
        assert manager.get(operation_id)["cancel_reason"] == "first-owner"

    asyncio.run(run())


@pytest.mark.parametrize("result_code,state", [(SUCCEEDED, "SUCCEEDED"), (ABORTED, "FAILED")])
def test_stop_tool_preserves_an_existing_terminal_result(tmp_path, result_code, state):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)
    client = FakeActionClient()
    dispatcher = PiToolDispatcher(
        SimpleNamespace(_operation_manager=manager, _store=SimpleNamespace(connection=conn))
    )

    async def run():
        operation = await manager.start_action(
            task_id="task_1",
            attempt_id="",
            action="/fake/action",
            action_type="test/action/Fake",
            args={},
            client=client,
        )
        client.emit_result(result_code, {})
        await asyncio.sleep(0)
        result = await dispatcher._process_stop(
            _request("rosclaw_process_stop", arguments={"operation_id": operation["operation_id"]})
        )
        assert result.status == state
        assert "已取消" not in result.summary
        assert client.cancelled == []
        assert manager.get(operation["operation_id"])["state"] == state

    asyncio.run(run())


def test_repeated_action_cancel_keeps_single_grace_and_original_intent(tmp_path):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)
    client = FakeActionClient()

    async def run():
        operation = await manager.start_action(
            task_id="task_1",
            attempt_id="",
            action="/fake/action",
            action_type="test/action/Fake",
            args={},
            client=client,
        )
        operation_id = operation["operation_id"]
        await manager.cancel(operation_id, reason="first-owner")
        grace = manager._drivers[operation_id]
        try:
            await manager.cancel(operation_id, reason="second-caller")
            assert manager._drivers[operation_id] is grace
            assert client.cancelled == [operation["goal_id"]]
            assert manager.get(operation_id)["state"] == "CANCELING"
            assert manager.get(operation_id)["cancel_reason"] == "first-owner"
            client.emit_result(CANCELED, {})
            # Action results marshal onto the running manager event loop.
            await asyncio.sleep(0)
            assert manager.get(operation_id)["state"] == "CANCELLED"
        finally:
            grace.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await grace

    asyncio.run(run())


@pytest.mark.parametrize("tool", ["rosclaw_process_stop", "rosclaw_stop_operation"])
def test_stop_tool_projects_pending_ledger_instead_of_claiming_cancelled(tmp_path, tool):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)
    client = FakeActionClient()
    # Exercise both real stop routes. Gateway admission is tested separately.
    dispatcher = PiToolDispatcher(
        SimpleNamespace(_operation_manager=manager, _store=SimpleNamespace(connection=conn))
    )

    async def run():
        operation = await manager.start_action(
            task_id="task_1",
            attempt_id="",
            action="/fake/action",
            action_type="test/action/Fake",
            args={},
            client=client,
        )
        operation_id = operation["operation_id"]
        try:
            result = await dispatcher._dispatch(
                _request(tool, arguments={"operation_id": operation_id})
            )
            assert manager.get(operation_id)["state"] == "CANCELING"
            assert result.status == "CANCELING"
            assert "已取消" not in result.summary
            client.emit_result(CANCELED, {})
            await asyncio.sleep(0)
            result = await dispatcher._dispatch(
                _request(tool, arguments={"operation_id": operation_id})
            )
            assert result.status == "CANCELLED"
            assert "已取消" in result.summary
        finally:
            grace = manager._drivers[operation_id]
            grace.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await grace

    asyncio.run(run())
