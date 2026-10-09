"""Dropping a cancellation request cannot abandon its owned process cleanup."""

import asyncio
import contextlib
import os
import signal
import sys

import pytest

from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _events, _task


@pytest.mark.parametrize("wrapper_exited", [False, True])
def test_cancel_request_disconnection_keeps_cleanup_owned(tmp_path, wrapper_exited):
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
                "print('cancel-owner-ready',flush=True); time.sleep(30)",
            ],
        )
        operation_id = operation["operation_id"]
        process = manager._procs[operation_id]
        request = None
        try:
            async with asyncio.timeout(5):
                while not any(
                    e["type"] == "operation.output"
                    and "cancel-owner-ready" in e["payload"].get("text", "")
                    for e in _events(conn)
                ):
                    await asyncio.sleep(0.01)
                if wrapper_exited:
                    while process.returncode is None:
                        await asyncio.sleep(0.01)
                request = asyncio.create_task(
                    manager.cancel(operation_id, reason="original-request")
                )
                while manager.get(operation_id)["state"] != "CANCELING":
                    await asyncio.sleep(0.01)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                await manager.cancel(operation_id, reason="later-request")
                assert manager.get(operation_id)["cancel_reason"] == "original-request"
                assert manager.get(operation_id)["state"] == "CANCELING"
                assert not any(e["type"] == "operation.cancelled" for e in _events(conn))
                with pytest.raises(TimeoutError):
                    await manager.wait(operation_id, timeout=0.01)
                assert manager.get(operation_id)["state"] == "CANCELING"
            result = await manager.wait(operation_id, timeout=7)
            assert result["state"] == "CANCELLED"
            assert process.stdout is not None and process.stdout.at_eof()
            assert len([e for e in _events(conn) if e["type"] == "operation.cancelled"]) == 1
        finally:
            if request is not None and not request.done():
                request.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await request
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await manager.wait(operation_id, timeout=5)

    asyncio.run(run())
