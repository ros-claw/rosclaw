"""Real detached host processes: recovery must not invent cancellation or ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _task
from tests.agentd.test_pi_tool_bridge import _request

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="Linux birth identity contract"
)


def birth(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rpartition(")")[2].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "uid": root.stat().st_uid,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cwd": os.readlink(root / "cwd"),
            "command_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest(),
        }
    except FileNotFoundError:
        return None


def detached(conn, request, command=None):
    script = """
import asyncio,json,os,sqlite3,sys
from rosclaw.task_kernel.operation_manager import OperationManager
conn=sqlite3.connect(sys.argv[1],check_same_thread=False);conn.row_factory=sqlite3.Row
manager=OperationManager(None,conn)
async def run():
    op=await manager.start(task_id='task_1',attempt_id='',kind='process',argv=json.loads(sys.argv[2]))
    conn.commit();os.write(1,json.dumps(op).encode());os._exit(0)
asyncio.run(run())
"""
    database = conn.execute("PRAGMA database_list").fetchone()[2]
    command = command or [sys.executable, "-c", "import time; time.sleep(30)"]
    result = subprocess.run(
        [sys.executable, "-c", script, database, json.dumps(command)],
        capture_output=True,
        check=True,
        timeout=10,
    )
    operation = json.loads(result.stdout)
    identity = birth(operation["pid"])
    assert identity and identity["pgid"] == identity["sid"] == identity["pid"]

    def cleanup():
        current = birth(identity["pid"])
        if (
            current
            and current["start_ticks"] == identity["start_ticks"]
            and current["boot_id"] == identity["boot_id"]
        ):
            os.killpg(identity["pid"], signal.SIGKILL)

    request.addfinalizer(cleanup)
    return operation, identity


async def test_cancel_recovered_process_waits_for_actual_stop(tmp_path, request):
    conn = _conn(tmp_path)
    _task(conn)
    operation, identity = detached(conn, request)
    manager = OperationManager(None, conn)
    try:
        recovery = await manager.recover_on_boot()
        assert recovery["reattached"] == 1
        await manager.cancel(operation["operation_id"], reason="user-request")
        assert manager.get(operation["operation_id"])["state"] == "CANCELLED"
        assert birth(identity["pid"]) is None, (
            "CANCELLED returned while the original process is still running"
        )
    finally:
        await manager.close()
        conn.close()


@pytest.mark.parametrize("outcome", ["live", "exitcode", "dead"])
async def test_restart_finishes_persisted_cancellation_instead_of_watching_forever(
    tmp_path, request, outcome
):
    conn = _conn(tmp_path)
    _task(conn)
    operation, identity = detached(conn, request)
    operation_id = operation["operation_id"]
    conn.execute(
        "UPDATE operations SET state='CANCELING',cancel_reason='original-stop' "
        "WHERE operation_id=?",
        (operation_id,),
    )
    if outcome != "live":
        Path(operation["exitcode_path"]).write_text("0")
    if outcome == "dead":
        os.killpg(identity["pid"], signal.SIGKILL)
        deadline = asyncio.get_running_loop().time() + 5
        while birth(identity["pid"]) is not None:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
    conn.commit()
    manager = OperationManager(None, conn)
    try:
        recovery = await manager.recover_on_boot()
        row = manager.get(operation_id)
        assert row["state"] == ("LOST" if outcome == "dead" else "CANCELLED")
        assert row["cancel_reason"] == "original-stop"
        assert birth(identity["pid"]) is None
        assert recovery["reattached"] == 0
        assert recovery["lost" if outcome == "dead" else "terminated"] == 1
        assert not manager._drivers
    finally:
        await manager.close()
        conn.close()


async def test_recovered_subgroup_cleanup_survives_caller_loss_and_shutdown(tmp_path, request):
    conn = _conn(tmp_path)
    _task(conn)
    pid_file = tmp_path / "resistant-child.pid"
    child_code = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],process_group=0);"
        "time.sleep(30)"
    )
    operation, leader = detached(conn, request, [sys.executable, "-c", parent_code])
    deadline = asyncio.get_running_loop().time() + 5
    while not pid_file.exists():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)
    child = birth(int(pid_file.read_text()))
    assert child and child["sid"] == leader["sid"] and child["pgid"] != leader["pgid"]

    def cleanup_child():
        current = birth(child["pid"])
        if current and current["start_ticks"] == child["start_ticks"]:
            os.killpg(child["pgid"], signal.SIGKILL)

    request.addfinalizer(cleanup_child)
    manager = OperationManager(None, conn)
    try:
        await manager.recover_on_boot()
        cancelling = asyncio.create_task(
            manager.cancel(operation["operation_id"], reason="transport-lost")
        )
        while manager.get(operation["operation_id"])["state"] != "CANCELING":
            await asyncio.sleep(0)
        cancelling.cancel()
        await asyncio.gather(cancelling, return_exceptions=True)
        # Join the same cancellation owner; closing alone never signals other
        # recovered operations that the user has not requested to cancel.
        await asyncio.wait_for(manager.close(), 12)
        row = manager.get(operation["operation_id"])
        assert row["state"] == "CANCELLED" and row["cancel_reason"] == "transport-lost"
        assert birth(leader["pid"]) is None and birth(child["pid"]) is None
        events = conn.execute(
            "SELECT event_type FROM task_events WHERE operation_id=?", (operation["operation_id"],)
        ).fetchall()
        assert sum(r[0] == "operation.canceling" for r in events) == 1
        assert sum(r[0] == "operation.cancelled" for r in events) == 1
    finally:
        await manager.close()
        conn.close()


@pytest.mark.parametrize("fault", ["missing", "wrong_birth", "wrong_boot"])
async def test_unproved_recovery_neither_adopts_nor_signals_live_pid(tmp_path, request, fault):
    conn = _conn(tmp_path)
    _task(conn)
    operation, identity = detached(conn, request)
    checkpoint = {
        "schema_version": "rosclaw.operation.process_checkpoint.v1",
        "process_identity": identity.copy(),
    }
    if fault == "missing":
        checkpoint = {}
    elif fault == "wrong_birth":
        checkpoint["process_identity"]["start_ticks"] += 1
    else:
        checkpoint["process_identity"]["boot_id"] = "different-boot"
    raw = json.dumps(checkpoint)
    conn.execute(
        "UPDATE operations SET checkpoint_json=? WHERE operation_id=?",
        (raw, operation["operation_id"]),
    )
    conn.commit()
    manager = OperationManager(None, conn)
    try:
        recovery = await manager.recover_on_boot()
        assert recovery["reattached"] == 0
        assert manager.get(operation["operation_id"])["state"] == "LOST"
        assert manager.get(operation["operation_id"])["checkpoint_json"] == raw
        assert birth(identity["pid"])["start_ticks"] == identity["start_ticks"]
    finally:
        await manager.close()
        conn.close()


async def test_cancel_without_proved_recovery_stays_unknown_and_does_not_signal(tmp_path, request):
    conn = _conn(tmp_path)
    _task(conn)
    operation, identity = detached(conn, request)
    conn.execute(
        "UPDATE operations SET checkpoint_json=NULL WHERE operation_id=?",
        (operation["operation_id"],),
    )
    conn.commit()
    manager = OperationManager(None, conn)
    try:
        await manager.cancel(operation["operation_id"], reason="operator-request")
        assert manager.get(operation["operation_id"])["state"] == "LOST"
        assert birth(identity["pid"])["start_ticks"] == identity["start_ticks"]
    finally:
        await manager.close()
        conn.close()


async def test_native_stop_reports_reconciliation_for_unproved_process(tmp_path, request):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    conn = _conn(tmp_path)
    _task(conn)
    operation, identity = detached(conn, request)
    conn.execute(
        "UPDATE operations SET checkpoint_json=NULL WHERE operation_id=?",
        (operation["operation_id"],),
    )
    conn.commit()
    manager = OperationManager(None, conn)
    service = SimpleNamespace(_store=SimpleNamespace(connection=conn), _operation_manager=manager)
    try:
        result = await PiToolDispatcher(service)._process_stop(
            _request("rosclaw_process_stop", arguments={"operation_id": operation["operation_id"]})
        )
        assert result.status == "LOST" and result.ok is False
        assert result.reconciliation_required is True
        assert result.error_code == "process_cancel_unconfirmed"
        assert result.details["operation_id"] == operation["operation_id"]
        assert birth(identity["pid"])["start_ticks"] == identity["start_ticks"]
    finally:
        await manager.close()
        conn.close()


@pytest.mark.parametrize("state", ["CANCELING", "CANCELLED", "LOST"])
async def test_native_start_does_not_invent_started_after_admission_cancel(tmp_path, state):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    conn = _conn(tmp_path)
    _task(conn)
    manager = SimpleNamespace(
        start=AsyncMock(return_value={"operation_id": "op_stable", "state": state})
    )
    service = SimpleNamespace(
        _store=SimpleNamespace(connection=conn),
        _operation_manager=manager,
        _task_kernel=SimpleNamespace(
            active_task_for=lambda *args: {"task_id": "task_1", "workspace_path": str(tmp_path)}
        ),
    )
    try:
        result = await PiToolDispatcher(service)._process_start(
            _request(
                "rosclaw_process_start",
                arguments={"command": "unused-in-this-admission-result-test"},
            )
        )
        assert result.status == state
        assert result.details["operation_id"] == "op_stable"
        assert result.reconciliation_required is (state == "LOST")
        if state in {"CANCELLED", "LOST"}:
            assert result.ok is False
    finally:
        conn.close()


async def test_cancel_during_spawn_waits_for_admission_and_reaps_the_child(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    _task(conn)
    manager = OperationManager(None, conn)
    entered, release = asyncio.Event(), asyncio.Event()
    original_spawn = manager._spawn
    created = []

    async def paused_spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        process = await original_spawn(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(manager, "_spawn", paused_spawn)
    starting = asyncio.create_task(
        manager.start(
            task_id="task_1",
            attempt_id="",
            kind="process",
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        )
    )
    cancellation = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        operation_id = conn.execute("SELECT operation_id FROM operations").fetchone()[0]
        cancellation = asyncio.create_task(
            manager.cancel(operation_id, reason="cancel-during-admission")
        )
        await asyncio.sleep(0.02)
        returned_before_spawn = cancellation.done()
        release.set()
        await starting
        await cancellation
        assert not returned_before_spawn, (
            "Cancellation cannot be confirmed before the owned spawn returns"
        )
        assert manager.get(operation_id)["state"] == "CANCELLED"
        assert created[0].returncode is not None
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        if cancellation:
            await asyncio.gather(cancellation, return_exceptions=True)
        for process in created:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        await manager.close()
        conn.close()
