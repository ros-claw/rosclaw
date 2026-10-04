"""Real producers retain output/exit status across observer shutdown, without kill."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager


def database(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn


def text(manager, task="t"):
    return "".join(
        e["payload"]["text"]
        for e in manager.events_since(task, 0)
        if e["event_type"] == "operation.output"
    )


@pytest.mark.asyncio
async def test_real_large_utf8_producer_survives_close_reopen_exactly_once(tmp_path):
    db = tmp_path / "ledger.db"
    conn = database(db)
    manager = OperationManager(None, conn)
    expected = "中🎾AB" * 300000 + "\nERR\nTAIL"
    program = (
        'import os,time,sys; data=("中🎾AB"*300000).encode(); '
        "\nfor i in range(0,len(data),4093):\n os.write(1,data[i:i+4093]); time.sleep(.0007)\n"
        'os.write(2,b"\\nERR\\n"); os.write(1,b"TAIL"); sys.exit(7)'
    )
    op = await manager.start(
        task_id="t", attempt_id="", kind="process", argv=[sys.executable, "-c", program]
    )
    proc = manager._procs[op["operation_id"]]
    try:
        await asyncio.sleep(0.04)
        await asyncio.gather(manager.close(), manager.close())
        assert proc.returncode is None
        conn.close()
        await asyncio.sleep(0.06)
        conn = database(db)
        recovered = OperationManager(None, conn)
        report = await recovered.recover_on_boot()
        assert report["reattached"] == 1
        final = await recovered.wait(op["operation_id"], timeout=20)
        assert final["state"] == "FAILED" and final["failure_code"] == "exit_7"
        assert text(recovered) == expected
        assert proc.stdout is None
        assert await proc.wait() == 7
        cursor = json.loads(final["output_checkpoint_json"])
        assert cursor["offset"] == len(expected.encode()) and cursor["finalized"]
        assert all(
            len(e["payload"]["text"]) <= 4000
            for e in recovered.events_since("t", 0)
            if e["event_type"] == "operation.output"
        )
        await recovered.close()
    finally:
        await asyncio.wait_for(proc.wait(), 20)
        for driver in manager._drivers.values():
            await asyncio.gather(driver, return_exceptions=True)
        conn.close()


@pytest.mark.asyncio
async def test_split_utf8_pending_and_finished_process_recover_before_terminal(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[
            sys.executable,
            "-c",
            r'import os,time; os.write(1,b"\xe4"); time.sleep(.2); os.write(1,b"\xb8\xad\xf0")',
        ],
    )
    proc = manager._procs[op["operation_id"]]
    try:
        await asyncio.sleep(0.06)
        await manager.close()
        conn.close()
        await proc.wait()
        conn = database(tmp_path / "ledger.db")
        second = OperationManager(None, conn)
        await second.recover_on_boot()
        result = await second.wait(op["operation_id"], timeout=3)
        assert result["state"] == "SUCCEEDED"
        assert text(second) == "中�"
        await second.close()
    finally:
        await proc.wait()
        conn.close()


@pytest.mark.asyncio
async def test_descendant_keeps_inherited_output_after_wrapper_exit(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    child = 'import time,os; time.sleep(.15); os.write(1,b"DESCENDANT_TAIL")'
    parent = (
        f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{child!r}]); sys.exit(7)'
    )
    op = await manager.start(
        task_id="t", attempt_id="", kind="process", argv=[sys.executable, "-c", parent]
    )
    result = await manager.wait(op["operation_id"], timeout=3)
    assert result["state"] == "FAILED" and result["failure_code"] == "exit_7"
    assert text(manager) == "DESCENDANT_TAIL"
    await manager.close()
    conn.close()


@pytest.mark.asyncio
async def test_reader_failure_rolls_back_event_cursor_never_false_success(tmp_path, monkeypatch):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    emit = manager._emit

    def fault(task, event, payload, **kwargs):
        emit(task, event, payload, **kwargs)
        if event == "operation.output":
            raise OSError("fixture storage observer failure after insert")

    monkeypatch.setattr(manager, "_emit", fault)
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", 'print("COMPLETE_TEXT")'],
    )
    proc = manager._procs[op["operation_id"]]
    result = await manager.wait(op["operation_id"], timeout=3)
    assert result["state"] == "DEGRADED"
    assert result["failure_code"] == "OUTPUT_OBSERVATION_UNRESOLVED"
    assert text(manager) == ""
    assert json.loads(result["output_checkpoint_json"])["offset"] == 0
    await manager.close()
    await proc.wait()
    monkeypatch.setattr(manager, "_emit", emit)
    recovered = OperationManager(None, conn)
    await recovered.recover_on_boot()
    final = await recovered.wait(op["operation_id"], timeout=3)
    assert final["state"] == "SUCCEEDED"
    assert text(recovered) == "COMPLETE_TEXT\n"
    await recovered.close()
    conn.close()


@pytest.mark.asyncio
async def test_legacy_pipe_close_is_unresolved_without_cancel_or_db_close(tmp_path):
    from rosclaw.task_kernel.operation_manager import OperationHandoffUnresolvedError

    class LegacyManager(OperationManager):
        async def _spawn(self, operation_id, argv, cwd, env, exitcode_path):
            return await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

    conn = database(tmp_path / "ledger.db")
    manager = LegacyManager(None, conn)
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", 'import time; time.sleep(.1); print("LEGACY_SURVIVES")'],
    )
    with pytest.raises(OperationHandoffUnresolvedError, match="HANDOFF_UNRESOLVED") as caught:
        await manager.close()
    assert caught.value.operations[0]["reason"] == "legacy_active_PIPE"
    assert not manager._drivers[op["operation_id"]].cancelling()
    assert conn.execute("SELECT 1").fetchone()[0] == 1
    final = await manager.wait(op["operation_id"], timeout=3)
    assert final["state"] == "SUCCEEDED"
    assert text(manager) == "LEGACY_SURVIVES\n"
    await manager.close()
    conn.close()


@pytest.mark.asyncio
async def test_invalid_cursor_and_spool_symlink_fail_closed(tmp_path):
    from rosclaw.task_kernel.operation_manager import OperationHandoffUnresolvedError

    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", 'import time; time.sleep(.15); print("OWNED")'],
    )
    row = manager.get(op["operation_id"])
    path = Path(row["output_path"])
    assert path.stat().st_mode & 0o777 == 0o600
    original = row["output_checkpoint_json"]
    checkpoint = json.loads(original)
    checkpoint["offset"] = 10**9
    conn.execute(
        "UPDATE operations SET output_checkpoint_json=? WHERE operation_id=?",
        (json.dumps(checkpoint), op["operation_id"]),
    )
    with pytest.raises(OperationHandoffUnresolvedError):
        manager.preflight_handoff()
    conn.execute(
        "UPDATE operations SET output_checkpoint_json=? WHERE operation_id=?",
        (original, op["operation_id"]),
    )
    moved = path.with_suffix(".saved")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(OperationHandoffUnresolvedError):
        manager.preflight_handoff()
    path.unlink()
    moved.rename(path)
    await manager.wait(op["operation_id"], timeout=3)
    await manager.close()
    conn.close()


@pytest.mark.asyncio
async def test_active_action_shutdown_is_explicitly_unresolved(tmp_path):
    from rosclaw.task_kernel.operation_manager import OperationHandoffUnresolvedError

    class MockActionClient:
        def send_goal(self, **kwargs):
            self.callbacks = kwargs

    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    client = MockActionClient()
    await manager.start_action(
        task_id="t",
        attempt_id="",
        action="/fixture",
        action_type="fixture/Action",
        args={},
        client=client,
    )
    with pytest.raises(OperationHandoffUnresolvedError) as caught:
        await manager.close()
    assert caught.value.operations[0]["reason"] == "active_action_no_handoff"
    assert not manager._closing
    assert conn.execute("SELECT 1").fetchone()[0] == 1
    conn.close()


@pytest.mark.asyncio
async def test_fresh_parent_process_and_event_loop_exit_preserve_child(tmp_path):
    import os

    root = Path(__file__).resolve().parents[2]
    db = tmp_path / "ledger.db"
    expected = "中🎾AB" * 300000 + "END"
    producer = (
        'import os,time,sys; time.sleep(.2); data=("中🎾AB"*300000).encode(); '
        "\nfor i in range(0,len(data),4093):\n os.write(1,data[i:i+4093]); time.sleep(.0007)\n"
        'os.write(1,b"END");sys.exit(7)'
    )
    creator = f"""
import asyncio,sqlite3,json
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager
async def main():
    c=sqlite3.connect({str(db)!r},isolation_level=None);c.row_factory=sqlite3.Row
    MigrationRunner().apply(c,'sqlite');m=OperationManager(None,c)
    op=await m.start(task_id='t',attempt_id='',kind='process',argv=[{sys.executable!r},'-c',{producer!r}])
    await m.close();c.close();print(json.dumps(op))
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        creator,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
    assert process.returncode == 0, stderr.decode()
    op = json.loads(stdout)
    conn = database(db)
    manager = OperationManager(None, conn)
    try:
        assert manager._pid_alive(op["pid"])
        assert (await manager.recover_on_boot())["reattached"] == 1
        final = await manager.wait(op["operation_id"], timeout=20)
        assert final["state"] == "FAILED" and final["failure_code"] == "exit_7"
        assert text(manager) == expected
    finally:
        if manager.get(op["operation_id"])["state"] not in {
            "FAILED",
            "SUCCEEDED",
            "LOST",
            "CANCELLED",
        }:
            await manager.cancel(op["operation_id"], reason="private-fixture-cleanup")
        await manager.close()
        conn.close()


def test_migration_preserves_legacy_rows_as_unknown_output(tmp_path):
    import shutil

    import rosclaw.storage.migrations as migrations

    existing = tmp_path / "old-migrations"
    existing.mkdir()
    for path in (Path(migrations.__file__).parent / "migrations").glob("*.sql"):
        if not path.name.startswith("041_"):
            shutil.copyfile(path, existing / path.name)
    conn = sqlite3.connect(tmp_path / "ledger.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    MigrationRunner(existing).apply(conn, "sqlite")
    conn.execute(
        "INSERT INTO operations(operation_id,task_id,attempt_id,kind,state,started_at) "
        "VALUES('legacy','t','','process','RUNNING','2026-10-04T00:00:00+00:00')"
    )
    MigrationRunner().apply(conn, "sqlite")
    row = dict(conn.execute("SELECT * FROM operations WHERE operation_id='legacy'").fetchone())
    assert row["output_path"] == row["output_checkpoint_json"] == ""
    assert row["state"] == "RUNNING"
    conn.close()


@pytest.mark.asyncio
async def test_cancelled_operation_unread_output_drains_without_changing_terminal(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    original = manager._drive_spool

    async def delayed_observer(*args, **kwargs):
        await asyncio.Event().wait()
        await original(*args, **kwargs)

    manager._drive_spool = delayed_observer
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[
            sys.executable,
            "-c",
            'import os,time; os.write(1,b"BEFORE_CANCEL\\n"); time.sleep(2)',
        ],
    )
    await asyncio.sleep(0.05)
    await manager.cancel(op["operation_id"], reason="private-owned-fixture")
    assert manager.get(op["operation_id"])["state"] == "CANCELLED"
    await manager.close()
    recovered = OperationManager(None, conn)
    report = await recovered.recover_on_boot()
    assert report["terminal_output_drained"] == 1
    assert recovered.get(op["operation_id"])["state"] == "CANCELLED"
    assert text(recovered) == "BEFORE_CANCEL\n"
    assert not recovered.stop_confirmation_missing(recovered.get(op["operation_id"]))
    await recovered.close()
    conn.close()


@pytest.mark.asyncio
async def test_overlapping_recovery_observers_do_not_duplicate_bytes(tmp_path):
    conn = database(tmp_path / "ledger.db")
    first = OperationManager(None, conn)
    expected = "中🎾AB" * 1500
    op = await first.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[
            sys.executable,
            "-c",
            f"import os,time; time.sleep(.05);os.write(1,{expected!r}.encode())",
        ],
    )
    second = OperationManager(None, conn)
    await second.recover_on_boot()
    await asyncio.wait_for(asyncio.gather(*first._drivers.values(), *second._drivers.values()), 5)
    assert first.get(op["operation_id"])["state"] == "SUCCEEDED"
    assert text(first) == expected
    assert (
        len([e for e in first.events_since("t", 0) if e["event_type"] == "operation.completed"])
        == 1
    )
    await first.close()
    await second.close()
    conn.close()


@pytest.mark.asyncio
async def test_legacy_reader_fault_never_promotes_exit_zero_to_success(tmp_path):
    class LegacyFaultManager(OperationManager):
        async def _spawn(self, operation_id, argv, cwd, env, exitcode_path):
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self.real_read = proc.stdout.read

            async def read_fault(*args, **kwargs):
                raise OSError("private legacy reader fault")

            proc.stdout.read = read_fault
            return proc

    conn = database(tmp_path / "ledger.db")
    manager = LegacyFaultManager(None, conn)
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", 'print("LEGACY_UNOBSERVED")'],
    )
    result = await manager.wait(op["operation_id"], timeout=3)
    assert result["state"] == "DEGRADED"
    assert result["failure_code"] == "OUTPUT_OBSERVATION_UNRESOLVED"
    proc = manager._procs[op["operation_id"]]
    assert await proc.wait() == 0
    assert not any(e["event_type"] == "operation.completed" for e in manager.events_since("t", 0))
    # Only fixture cleanup drains the finite legacy PIPE. No claim of replay.
    proc.stdout.read = manager.real_read
    assert await proc.stdout.read() == b"LEGACY_UNOBSERVED\n"
    await manager.close()
    conn.close()


@pytest.mark.asyncio
async def test_invalid_utf8_boundary_still_bounds_event_text(tmp_path):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    program = r'import os,time;os.write(1,b"\xe4");time.sleep(.05);os.write(1,b"A"*4000)'
    op = await manager.start(
        task_id="t", attempt_id="", kind="process", argv=[sys.executable, "-c", program]
    )
    await manager.wait(op["operation_id"], timeout=3)
    assert text(manager) == "�" + "A" * 4000
    assert all(
        len(e["payload"]["text"]) <= 4000
        for e in manager.events_since("t", 0)
        if e["event_type"] == "operation.output"
    )
    await manager.close()
    conn.close()


@pytest.mark.asyncio
async def test_terminal_escaped_writer_recovery_and_wait_do_not_block_startup(tmp_path):
    from rosclaw.task_kernel.process_identity import ProcessIdentity

    ledger = tmp_path / "ledger.db"
    pidfile = tmp_path / "escaped_pid"
    conn = database(ledger)
    manager = OperationManager(None, conn)
    code = (
        "import os,time,pathlib; child=os.fork();\n"
        "if child==0:\n"
        " os.setsid();pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid()));"
        'time.sleep(2);os.write(1,"中🎾LATE".encode());os._exit(0)\n'
        "else:\n time.sleep(5)\n"
    )
    op = await manager.start(
        task_id="t", attempt_id="", kind="process", argv=[sys.executable, "-c", code]
    )
    oid = op["operation_id"]
    for _ in range(100):
        if pidfile.exists():
            break
        await asyncio.sleep(0.01)
    escaped = ProcessIdentity.capture(int(pidfile.read_text()))
    owner = ProcessIdentity.parse(manager.get(oid)["process_identity_json"])
    assert escaped and owner and escaped.sid != owner.sid
    await manager.cancel(oid, reason="private-owned-fixture")
    await manager.close()
    conn.close()
    conn = database(ledger)
    recovered = OperationManager(None, conn)
    report = await asyncio.wait_for(recovered.recover_on_boot(), 0.25)
    assert report["terminal_output_pending"] == 1
    assert report["terminal_output_drained"] == 0
    assert escaped.matches(), "handoff/recovery must not signal escaped writer"
    final = await asyncio.wait_for(recovered.wait(oid), 0.25)
    assert final["state"] == "CANCELLED"
    assert not recovered.stop_confirmation_missing(final)
    await asyncio.wait_for(asyncio.shield(recovered._drivers[oid]), 4)
    assert text(recovered) == "中🎾LATE"
    assert recovered.get(oid)["state"] == "CANCELLED"
    assert json.loads(recovered.get(oid)["output_checkpoint_json"])["finalized"]
    await recovered.close()
    conn.close()
