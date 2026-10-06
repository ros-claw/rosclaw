"""Process birth acquisition: empty raw cmdline capture is never authority.

SOURCE-only scope: real finite owned subprocess fixtures, a declared raw-read
injection (real PID, real capture API), and the durable ledger. No signals to
foreign PIDs, no ROS, no legacy ledger rewrite. These tests do not prove the
original incident's race timing or cause.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OPERATION_TERMINAL, OperationManager
from rosclaw.task_kernel.process_identity import ProcessIdentity

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _manager(tmp_path: Path) -> tuple[sqlite3.Connection, OperationManager]:
    conn = sqlite3.connect(tmp_path / "ledger.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    conn.execute(
        "INSERT INTO tasks(task_id,mission_id,root_goal,mode,state,active_revision,"
        "workspace_path,created_at,updated_at)VALUES('t','m','birth tests','SIMULATION',"
        "'RUNNING',1,?,'now','now')",
        (str(tmp_path),),
    )
    conn.commit()
    return conn, OperationManager(None, conn)


def test_empty_raw_cmdline_read_is_rejected_at_capture(tmp_path: Path) -> None:
    """Declared synthetic injection: the real capture API against a real owned
    live PID sees an empty raw cmdline read and must return None — the empty
    capture is never adopted. The finally block performs the test's own
    finite cleanup of the owned child: Popen.terminate() then wait()."""
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        original = Path.read_bytes
        target = Path("/proc") / str(child.pid) / "cmdline"
        calls = []

        def read(path, *args, **kwargs):
            if path == target:
                calls.append(1)
                return b""
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", read):
            captured = ProcessIdentity.capture(child.pid)
        assert calls, "raw cmdline read was not exercised"
        assert captured is None, "empty raw capture was adopted as authority"
        # Sanity: unpatched capture of the same live child is complete.
        real = ProcessIdentity.capture(child.pid)
        assert real is not None and real.command_sha256 != _EMPTY_SHA256
    finally:
        child.terminate()
        child.wait(timeout=3)


def test_legacy_empty_hash_row_parses_untrusted_and_bytes_survive() -> None:
    """An old incomplete legacy row (empty-cmdline hash) stays intact on disk
    and is reported unknown by parse instead of being adopted."""
    row = {
        "boot_id": "ea707e7c-9144-4e2a-8a82-5f7a4d486032",
        "command_sha256": _EMPTY_SHA256,
        "cwd": "/tmp/legacy",
        "pgid": 337009,
        "pid": 337009,
        "sid": 337009,
        "start_ticks": 24733049,
        "uid": 1000,
    }
    raw = json.dumps(row)
    assert ProcessIdentity.parse(raw) is None
    assert json.dumps(row) == raw, "legacy row bytes were rewritten"


@pytest.mark.asyncio
async def test_start_persists_valid_authority_and_never_refreshes(tmp_path: Path) -> None:
    """Bounded initial acquisition yields a complete persisted 8-field
    identity; a later natural finite exit never refreshes it."""
    conn, manager = _manager(tmp_path)
    try:
        op = await manager.start(
            task_id="t",
            attempt_id="",
            kind="process",
            argv=[sys.executable, "-B", "-c", "import time; time.sleep(.3)"],
            cwd=str(tmp_path),
        )
        saved = op.get("process_identity_json")
        identity = ProcessIdentity.parse(saved or "")
        assert identity is not None, "valid authority was not captured"
        assert identity.command_sha256 != _EMPTY_SHA256
        proc = manager._procs.get(op["operation_id"])
        if proc is not None:
            await asyncio.wait_for(proc.wait(), 3)
        until = asyncio.get_running_loop().time() + 3
        while manager.get(op["operation_id"])["state"] not in OPERATION_TERMINAL:
            assert asyncio.get_running_loop().time() < until
            await asyncio.sleep(0.01)
        current = manager.get(op["operation_id"])
        assert current["state"] == "SUCCEEDED"
        assert current.get("process_identity_json") == saved, "accepted authority was refreshed"
        assert not identity.same_birth(ProcessIdentity.capture(identity.pid))
    finally:
        await manager.close()
        conn.commit()
        conn.close()


@pytest.mark.asyncio
async def test_very_short_natural_exit_preserves_actual_code(tmp_path: Path) -> None:
    """A process exiting naturally almost immediately keeps its real exit code
    and output; nothing claims a cancel stop confirmation."""
    conn, manager = _manager(tmp_path)
    try:
        op = await manager.start(
            task_id="t",
            attempt_id="",
            kind="process",
            argv=[sys.executable, "-B", "-c", "print('SHORT', flush=True); raise SystemExit(3)"],
            cwd=str(tmp_path),
        )
        proc = manager._procs.get(op["operation_id"])
        if proc is not None:
            await asyncio.wait_for(proc.wait(), 3)
        until = asyncio.get_running_loop().time() + 3
        while manager.get(op["operation_id"])["state"] not in OPERATION_TERMINAL:
            assert asyncio.get_running_loop().time() < until
            await asyncio.sleep(0.01)
        current = manager.get(op["operation_id"])
        assert current["state"] == "FAILED"
        assert current["failure_code"] == "exit_3"
        events = conn.execute(
            "SELECT event_type FROM task_events WHERE operation_id = ?",
            (op["operation_id"],),
        ).fetchall()
        types = {row["event_type"] for row in events}
        assert "operation.process_stopped" not in types
        assert "operation.cancelled" not in types
    finally:
        await manager.close()
        conn.commit()
        conn.close()
