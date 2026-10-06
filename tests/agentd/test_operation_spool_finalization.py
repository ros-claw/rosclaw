"""Stale empty reads must never seal spool finalization with late bytes unread.

Regression for the demonstrated race: an owned finite producer can write its
final bytes after the observer's empty read but before the writer-closed/exit
checks. Finalization may only seal on a confirming empty read taken while every
writer is provably closed; the durable output, raw spool and checkpoint must
all cover those late bytes exactly once, across close/handoff too.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager


def database(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn


def output_events(manager, task="t"):
    return [e for e in manager.events_since(task, 0) if e["event_type"] == "operation.output"]


def text(manager, task="t"):
    return "".join(e["payload"]["text"] for e in output_events(manager, task))


async def _start_gated_late_writer(
    manager, tmp_path, expected, rc, *, block_probe_until_closed, write_delay=0.0
):
    """Own finite producer holding its final write on a gate file. The wrapped
    writer-closed probe releases the gate only after a production empty read
    already reached the probe. With block_probe_until_closed the probe then
    returns closed only once the real writer FD is gone and the real PID is
    dead — the exact stale-empty-read interleaving."""
    gate = tmp_path / f"release_{time.monotonic_ns()}"
    original = manager._writers_closed
    spawned = []
    original_spawn = manager._spawn

    async def capture_spawn(*args, **kwargs):
        proc = await original_spawn(*args, **kwargs)
        spawned.append(proc)
        return proc

    manager._spawn = capture_spawn

    def probe(fd):
        if not gate.exists():
            gate.write_text("GO")
        if not block_probe_until_closed:
            return original(fd)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            closed = original(fd)
            dead = bool(spawned) and not manager._pid_alive(spawned[0].pid)
            if closed and dead:
                return True
            time.sleep(0.001)
        raise AssertionError("owned producer did not close writer and exit in time")

    manager._writers_closed = probe
    delay = f"time.sleep({write_delay})\n" if write_delay else ""
    program = (
        "import os,sys,time\n"
        f"deadline = time.monotonic() + 5\n"
        f"while not os.path.exists({str(gate)!r}):\n"
        "    if time.monotonic() > deadline: raise SystemExit('fixture gate timeout')\n"
        "    time.sleep(0.001)\n"
        f"{delay}"
        f"os.write(1, {expected.encode()!r})\n"
        f"sys.exit({rc})"
    )
    op = await manager.start(
        task_id="t",
        attempt_id="",
        kind="process",
        argv=[sys.executable, "-c", program],
        cwd=str(tmp_path),
    )
    return op, spawned[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("rc", [0, 7])
async def test_stale_empty_read_race_drains_utf8_tail_exactly_once(tmp_path, rc):
    conn = database(tmp_path / "ledger.db")
    manager = OperationManager(None, conn)
    # Multibyte tail without trailing newline: the drain must feed the same
    # incremental UTF-8 decoder and keep the replace policy intact.
    expected = f"中🎾LATE_TAIL_{rc}\n中🎾"
    op, proc = await _start_gated_late_writer(
        manager, tmp_path, expected, rc, block_probe_until_closed=True
    )
    oid = op["operation_id"]
    try:
        final = await manager.wait(oid, timeout=5)
        assert final["state"] == ("SUCCEEDED" if rc == 0 else "FAILED")
        assert final["failure_code"] == ("" if rc == 0 else f"exit_{rc}")
        events = manager.events_since("t", 0)
        outputs = output_events(manager)
        # Every byte delivered exactly once: concatenation equality plus a
        # single terminal completion event.
        assert "".join(e["payload"]["text"] for e in outputs) == expected
        assert all(0 < len(e["payload"]["text"]) <= 4000 for e in outputs)
        terminal_event = "operation.completed" if rc == 0 else "operation.failed"
        assert len([e for e in events if e["event_type"] == terminal_event]) == 1
        raw = Path(final["output_path"]).read_bytes()
        assert raw == expected.encode()
        checkpoint = json.loads(final["output_checkpoint_json"])
        assert checkpoint["finalized"] is True
        assert checkpoint["offset"] == len(raw)
        assert checkpoint["pending_hex"] == ""
        info = os.stat(final["output_path"])
        assert checkpoint["device"] == info.st_dev
        assert checkpoint["inode"] == info.st_ino
    finally:
        await asyncio.wait_for(proc.wait(), 5)
        await manager.close()
        conn.close()


@pytest.mark.asyncio
async def test_late_tail_survives_close_reopen_handoff_without_duplicates(tmp_path):
    db = tmp_path / "ledger.db"
    conn = database(db)
    manager = OperationManager(None, conn)
    expected = "中🎾HANDOFF_LATE_TAIL\n"
    op, proc = await _start_gated_late_writer(
        manager,
        tmp_path,
        expected,
        0,
        block_probe_until_closed=False,
        write_delay=0.3,
    )
    oid = op["operation_id"]
    try:
        # Observer released the producer on its first empty-read probe and is
        # now polling; hand off while the late write is still in flight.
        await asyncio.sleep(0.1)
        await asyncio.gather(manager.close(), manager.close())
        assert manager.get(oid)["state"] == "RUNNING"
        conn.close()
        await asyncio.wait_for(proc.wait(), 5)
        conn = database(db)
        recovered = OperationManager(None, conn)
        report = await recovered.recover_on_boot()
        # Dead writer at boot is drained synchronously (terminated); a still
        # living one is reattached. Either way exactly one owner drains it.
        assert report["terminated"] + report["reattached"] == 1
        final = await recovered.wait(oid, timeout=5)
        assert final["state"] == "SUCCEEDED" and final["failure_code"] == ""
        events = recovered.events_since("t", 0)
        assert text(recovered) == expected
        assert len([e for e in events if e["event_type"] == "operation.completed"]) == 1
        raw = Path(final["output_path"]).read_bytes()
        assert raw == expected.encode()
        checkpoint = json.loads(final["output_checkpoint_json"])
        assert checkpoint["finalized"] is True
        assert checkpoint["offset"] == len(raw)
        assert checkpoint["pending_hex"] == ""
        await recovered.close()
    finally:
        await asyncio.wait_for(proc.wait(), 5)
        for driver in manager._drivers.values():
            await asyncio.gather(driver, return_exceptions=True)
        conn.close()
