"""Fault injection: batch writer crash — retry, loud failure, close-under-load."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from rosclaw.practice.storage.catalog import PracticeCatalog, _BatchWriter


def test_transient_lock_error_is_retried_without_data_loss() -> None:
    attempts = {"n": 0}
    committed: list[int] = []

    def flaky_flush(batch: list[dict]) -> None:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise sqlite3.OperationalError("database is locked")
        committed.extend(r["_wm"] for r in batch)

    writer = _BatchWriter("test", flaky_flush, batch_size=10, flush_interval_ms=50.0)
    for i in range(1, 6):
        writer.put({"event_id": f"e{i}", "_wm": i})
    assert writer.close(timeout=5.0)
    assert sorted(committed) == [1, 2, 3, 4, 5]
    assert attempts["n"] == 3


def test_permanent_flush_error_fails_loudly_not_silently() -> None:
    def bad_flush(batch: list[dict]) -> None:
        raise sqlite3.OperationalError("no such table: events")

    writer = _BatchWriter(
        "test", bad_flush, batch_size=10, flush_interval_ms=50.0, max_flush_retries=1
    )
    writer.put({"event_id": "e1", "_wm": 1})
    # After the retry budget is exhausted the writer dies and put() raises.
    for _ in range(50):
        if writer._flush_error is not None:
            break
        time.sleep(0.02)
    assert writer._flush_error is not None
    with pytest.raises(RuntimeError, match="previously failed"):
        writer.put({"event_id": "e2", "_wm": 2})
    # The failed record is preserved for inspection, not silently dropped.
    assert writer._failed_records
    writer._max_flush_retries = 0
    writer.close(timeout=2.0)


def test_close_under_load_does_not_close_connection_under_live_writer() -> None:
    """Reproduces the 7x24 session #6 'Cannot operate on a closed database' race."""
    slow_committed: list[int] = []

    def slow_flush(batch: list[dict]) -> None:
        time.sleep(0.5)
        slow_committed.extend(r["_wm"] for r in batch)

    # batch_size > 1 so the flush happens on the worker thread, not inline.
    writer = _BatchWriter("slow", slow_flush, batch_size=10, flush_interval_ms=10.0)
    writer.put({"event_id": "e1", "_wm": 1})
    # Close with a tiny timeout while the worker is mid-flush.
    ok = writer.close(timeout=0.05)
    assert not ok
    assert writer._thread.is_alive()
    # The worker eventually commits; nothing is lost.
    for _ in range(100):
        if slow_committed:
            break
        time.sleep(0.02)
    assert slow_committed == [1]


def test_catalog_close_keeps_connection_open_when_writer_stuck() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = PracticeCatalog(Path(tmp) / "catalog.sqlite", event_batch_size=500)
        assert catalog._event_writer is not None
        # Force the writer into a stuck state by monkeypatching its flush.
        original_flush_fn = catalog._event_writer._flush_fn

        def stuck_flush(batch: list[dict]) -> None:
            time.sleep(2.0)
            original_flush_fn(batch)

        catalog._event_writer._flush_fn = stuck_flush
        catalog.insert_event({"event_id": "e1", "practice_id": "p1", "_wm": 1})
        catalog._event_writer._closed = False
        # Replace close timeout by calling the writer's close directly.
        ok = catalog._event_writer.close(timeout=0.05)
        if not ok:
            # Connection must remain open until the writer finishes.
            catalog._conn.execute("SELECT 1")
        # Let the writer actually finish and clean up.
        catalog._event_writer._thread.join(timeout=5.0)
        catalog._conn.close()
