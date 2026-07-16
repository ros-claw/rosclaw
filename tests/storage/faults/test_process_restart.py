"""Fault injection: process restart — reconcile finds the gap, repair closes it."""

from __future__ import annotations

from pathlib import Path

from rosclaw.practice.storage.catalog import PracticeCatalog


def test_restart_reconcile_and_backfill(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"

    # Process A: queue 50 events with a very slow flush interval, then "die"
    # before any flush (simulated by abandoning the catalog without close).
    catalog_a = PracticeCatalog(db, event_batch_size=500, event_flush_ms=60_000.0)
    expected_ids = {f"e{i:03d}" for i in range(50)}
    for i, event_id in enumerate(sorted(expected_ids), start=1):
        catalog_a.insert_event({"event_id": event_id, "practice_id": "p1", "_wm": i})
    # Kill the writers without flushing (process crash simulation).
    catalog_a._event_writer._closed = True
    catalog_a._event_index_writer._closed = True

    # Process B: reopen and reconcile — the unflushed events are missing.
    catalog_b = PracticeCatalog(db, event_batch_size=1)
    persisted = {
        row[0]
        for row in catalog_b._conn.execute("SELECT event_id FROM events WHERE practice_id = 'p1'")
    }
    missing = expected_ids - persisted
    assert len(missing) == 50  # nothing was committed before the crash

    # Backfill from the raw JSONL (source of truth) and reconcile again.
    for event_id in sorted(missing):
        catalog_b.insert_event({"event_id": event_id, "practice_id": "p1"})
    persisted = {
        row[0]
        for row in catalog_b._conn.execute("SELECT event_id FROM events WHERE practice_id = 'p1'")
    }
    assert persisted == expected_ids
    catalog_b.close()
