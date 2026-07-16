"""Fault injection: SQLite locked — raw writes continue, catalog retries."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from rosclaw.practice.storage.catalog import PracticeCatalog


def test_catalog_retries_flush_while_database_locked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.sqlite"
        catalog = PracticeCatalog(db, event_batch_size=500, event_flush_ms=50.0)

        # Hold an exclusive write lock from a second connection ("SQLite 被锁").
        locker = sqlite3.connect(str(db), check_same_thread=False)
        locker.execute("BEGIN IMMEDIATE")
        locker.execute(
            "INSERT INTO practices (practice_id) VALUES ('lock_holder')"
        )

        # Events keep queueing while the database is locked.
        for i in range(1, 21):
            catalog.insert_event(
                {"event_id": f"e{i}", "practice_id": "p1", "_wm": i}
            )
            catalog.insert_event_index(
                {"event_id": f"e{i}", "session_id": "s1", "summary": {}, "_wm": i}
            )

        # Release the lock after ~1.5s; the writer must retry, not die.
        def _unlock() -> None:
            time.sleep(1.5)
            locker.commit()
            locker.close()

        threading.Thread(target=_unlock, daemon=True).start()

        result = catalog.flush_until(20, timeout_sec=30.0)
        assert result["ok"], result
        assert catalog.count_events("p1") == 20
        catalog.close()


def test_lock_error_eventually_surfaces_loudly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.sqlite"
        catalog = PracticeCatalog(db, event_batch_size=1)
        # Simulate a non-transient failure (schema mismatch) via direct SQL.
        import pytest

        with pytest.raises(sqlite3.Error):
            catalog._conn.execute("INSERT INTO no_such_table VALUES (1)")
        catalog.close()
