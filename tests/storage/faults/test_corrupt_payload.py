"""Fault injection: corrupt payload — dead letter, never uploaded as {}."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.storage.cli import reconcile_practice
from rosclaw.storage.outbox import OutboxStore, OutboxWorker


@pytest.fixture
def outbox(tmp_path: Path) -> OutboxStore:
    store = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    store.connect()
    return store


def test_corrupt_payload_dead_lettered_and_never_sent(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("seekdb_http", {"valid": True})
    # Simulate on-disk corruption of the stored JSON.
    outbox._connection.execute(
        "UPDATE outbox SET payload_json = ? WHERE id = ?",
        ('{"broken": truncated', rid),
    )
    outbox._connection.commit()

    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    worker = OutboxWorker(outbox, committer, interval_sec=60.0)
    worker._drain_once()
    worker.stop()

    committer.save_to_seekdb.assert_not_called()
    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0].id == rid
    assert dead[0].error_log == "payload checksum mismatch"


def test_corrupt_jsonl_line_fails_reconcile(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "prac_corrupt"
    raw = session_dir / "raw"
    raw.mkdir(parents=True)
    with (raw / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "e1", "practice_id": "prac_corrupt"}) + "\n")
        f.write("{not valid json\n")
        f.write(json.dumps({"event_id": "e2", "practice_id": "prac_corrupt"}) + "\n")
    (session_dir / "episode.json").write_text(json.dumps({"event_count": 3}))
    (session_dir / "manifest.yaml").write_text("event_count: 3\n")

    indexes = tmp_path / "indexes"
    indexes.mkdir()
    conn = sqlite3.connect(str(indexes / "practice_catalog.sqlite"))
    conn.execute("CREATE TABLE practices (practice_id TEXT, session_id TEXT, episode_id TEXT)")
    conn.execute("CREATE TABLE events (event_id TEXT, practice_id TEXT)")
    conn.execute("CREATE TABLE practice_event_index (event_id TEXT, session_id TEXT, episode_id TEXT)")
    conn.execute("INSERT INTO practices VALUES ('prac_corrupt', 's1', 'ep1')")
    for eid in ("e1", "e2"):
        conn.execute("INSERT INTO events VALUES (?, 'prac_corrupt')", (eid,))
        conn.execute("INSERT INTO practice_event_index VALUES (?, 's1', 'ep1')", (eid,))
    conn.commit()
    conn.close()

    report = reconcile_practice("prac_corrupt", str(tmp_path))
    assert not report["passed"]
    assert report["parse_errors"] == 1
