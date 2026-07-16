"""Tests for outbox schema v2: idempotency, lease claim, delivered retention."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.storage.outbox import OutboxStore, OutboxWorker


@pytest.fixture
def outbox(tmp_path: Path) -> OutboxStore:
    store = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    store.connect()
    return store


def test_enqueue_is_idempotent_on_key(outbox: OutboxStore) -> None:
    first = outbox.enqueue("t", {"a": 1}, idempotency_key="episode:ep1:v1")
    second = outbox.enqueue("t", {"a": 1}, idempotency_key="episode:ep1:v1")
    assert first == second
    assert outbox.stats()["total"] == 1


def test_enqueue_derives_key_and_stores_hash(outbox: OutboxStore) -> None:
    outbox.enqueue("t", {"a": 1}, entity_type="memory", entity_id="m1")
    record = outbox.pending(limit=1)[0]
    assert record.idempotency_key is not None
    assert record.idempotency_key.startswith("memory:m1:")
    assert record.payload_sha256 is not None
    assert len(record.payload_sha256) == 64


def test_claim_sets_inflight_and_lease(outbox: OutboxStore) -> None:
    outbox.enqueue("t", {"a": 1})
    claimed = outbox.claim(10, owner="w1", lease_sec=30.0)
    assert len(claimed) == 1
    record = claimed[0]
    assert record.status == "inflight"
    assert record.lease_owner == "w1"
    assert record.lease_expires_at is not None and record.lease_expires_at > time.time()
    # A second worker cannot claim the leased record.
    assert outbox.claim(10, owner="w2", lease_sec=30.0) == []


def test_claim_recovers_expired_lease(outbox: OutboxStore) -> None:
    outbox.enqueue("t", {"a": 1})
    claimed = outbox.claim(10, owner="w1", lease_sec=0.05)
    assert len(claimed) == 1
    time.sleep(0.08)
    reclaimed = outbox.claim(10, owner="w2", lease_sec=30.0)
    assert len(reclaimed) == 1
    assert reclaimed[0].lease_owner == "w2"


def test_concurrent_workers_claim_disjoint_records(outbox: OutboxStore) -> None:
    for i in range(6):
        outbox.enqueue("t", {"i": i})
    w1 = outbox.claim(3, owner="w1")
    w2 = outbox.claim(3, owner="w2")
    w3 = outbox.claim(3, owner="w3")
    assert len(w1) == 3
    assert len(w2) == 3
    assert w3 == []
    assert {r.id for r in w1}.isdisjoint({r.id for r in w2})


def test_mark_delivered_requires_owner(outbox: OutboxStore) -> None:
    outbox.enqueue("t", {"a": 1})
    claimed = outbox.claim(10, owner="w1")
    outbox.mark_delivered(claimed[0].id, owner="intruder")
    # Not delivered: lease owner mismatch, record still inflight under w1.
    assert outbox.stats()["inflight"] == 1
    outbox.mark_delivered(claimed[0].id, owner="w1", remote_revision="rev-7")
    assert outbox.stats()["inflight"] == 0
    assert outbox.stats()["delivered"] == 1
    record = outbox.records()[0]
    assert record.status == "delivered"
    assert record.remote_revision == "rev-7"


def test_purge_delivered_respects_retention(outbox: OutboxStore) -> None:
    outbox.enqueue("t", {"a": 1})
    claimed = outbox.claim(10, owner="w1")
    outbox.mark_delivered(claimed[0].id, owner="w1")
    assert outbox.purge_delivered(retention_sec=3600.0) == 0
    assert outbox.purge_delivered(retention_sec=0.0) == 1
    assert outbox.stats()["delivered"] == 0


def test_corrupt_payload_goes_to_dead_letters(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("t", {"a": 1})
    # Simulate on-disk corruption of the stored payload.
    outbox._connection.execute("UPDATE outbox SET payload_json = '{corrupted' WHERE id = ?", (rid,))
    outbox._connection.commit()
    claimed = outbox.claim(10, owner="w1")
    assert claimed == []
    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0].id == rid
    assert dead[0].error_log == "payload checksum mismatch"


def test_requeue_dead_letter(outbox: OutboxStore) -> None:
    rid = outbox.enqueue("t", {"a": 1})
    outbox.mark_failed(rid, "boom", max_retries=1)
    assert outbox.stats()["dead_letters"] == 1
    assert outbox.requeue_dead_letter(rid)
    assert outbox.stats()["dead_letters"] == 0
    pending = outbox.pending(limit=10)
    assert len(pending) == 1
    assert pending[0].id == rid
    assert pending[0].retry_count == 0
    # Requeueing a missing id is a no-op.
    assert not outbox.requeue_dead_letter("missing")


def test_worker_injects_idempotency_key(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    worker = OutboxWorker(outbox, committer, interval_sec=60.0)
    outbox.enqueue("t", {"a": 1}, idempotency_key="fact:f1:v2")
    worker._drain_once()
    worker.stop()
    payload = committer.save_to_seekdb.call_args[0][0]
    assert payload["idempotency_key"] == "fact:f1:v2"
    assert payload["a"] == 1
    # Original stored payload is not mutated.
    assert outbox.records()[0].payload == {"a": 1}


def test_worker_restart_resumes_unfinished_records(tmp_path: Path) -> None:
    store = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    store.connect()
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    committer.save_to_seekdb.side_effect = RuntimeError("down")
    worker = OutboxWorker(store, committer, interval_sec=60.0, lease_sec=0.05, max_retries=5)
    store.enqueue("t", {"a": 1})
    worker._drain_once()  # fails, record goes to retry
    worker.stop()
    assert store.stats()["total"] == 1

    # Simulate process restart: new store + worker over the same file.
    store.close()
    store2 = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    store2.connect()
    committer2 = MagicMock()
    committer2.save_to_seekdb_batch = None
    worker2 = OutboxWorker(store2, committer2, interval_sec=60.0)
    # Force the retry to be due.
    store2._connection.execute("UPDATE outbox SET next_retry_at = 0")
    store2._connection.commit()
    worker2._drain_once()
    worker2.stop()
    assert committer2.save_to_seekdb.call_count == 1
    assert store2.stats()["total"] == 0
    assert store2.stats()["delivered"] == 1
    store2.close()


def test_v1_database_migrates_in_place(tmp_path: Path) -> None:
    db = tmp_path / "outbox.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE outbox (
            id TEXT PRIMARY KEY,
            target TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            retry_count INTEGER DEFAULT 0,
            next_retry_at REAL,
            error_log TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO outbox (id, target, payload_json, created_at, retry_count) "
        "VALUES ('r1', 't', '{}', 1.0, 0)"
    )
    conn.commit()
    conn.close()

    store = OutboxStore(db_path=str(db))
    store.connect()
    record = store.pending(limit=1)[0]
    assert record.id == "r1"
    assert record.status == "pending"
    # New columns exist and are nullable for legacy rows.
    columns = {row[1] for row in store._connection.execute("PRAGMA table_info(outbox)")}
    for expected in ("idempotency_key", "status", "lease_owner", "payload_sha256"):
        assert expected in columns
    store.close()
