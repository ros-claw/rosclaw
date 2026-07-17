"""Fault injection: outbox worker crash — lease expiry, reclaim, redelivery."""

from __future__ import annotations

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


def test_worker_crash_mid_claim_recovers_via_lease(outbox: OutboxStore) -> None:
    outbox.enqueue("seekdb_http", {"event": "a"}, idempotency_key="episode:ep1:v1")
    # Worker 1 claims and "crashes" (never delivers, never releases).
    claimed = outbox.claim(10, owner="crashed-worker", lease_sec=0.05)
    assert len(claimed) == 1
    assert outbox.stats()["inflight"] == 1

    time.sleep(0.08)
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    worker2 = OutboxWorker(outbox, committer, interval_sec=60.0)
    worker2._drain_once()
    worker2.stop()
    assert committer.save_to_seekdb.call_count == 1
    assert outbox.stats()["delivered"] == 1


def test_crash_after_remote_commit_redelivers_but_remote_stays_idempotent(
    outbox: OutboxStore,
) -> None:
    """Worker delivers remotely, crashes before mark_delivered: the redelivery
    must not duplicate the remote record (idempotency key upsert)."""
    remote: dict[str, dict] = {}

    class IdempotentRemote:
        save_to_seekdb_batch = None

        def save_to_seekdb(self, payload: dict) -> None:
            # Idempotent upsert by the injected key.
            remote[payload["idempotency_key"]] = payload
            # Simulate the worker dying right after a successful remote commit:
            raise _WorkerDiedError() if payload.get("_die_once") else None

    class _WorkerDiedError(Exception):
        pass

    outbox.enqueue("seekdb_http", {"event": "a"}, idempotency_key="episode:ep2:v1")
    # First worker: remote commit succeeds but process "dies" before marking.
    claimed = outbox.claim(10, owner="w1", lease_sec=0.05)
    assert len(claimed) == 1
    remote[claimed[0].idempotency_key] = claimed[0].payload  # remote commit OK
    # (crash: no mark_delivered)

    time.sleep(0.08)
    committer = MagicMock()
    committer.save_to_seekdb_batch = None

    def upsert(payload: dict) -> None:
        remote[payload["idempotency_key"]] = payload

    committer.save_to_seekdb.side_effect = upsert
    worker2 = OutboxWorker(outbox, committer, interval_sec=60.0)
    worker2._drain_once()
    worker2.stop()
    # Redelivery happened, but the remote still has exactly one record.
    assert len(remote) == 1
    assert outbox.stats()["delivered"] == 1
