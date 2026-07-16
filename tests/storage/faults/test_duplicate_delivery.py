"""Fault injection: duplicate delivery — remote holds exactly one record."""

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


def test_duplicate_enqueue_same_episode_creates_one_record(outbox: OutboxStore) -> None:
    key = "episode:ep_2026:v2"
    first = outbox.enqueue("seekdb_http", {"ep": 1}, idempotency_key=key)
    second = outbox.enqueue("seekdb_http", {"ep": 1}, idempotency_key=key)
    third = outbox.enqueue("seekdb_http", {"ep": 1}, idempotency_key=key)
    assert first == second == third
    assert outbox.stats()["total"] == 1


def test_two_workers_cannot_double_deliver(outbox: OutboxStore) -> None:
    remote: dict[str, dict] = {}

    class IdempotentCommitter:
        save_to_seekdb_batch = None

        def save_to_seekdb(self, payload: dict) -> None:
            time.sleep(0.01)  # widen the race window
            remote[payload["idempotency_key"]] = payload

    for i in range(10):
        outbox.enqueue("seekdb_http", {"seq": i}, idempotency_key=f"memory:m{i}:v1")

    w1 = OutboxWorker(outbox, IdempotentCommitter(), interval_sec=0.01, batch_size=4)
    w2 = OutboxWorker(outbox, IdempotentCommitter(), interval_sec=0.01, batch_size=4)
    w1.start()
    w2.start()
    for _ in range(1000):
        if outbox.stats()["total"] == 0 and len(remote) == 10:
            break
        time.sleep(0.02)
    w1.stop()
    w2.stop()
    assert len(remote) == 10
    assert outbox.stats()["delivered"] == 10
