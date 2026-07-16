"""Fault injection: remote timeout — backoff, dead letter with payload, replay."""

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


def test_repeated_timeouts_dead_letter_with_full_payload(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    committer.save_to_seekdb.side_effect = TimeoutError("read timed out")
    worker = OutboxWorker(outbox, committer, interval_sec=0.02, max_retries=3)
    rid = outbox.enqueue(
        "seekdb_http",
        {"event": "important", "data": [1, 2, 3]},
        idempotency_key="failure:f1:v1",
        entity_type="failure",
        entity_id="f1",
    )
    worker.start()
    for _ in range(500):
        if outbox.stats()["dead_letters"] == 1:
            break
        time.sleep(0.02)
    worker.stop()

    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0].id == rid
    # Payload and identity survive dead-lettering (no {} upload, no loss).
    assert dead[0].payload == {"event": "important", "data": [1, 2, 3]}
    assert dead[0].idempotency_key == "failure:f1:v1"
    assert "timed out" in (dead[0].error_log or "")

    # Operator replays the dead letter once the remote is healthy again.
    assert outbox.requeue_dead_letter(rid)
    committer.save_to_seekdb.side_effect = None
    worker2 = OutboxWorker(outbox, committer, interval_sec=0.02)
    worker2.start()
    for _ in range(500):
        if outbox.stats()["delivered"] == 1:
            break
        time.sleep(0.02)
    worker2.stop()
    assert outbox.stats()["delivered"] == 1
    payload = committer.save_to_seekdb.call_args_list[-1][0][0]
    assert payload["idempotency_key"] == "failure:f1:v1"
