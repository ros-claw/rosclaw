"""Fault injection: network partition — outbox backlogs, robot task unaffected."""

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


def test_partition_backlogs_outbox_without_blocking_producer(outbox: OutboxStore) -> None:
    committer = MagicMock()
    committer.save_to_seekdb_batch = None
    committer.save_to_seekdb.side_effect = ConnectionError("network unreachable")
    worker = OutboxWorker(outbox, committer, interval_sec=0.05, batch_size=10, max_retries=100)
    worker.start()
    try:
        # Producer (robot task) keeps enqueueing during the partition — no raise.
        for i in range(25):
            outbox.enqueue("seekdb_http", {"seq": i}, idempotency_key=f"memory:m{i}:h{i}")
        time.sleep(0.4)
        stats = outbox.stats()
        assert stats["total"] == 25  # nothing delivered, nothing lost
        assert stats["delivered"] == 0
    finally:
        worker.stop()

    # Partition heals: drain everything.
    delivered_seqs: list[int] = []

    def _record_delivery(payload: dict) -> None:
        delivered_seqs.append(payload["seq"])

    committer.save_to_seekdb.side_effect = _record_delivery
    worker2 = OutboxWorker(outbox, committer, interval_sec=0.02, batch_size=10)
    worker2.start()
    for _ in range(500):
        if outbox.stats()["total"] == 0:
            break
        time.sleep(0.02)
    worker2.stop()
    assert outbox.stats()["delivered"] == 25
    assert sorted(delivered_seqs) == list(range(25))
