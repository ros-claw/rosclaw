"""SQLite → native SeekDB retrieval projection (PR-SDB-1 §7.5).

Transition architecture::

    MemoryRepository
        → Transaction (local store commit)
        → Local Store  (SQLite = source of truth)
        → Outbox       (durable, at-least-once)
        → SeekDB Projection  (native vector/BM25/hybrid retrieval index)

The projection is *disposable*: it can be dropped and rebuilt from the local
store at any time (:meth:`SeekDBProjection.rebuild`).  Projection writes are
idempotent (keyed by ``memory:<memory_id>``), so outbox redelivery never
duplicates the remote record.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rosclaw.storage.seekdb_projection")

PROJECTION_TABLE = "memory_items"
PROJECTION_TARGET = "seekdb_projection"


class SeekDBProjectionCommitter:
    """Outbox committer that upserts memory records into native SeekDB.

    Implements the ``save_to_seekdb`` / ``save_to_seekdb_batch`` protocol used
    by :class:`rosclaw.storage.outbox.OutboxWorker`.
    """

    def __init__(self, store: Any):
        self._store = store

    def save_to_seekdb(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.pop("idempotency_key", None)
        self._store.connect()
        self._store.insert(PROJECTION_TABLE, record)
        self._store.refresh_index(PROJECTION_TABLE)

    def save_to_seekdb_batch(self, payloads: list[dict[str, Any]]) -> None:
        records = []
        for payload in payloads:
            record = dict(payload)
            record.pop("idempotency_key", None)
            records.append(record)
        self._store.connect()
        self._store.insert_many(PROJECTION_TABLE, records)


class SeekDBProjection:
    """Maintains the native SeekDB retrieval projection of memory_items."""

    def __init__(self, store: Any, outbox: Any | None = None):
        self._store = store
        self._outbox = outbox

    def project(self, item_record: dict[str, Any]) -> None:
        """Project one memory record (post local-commit).

        With an outbox the projection is asynchronous and durable; without one
        it is a direct synchronous upsert (still idempotent).
        """
        memory_id = item_record.get("id")
        if not memory_id:
            logger.warning("projection skipped: record without id")
            return
        if self._outbox is not None:
            self._outbox.enqueue(
                PROJECTION_TARGET,
                item_record,
                idempotency_key=f"memory:{memory_id}:projection",
                entity_type="memory",
                entity_id=str(memory_id),
            )
            return
        self._store.connect()
        self._store.insert(PROJECTION_TABLE, item_record)

    def project_delete(self, memory_id: str) -> None:
        """Remove a memory from the projection (delete sync)."""
        self._store.connect()
        self._store.delete(PROJECTION_TABLE, memory_id)

    def rebuild(self, repository: Any, *, batch_size: int = 200) -> dict[str, Any]:
        """Rebuild the whole projection from the SQLite source of truth."""
        started = time.time()
        items = repository.query(limit=500_000)
        self._store.connect()
        total = 0
        for offset in range(0, len(items), batch_size):
            batch = [item.to_record() for item in items[offset : offset + batch_size]]
            total += self._store.insert_many(PROJECTION_TABLE, batch)
        self._store.refresh_index(PROJECTION_TABLE)
        return {
            "rebuilt": total,
            "elapsed_s": round(time.time() - started, 2),
            "projection": type(self._store).__name__,
        }
