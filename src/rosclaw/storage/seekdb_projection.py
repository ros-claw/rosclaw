"""SQLite → native SeekDB retrieval projection (PR-SDB-1 §7.5).

Transition architecture::

    MemoryRepository
        → Transaction (local store commit)
        → Local Store  (SQLite = source of truth)
        → Outbox       (durable, at-least-once)
        → SeekDB Projection  (native vector/BM25/hybrid retrieval index)

The projection is *disposable*: it can be dropped and rebuilt from the local
store at any time (:meth:`MemoryRetrievalProjection.rebuild`).  Projection writes are
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


class MemoryRetrievalProjectionCommitter:
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


class MemoryRetrievalProjection:
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

    def status(self, repository: Any | None = None) -> dict[str, Any]:
        """Projection observability (PR-DF-07 §18): watermark + lag.

        ``source_count`` is the structured-store memory_items watermark (when
        a repository is given); ``projection_count`` is what the retrieval
        index currently holds; ``lag`` is the difference.  The projection is
        a rebuildable projection — lag is an operations signal, never a data
        loss (rebuild() restores parity from the source of truth).
        """
        result: dict[str, Any] = {
            "projection_backend": type(self._store).__name__,
            "outbox_enabled": self._outbox is not None,
        }
        try:
            self._store.connect()
            result["projection_count"] = self._store.count(PROJECTION_TABLE)
        except Exception as exc:  # noqa: BLE001
            result["projection_count"] = None
            result["error"] = str(exc)
        if repository is not None:
            try:
                result["source_count"] = repository._client.count(PROJECTION_TABLE)
                if result.get("projection_count") is not None:
                    result["lag"] = result["source_count"] - result["projection_count"]
            except Exception as exc:  # noqa: BLE001
                result["source_count"] = None
                result.setdefault("error", str(exc))
        if self._outbox is not None:
            try:
                stats = self._outbox.stats()
                result["outbox_pending"] = stats.get("pending")
                result["outbox_deadletters"] = stats.get("deadletters", stats.get("dead_letters"))
            except Exception:  # noqa: BLE001
                pass
        return result

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


# ADR-0010 compatibility aliases (PR-DF-01): pre-rename names.
SeekDBProjection = MemoryRetrievalProjection
SeekDBProjectionCommitter = MemoryRetrievalProjectionCommitter
