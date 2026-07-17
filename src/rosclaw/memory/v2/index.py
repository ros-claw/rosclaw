"""Memory 2.0 embedding index lifecycle manager (§6.5, §6.6).

Every index records::

    index_id / table_or_collection / embedder_type / model_name /
    model_revision / dimension / distance_metric / corpus_hash /
    index_version / record_count / created_at / completed_at / status

Lifecycle::

    BUILDING new_index → 全量校验 → READY → 原子切换 active_index → OLD → 延迟清理

Mixing embeddings from different models/dimensions/vocabularies in one query
is forbidden: the manager refuses to serve a query embedder whose identity
does not match the active index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from rosclaw.memory.v2.models import MemoryItem

logger = logging.getLogger("rosclaw.memory.v2.index")

REGISTRY_TABLE = "memory_index_registry"
VECTOR_TABLE = "memory_items"

# SQLite fallback boundaries (§6.6): above this many vectors, full-scan search
# is warned about; above the hard cap the fallback refuses to grow silently.
SQLITE_SCAN_WARN_RECORDS = 10_000
SQLITE_HARD_MAX_RECORDS = 200_000


class IndexModelMismatchError(RuntimeError):
    """Raised when a query embedder does not match the active index."""


class EmbeddingIndexManager:
    """Builds, switches, and maintains the memory embedding index."""

    def __init__(self, client: Any, vector_store: Any):
        self._client = client
        self._vector = vector_store

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def active_index(self, table: str = VECTOR_TABLE) -> dict[str, Any] | None:
        rows = self._client.query(
            REGISTRY_TABLE,
            filters={"table_or_collection": table, "status": "READY"},
            limit=1,
        )
        return rows[0] if rows else None

    def _insert_index_record(self, record: dict[str, Any]) -> None:
        self._client.insert(REGISTRY_TABLE, record)

    def _update_index_record(self, index_id: str, updates: dict[str, Any]) -> None:
        self._client.update(REGISTRY_TABLE, index_id, updates)

    # ------------------------------------------------------------------
    # Build + atomic switch
    # ------------------------------------------------------------------

    def build(
        self,
        memories: list[MemoryItem],
        embedder: Any,
        *,
        table: str = VECTOR_TABLE,
        embedder_type: str | None = None,
    ) -> dict[str, Any]:
        """Build a fresh index over *memories* and atomically make it active."""
        model_name = getattr(embedder, "model_name", None) or type(embedder).__name__
        model_revision = str(getattr(embedder, "revision", None) or "v1")
        dimension = getattr(embedder, "dim", None)
        if callable(dimension):
            dimension = dimension()

        index_id = f"idx_{uuid.uuid4().hex[:12]}"
        corpus_hash = self._corpus_hash(memories)
        previous = self.active_index(table)
        record = {
            "id": index_id,
            "table_or_collection": table,
            "embedder_type": embedder_type or type(embedder).__name__,
            "model_name": model_name,
            "model_revision": model_revision,
            "dimension": dimension,
            "distance_metric": "cosine",
            "corpus_hash": corpus_hash,
            "index_version": (previous or {}).get("index_version", 0) + 1,
            "record_count": 0,
            "created_at": time.time(),
            "completed_at": None,
            "status": "BUILDING",
        }
        self._insert_index_record(record)

        if len(memories) > SQLITE_HARD_MAX_RECORDS:
            self._update_index_record(index_id, {"status": "FAILED"})
            raise ValueError(
                f"SQLite vector fallback refuses {len(memories)} records "
                f"(hard max {SQLITE_HARD_MAX_RECORDS}); use native SeekDB instead"
            )
        if len(memories) > SQLITE_SCAN_WARN_RECORDS:
            logger.warning(
                "SQLite vector fallback: indexing %s records; search is a full "
                "table scan beyond %s — consider native SeekDB",
                len(memories),
                SQLITE_SCAN_WARN_RECORDS,
            )

        texts = [self._memory_text(item) for item in memories]
        try:
            embeddings = embedder.encode_batch(texts)
        except AttributeError:
            embeddings = [embedder.encode(text) for text in texts]
        if embeddings and dimension is not None:
            bad = [len(vec) for vec in embeddings if len(vec) != dimension]
            if bad:
                self._update_index_record(index_id, {"status": "FAILED"})
                raise IndexModelMismatchError(
                    f"Embedder returned dimensions {set(bad)} but declared {dimension}"
                )

        self._vector.upsert_many(
            table,
            [
                (item.memory_id, text, embedding)
                for item, text, embedding in zip(memories, texts, embeddings, strict=True)
            ],
        )
        self._update_index_record(
            index_id,
            {
                "status": "READY",
                "record_count": len(memories),
                "completed_at": time.time(),
            },
        )
        # Atomic switch: previous active index becomes OLD (kept for forensics).
        if previous is not None:
            self._update_index_record(previous["id"], {"status": "OLD"})
        logger.info(
            "Embedding index %s READY: %s records (%s/%s, dim=%s, version %s)",
            index_id,
            len(memories),
            model_name,
            model_revision,
            dimension,
            record["index_version"],
        )
        return self._client.query(REGISTRY_TABLE, filters={"id": index_id}, limit=1)[0]

    def needs_rebuild(self, embedder: Any, *, table: str = VECTOR_TABLE) -> tuple[bool, str]:
        """True when the embedder identity diverges from the active index."""
        active = self.active_index(table)
        if active is None:
            return True, "no active index"
        model_name = getattr(embedder, "model_name", None) or type(embedder).__name__
        dimension = getattr(embedder, "dim", None)
        if callable(dimension):
            dimension = dimension()
        if active.get("model_name") != model_name:
            return True, f"model changed: {active.get('model_name')} -> {model_name}"
        if dimension is not None and active.get("dimension") != dimension:
            return True, f"dimension changed: {active.get('dimension')} -> {dimension}"
        return False, "up to date"

    def check_query_embedder(self, embedder: Any, *, table: str = VECTOR_TABLE) -> None:
        """Forbid querying with an embedder that does not match the index."""
        rebuild, reason = self.needs_rebuild(embedder, table=table)
        if rebuild:
            raise IndexModelMismatchError(
                f"Query embedder does not match active index: {reason}. "
                "Rebuild the index first (memory index rebuild)."
            )

    # ------------------------------------------------------------------
    # Incremental maintenance
    # ------------------------------------------------------------------

    def on_memory_written(self, item: MemoryItem, embedder: Any) -> None:
        """Upsert one memory into the active index (called on store/update)."""
        active = self.active_index()
        if active is None:
            return
        self.check_query_embedder(embedder)
        self._vector.upsert_many(
            VECTOR_TABLE,
            [(item.memory_id, self._memory_text(item), embedder.encode(self._memory_text(item)))],
        )
        self._update_index_record(active["id"], {"record_count": self._vector.count(VECTOR_TABLE)})

    def on_memory_deleted(self, memory_id: str) -> bool:
        """Sync a memory deletion into the vector index (§6.6)."""
        deleted = self._vector.delete(VECTOR_TABLE, memory_id)
        active = self.active_index()
        if active is not None:
            self._update_index_record(
                active["id"], {"record_count": self._vector.count(VECTOR_TABLE)}
            )
        return deleted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_text(item: MemoryItem) -> str:
        return f"{item.title}\n{item.document}\n{' '.join(item.tags)}"

    @staticmethod
    def _corpus_hash(memories: list[MemoryItem]) -> str:
        digest = hashlib.sha256()
        for item in sorted(memories, key=lambda entry: entry.memory_id):
            digest.update(item.memory_id.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(item.content_hash.encode("utf-8"))
            digest.update(b"\x1e")
        return digest.hexdigest()

    def status(self) -> dict[str, Any]:
        """Registry summary for ``rosclaw memory index status``."""
        rows = self._client.query(REGISTRY_TABLE, limit=100)
        active = self.active_index()
        return {
            "indexes": rows,
            "active": active,
            "indexed_records": self._vector.count(VECTOR_TABLE),
            "sqlite_scan_warn_records": SQLITE_SCAN_WARN_RECORDS,
            "sqlite_hard_max_records": SQLITE_HARD_MAX_RECORDS,
        }


def registry_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a registry row for JSON output."""
    return json.loads(json.dumps(row, default=str))
