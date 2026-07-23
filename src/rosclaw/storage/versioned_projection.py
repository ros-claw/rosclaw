"""Incremental projection of memory mutations into the ACTIVE generation (v4 §3.8).

Flow::

    MemoryRepository commit
    → Outbox (durable, optional) or direct
    → ActiveProjectionCommitter
    → ACTIVE physical collection

The plain ``memory_items`` projection (:mod:`rosclaw.storage.seekdb_projection`)
keeps the logical collection current, but the versioned ACTIVE collection is
what runtime queries actually serve — without this module it silently goes
stale the moment a new memory is written.

Semantics:

* status ``active`` → embed with the ACTIVE profile's provider and upsert.
* status ``superseded`` / ``expired`` / ``quarantined`` / delete → remove
  from the ACTIVE collection (supersede and delete must sync, v4 §3.8).
* no ACTIVE pointer / provider unavailable → the mutation is NOT silently
  dropped: the watermark records the failure and :func:`catch_up_collection`
  repairs the lag before the next activation/rollback.

Rollback freshness (v4 §3.8): an OLD collection is never assumed current.
``VersionedCollectionManager.rollback(..., catch_up=...)`` runs the catch-up
before the pointer switch and aborts if it cannot verify.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from rosclaw.memory.v2.runtime_retrieval.active_resolver import (
    ActiveCollectionResolver,
    ActiveIndexDescriptor,
    ActiveIndexUnavailableError,
)
from rosclaw.memory.v2.runtime_retrieval.provider_resolver import (
    EmbeddingProviderResolver,
    ProviderUnavailableError,
)

logger = logging.getLogger("rosclaw.storage.versioned_projection")

WATERMARK_KIND = "projection_watermark"
ACTIVE_PROJECTION_TARGET = "active_projection"
_REGISTRY_TABLE = "projection_registry"


def _watermark_id(logical_name: str) -> str:
    digest = hashlib.sha256(logical_name.encode()).hexdigest()[:24]
    return f"projection_watermark_{digest}"


def read_watermark(store: Any, logical_name: str) -> dict[str, Any] | None:
    """Last projection watermark for the logical collection, if any."""
    rows = store.query(
        _REGISTRY_TABLE,
        filters={"id": _watermark_id(logical_name)},
        limit=1,
    )
    return rows[0] if rows else None


def _write_watermark(
    store: Any,
    logical_name: str,
    *,
    last_memory_id: str | None,
    last_event_time: float | None,
    projected_delta: int,
    failed_delta: int,
    note: str | None = None,
) -> None:
    previous = read_watermark(store, logical_name) or {}
    row = {
        "id": _watermark_id(logical_name),
        "row_kind": WATERMARK_KIND,
        "logical_name": logical_name,
        "last_memory_id": last_memory_id or previous.get("last_memory_id"),
        "last_memory_event_time": (
            last_event_time
            if last_event_time is not None
            else previous.get("last_memory_event_time")
        ),
        "projected_count": int(previous.get("projected_count") or 0) + projected_delta,
        "failed_count": int(previous.get("failed_count") or 0) + failed_delta,
        "note": note,
        "updated_at": time.time(),
    }
    store.insert(_REGISTRY_TABLE, row)


class ActiveProjectionCommitter:
    """Outbox committer that mirrors memory mutations into the ACTIVE collection.

    Implements the ``save_to_seekdb`` / ``save_to_seekdb_batch`` protocol used
    by :class:`rosclaw.storage.outbox.OutboxWorker`.
    """

    def __init__(
        self,
        store: Any,
        provider_resolver: EmbeddingProviderResolver | None = None,
        *,
        logical_name: str = "memory_items",
    ) -> None:
        self._store = store
        self._resolver = provider_resolver or EmbeddingProviderResolver()
        self._logical_name = logical_name

    def save_to_seekdb(self, payload: dict[str, Any]) -> None:
        self.save_to_seekdb_batch([payload])

    def save_to_seekdb_batch(self, payloads: list[dict[str, Any]]) -> None:
        try:
            descriptor = ActiveCollectionResolver(self._store).resolve(self._logical_name)
        except ActiveIndexUnavailableError as exc:
            # Not a dead-letter: nothing to project into yet.  The watermark
            # records the gap so catch-up can repair it after activation.
            _write_watermark(
                self._store,
                self._logical_name,
                last_memory_id=None,
                last_event_time=None,
                projected_delta=0,
                failed_delta=len(payloads),
                note=f"no_active:{exc.reason}",
            )
            return
        client = getattr(self._store, "_client", None)
        if client is None:
            raise RuntimeError("native store is not connected")
        collection = client.get_collection(descriptor.physical_collection)

        provider = None
        provider_note: str | None = None
        try:
            provider = self._resolver.resolve(descriptor)
        except ProviderUnavailableError as exc:
            provider_note = f"provider_unavailable:{exc.reason}"

        projected = 0
        failed = 0
        last_id: str | None = None
        last_time: float | None = None
        upserts: list[dict[str, Any]] = []
        for payload in payloads:
            record = dict(payload)
            record.pop("idempotency_key", None)
            memory_id = str(record.get("id") or record.get("memory_id") or "")
            if not memory_id:
                failed += 1
                continue
            last_id = memory_id
            event_time = record.get("event_time")
            if isinstance(event_time, (int, float)):
                last_time = max(last_time or 0.0, float(event_time))
            if str(record.get("status") or "active") != "active":
                # Supersede / expire / quarantine / delete sync (v4 §3.8).
                collection.delete(ids=[memory_id])
                projected += 1
                continue
            if provider is None:
                failed += 1
                continue
            upserts.append(record)

        if upserts:
            documents = [str(r.get("document") or r.get("title") or r["id"]) for r in upserts]
            embeddings = provider.encode_documents(documents)
            collection.upsert(
                ids=[str(r.get("id") or r.get("memory_id")) for r in upserts],
                documents=documents,
                embeddings=embeddings,
                metadatas=[self._store._metadata(r) for r in upserts],
            )
            projected += len(upserts)
        with_refresh = getattr(self._store, "refresh_index", None)
        if callable(with_refresh):
            with_refresh(descriptor.physical_collection)
        _write_watermark(
            self._store,
            self._logical_name,
            last_memory_id=last_id,
            last_event_time=last_time,
            projected_delta=projected,
            failed_delta=failed,
            note=provider_note,
        )


class ActiveProjection:
    """Repository projection hook targeting the ACTIVE physical collection.

    Mirrors :class:`rosclaw.storage.seekdb_projection.SeekDBProjection`:
    with an outbox the projection is asynchronous and durable; without one it
    is a direct synchronous commit (still idempotent — upsert keyed by id).
    """

    def __init__(
        self,
        store: Any,
        provider_resolver: EmbeddingProviderResolver | None = None,
        outbox: Any | None = None,
        *,
        logical_name: str = "memory_items",
    ) -> None:
        self._store = store
        self._resolver = provider_resolver
        self._outbox = outbox
        self._logical_name = logical_name

    def project(self, item_record: dict[str, Any]) -> None:
        memory_id = item_record.get("id") or item_record.get("memory_id")
        if not memory_id:
            logger.warning("active projection skipped: record without id")
            return
        if self._outbox is not None:
            self._outbox.enqueue(
                ACTIVE_PROJECTION_TARGET,
                item_record,
                idempotency_key=f"memory:{memory_id}:active_projection",
                entity_type="memory",
                entity_id=str(memory_id),
            )
            return
        ActiveProjectionCommitter(
            self._store, self._resolver, logical_name=self._logical_name
        ).save_to_seekdb(item_record)

    def project_delete(self, memory_id: str) -> None:
        """Remove a memory from the ACTIVE collection (delete sync)."""
        if self._outbox is not None:
            self._outbox.enqueue(
                ACTIVE_PROJECTION_TARGET,
                {"id": memory_id, "status": "deleted"},
                idempotency_key=f"memory:{memory_id}:active_projection_delete",
                entity_type="memory",
                entity_id=str(memory_id),
            )
            return
        try:
            descriptor = ActiveCollectionResolver(self._store).resolve(self._logical_name)
        except ActiveIndexUnavailableError:
            return  # nothing to delete from
        client = getattr(self._store, "_client", None)
        if client is None:
            return
        client.get_collection(descriptor.physical_collection).delete(ids=[memory_id])
        _write_watermark(
            self._store,
            self._logical_name,
            last_memory_id=memory_id,
            last_event_time=None,
            projected_delta=1,
            failed_delta=0,
            note="delete",
        )


# ---------------------------------------------------------------------------
# Catch-up (rollback freshness, v4 §3.8)
# ---------------------------------------------------------------------------


def _collection_rows(store: Any, collection: str, *, page: int = 1000) -> dict[str, dict[str, Any]]:
    """id → metadata for every row in a physical collection (paginated)."""
    client = getattr(store, "_client", None)
    if client is None:
        raise RuntimeError("native store is not connected")
    coll = client.get_collection(collection)
    rows: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        result = coll.get(limit=page, offset=offset, include=["metadatas"])
        ids = list((result or {}).get("ids") or [])
        metadatas = list((result or {}).get("metadatas") or [])
        if not ids:
            return rows
        for record_id, metadata in zip(ids, metadatas, strict=False):
            rows[str(record_id)] = dict(metadata or {})
        offset += len(ids)


def catch_up_collection(
    store: Any,
    repository: Any,
    logical_name: str,
    target_collection: str,
    provider: Any,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Bring one physical collection up to the source-of-truth active set.

    Upserts missing and stale memories (``updated_at`` older than the source)
    with fresh embeddings, removes rows whose memory is no longer active,
    then verifies count + dimension.  Returns a report; raises on verify
    failure so a pointer switch cannot proceed on a stale collection.
    """
    started = time.time()
    items = repository.query(limit=500_000)  # source of truth: active only
    wanted = {item.memory_id: item for item in items}
    existing = _collection_rows(store, target_collection)

    missing = [item for memory_id, item in wanted.items() if memory_id not in existing]
    stale = []
    for memory_id, item in wanted.items():
        current = existing.get(memory_id)
        if current is None:
            continue
        current_updated = current.get("updated_at")
        try:
            current_updated = float(current_updated) if current_updated is not None else 0.0
        except (TypeError, ValueError):
            current_updated = 0.0
        if item.updated_at > current_updated + 1e-6:
            stale.append(item)
    extra = [memory_id for memory_id in existing if memory_id not in wanted]

    client = getattr(store, "_client", None)
    if client is None:
        raise RuntimeError("native store is not connected")
    coll = client.get_collection(target_collection)

    to_upsert = missing + stale
    for start in range(0, len(to_upsert), batch_size):
        batch = to_upsert[start : start + batch_size]
        records = [item.to_record() for item in batch]
        documents = [str(r.get("document") or r.get("title") or r["id"]) for r in records]
        embeddings = provider.encode_documents(documents)
        coll.upsert(
            ids=[str(r["id"]) for r in records],
            documents=documents,
            embeddings=embeddings,
            metadatas=[store._metadata(r) for r in records],
        )
    for start in range(0, len(extra), batch_size):
        coll.delete(ids=extra[start : start + batch_size])
    refresh = getattr(store, "refresh_index", None)
    if callable(refresh):
        refresh(target_collection)

    final_count = store.count(target_collection)
    dimension = store.embedding_info(target_collection).get("dimension")
    expected_dimension = provider.profile.dimension
    verified = final_count == len(wanted) and dimension == expected_dimension
    report = {
        "target_collection": target_collection,
        "source_active": len(wanted),
        "upserted_missing": len(missing),
        "upserted_stale": len(stale),
        "deleted_extra": len(extra),
        "final_count": final_count,
        "dimension": dimension,
        "expected_dimension": expected_dimension,
        "verified": verified,
        "elapsed_s": round(time.time() - started, 2),
    }
    last_id = to_upsert[-1].memory_id if to_upsert else None
    _write_watermark(
        store,
        logical_name,
        last_memory_id=last_id,
        last_event_time=None,
        projected_delta=len(to_upsert),
        failed_delta=0,
        note=f"catch_up:{target_collection}",
    )
    if not verified:
        raise RuntimeError(f"catch-up verification failed: {report}")
    return report


def descriptor_for_row(row: dict[str, Any], logical_name: str) -> ActiveIndexDescriptor:
    """Build an index descriptor from a registry build row (e.g. the OLD
    rollback target, whose profile may differ from the current ACTIVE)."""
    return ActiveIndexDescriptor(
        logical_name=logical_name,
        physical_collection=str(row["physical_collection"]),
        embedding_profile_id=str(row["embedding_profile_id"]),
        model_id=str(row["model_id"]),
        model_revision=str(row["model_revision"]),
        dimension=int(row["dimension"]),
        analyzer=str(row["analyzer"]),
        normalize=bool(row.get("normalize", True)),
        distance=str(row.get("distance") or "cosine"),
        activated_at=float(row.get("activated_at") or 0.0),
    )
