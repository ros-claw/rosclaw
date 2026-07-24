"""Incremental ACTIVE projection + rollback catch-up tests (PR-MEM-5, v4 §13)."""

from __future__ import annotations

import time

import pytest

from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.runtime_retrieval import EmbeddingProviderResolver
from rosclaw.storage.versioned_collections import VersionedCollectionManager
from rosclaw.storage.versioned_projection import (
    ACTIVE_PROJECTION_TARGET,
    ActiveProjection,
    ActiveProjectionCommitter,
    catch_up_collection,
    read_watermark,
)
from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider

pyseekdb = pytest.importorskip("pyseekdb", reason="native SeekDB engine not installed")


def _resolver(provider: FakeProvider | None = None) -> EmbeddingProviderResolver:
    return EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: provider or FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )


def _record(memory_id: str, title: str) -> dict:
    return {
        "id": memory_id,
        "memory_type": "failure",
        "robot_id": "r1",
        "body_id": "rh56_left_01",
        "joint_name": "middle",
        "failure_type": "joint_not_reached",
        "title": title,
        "document": f"{title} — document",
        "status": "active",
        "outcome": "failure",
        "event_time": time.time(),
        "updated_at": time.time(),
        "evidence_refs": ["evt_1"],
    }


def _item(memory_id: str, title: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        memory_type="failure",
        robot_id="r1",
        body_id="rh56_left_01",
        joint_name="middle",
        failure_type="joint_not_reached",
        title=title,
        document=f"{title} — document",
        outcome="failure",
        evidence_refs=["evt_1"],
    )


@pytest.fixture()
def store(shared_embedded_seekdb_target):
    from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore

    client = SeekDBEmbeddedStore(shared_embedded_seekdb_target["path"])
    client.connect()
    try:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        yield client
    finally:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        for name in client.list_collections():
            if name.startswith("memory_items__"):
                client._client.delete_collection(name)
        client.disconnect()


def _build_and_activate(store, records, logical="memory_items") -> str:
    # Seed the source of truth first: catch-up semantics only make sense
    # when the ACTIVE collection and memory_items start from the same set.
    for record in records:
        store.insert("memory_items", record)
    mgr = VersionedCollectionManager(store, FakeProvider(FAKE_PROFILE))
    mgr.build(logical, records, analyzer="ik")
    return str(mgr.activate(logical, analyzer="ik")["physical_collection"])


def test_active_projection_incremental_sync(store) -> None:
    """v4 §3.8: new memories, supersedes, and deletes all sync to ACTIVE."""
    physical = _build_and_activate(store, [_record("mem_a", "alpha failure")])
    repo = MemoryRepository(store, projection=ActiveProjection(store, _resolver()))

    # New memory → projected into ACTIVE.
    repo.store(_item("mem_b", "bravo failure"))
    assert store.count(physical) == 2

    # Supersede: old leaves ACTIVE, replacement enters (net count unchanged).
    repo.supersede("mem_b", _item("mem_b2", "bravo failure revised"))
    rows = store.query(physical, limit=10)
    ids = {row["id"] for row in rows}
    assert "mem_b" not in ids
    assert "mem_b2" in ids
    assert store.count(physical) == 2

    # Delete sync.
    ActiveProjection(store, _resolver()).project_delete("mem_b2")
    assert store.count(physical) == 1

    watermark = read_watermark(store, "memory_items")
    assert watermark is not None
    assert int(watermark["projected_count"]) >= 3
    assert int(watermark["failed_count"]) == 0


def test_projection_without_active_records_watermark(store) -> None:
    committer = ActiveProjectionCommitter(store, _resolver())
    committer.save_to_seekdb(_record("mem_orphan", "orphan"))
    watermark = read_watermark(store, "memory_items")
    assert watermark is not None
    assert int(watermark["failed_count"]) == 1
    assert "no_active" in (watermark.get("note") or "")


def test_projection_outbox_enqueue_path(store) -> None:
    class FakeOutbox:
        def __init__(self) -> None:
            self.calls = []

        def enqueue(self, target, payload, *, idempotency_key, entity_type, entity_id):
            self.calls.append((target, payload, idempotency_key, entity_type, entity_id))

    outbox = FakeOutbox()
    projection = ActiveProjection(store, _resolver(), outbox=outbox)
    projection.project(_record("mem_q", "queued"))
    projection.project_delete("mem_q")
    assert len(outbox.calls) == 2
    target, payload, key, entity_type, entity_id = outbox.calls[0]
    assert target == ACTIVE_PROJECTION_TARGET
    # The key carries the mutation marker so a later supersede is never
    # deduped away while the original projection is still pending.
    assert key.startswith("memory:mem_q:active_projection:")
    assert entity_type == "memory" and entity_id == "mem_q"
    assert outbox.calls[1][1]["status"] == "deleted"
    # Two different mutations of the same memory get distinct keys.
    projection.project(_record("mem_q", "queued v2"))
    keys = [call[2] for call in outbox.calls]
    assert len(set(keys)) == len(keys)


def test_catch_up_repairs_projection_lag(store) -> None:
    physical = _build_and_activate(store, [_record("mem_a", "alpha")])
    # Simulate a write whose projection failed: present in source of truth,
    # absent from ACTIVE.
    repo = MemoryRepository(store)  # no projection → ACTIVE stays behind
    repo.store(_item("mem_lag", "lagging memory"))
    assert store.count(physical) == 1

    report = catch_up_collection(store, repo, "memory_items", physical, FakeProvider(FAKE_PROFILE))
    assert report["verified"] is True
    assert report["upserted_missing"] == 1
    assert store.count(physical) == 2


def test_catch_up_removes_no_longer_active_rows(store) -> None:
    physical = _build_and_activate(store, [_record("mem_a", "alpha")])
    repo = MemoryRepository(store, projection=ActiveProjection(store, _resolver()))
    repo.store(_item("mem_b", "bravo"))
    assert store.count(physical) == 2
    # Direct engine-side delete bypasses the projection: ACTIVE still has it.
    store.delete("memory_items", "mem_b")
    report = catch_up_collection(store, repo, "memory_items", physical, FakeProvider(FAKE_PROFILE))
    assert report["deleted_extra"] == 1
    assert store.count(physical) == 1


def test_rollback_catches_up_before_switch(store) -> None:
    """v4 §3.8: OLD is never assumed current — catch-up precedes the switch."""
    records = [_record("mem_a", "alpha"), _record("mem_b", "bravo")]
    gen1 = _build_and_activate(store, records)
    gen2 = _build_and_activate(store, records)
    assert gen1 != gen2

    # A new memory projects only into the CURRENT ACTIVE (gen2).
    repo = MemoryRepository(store, projection=ActiveProjection(store, _resolver()))
    repo.store(_item("mem_new", "new after gen1"))
    assert store.count(gen2) == 3
    assert store.count(gen1) == 2  # OLD is behind

    mgr = VersionedCollectionManager(store, None)

    def _catch_up(target_row):
        catch_up_collection(
            store,
            repo,
            "memory_items",
            str(target_row["physical_collection"]),
            FakeProvider(FAKE_PROFILE),
        )

    restored = mgr.rollback("memory_items", catch_up=_catch_up)
    assert restored["physical_collection"] == gen1
    assert store.count(gen1) == 3  # caught up BEFORE the switch
    ids = {row["id"] for row in store.query(gen1, limit=10)}
    assert "mem_new" in ids


def test_rollback_aborts_when_catch_up_fails(store) -> None:
    records = [_record("mem_a", "alpha")]
    gen1 = _build_and_activate(store, records)
    gen2 = _build_and_activate(store, records)
    mgr = VersionedCollectionManager(store, None)

    def _failing_catch_up(target_row):
        raise RuntimeError("provider unavailable for catch-up")

    with pytest.raises(RuntimeError, match="catch-up"):
        mgr.rollback("memory_items", catch_up=_failing_catch_up)
    # Pointer unchanged: gen2 is still ACTIVE.
    active = mgr.active("memory_items")
    assert active["physical_collection"] == gen2
    assert gen1 != gen2


def test_in_batch_supersede_never_resurrects(store) -> None:
    """Review finding: a batch [upsert(m1), supersede(m1)] must end with m1
    ABSENT from ACTIVE (the old inline-delete-then-re-upsert order
    resurrected superseded memories)."""
    physical = _build_and_activate(store, [_record("mem_a", "alpha")])
    committer = ActiveProjectionCommitter(store, _resolver())
    committer.save_to_seekdb_batch(
        [
            _record("mem_m1", "first version"),
            {**_record("mem_m1", "first version"), "status": "superseded"},
        ]
    )
    ids = {row["id"] for row in store.query(physical, limit=10)}
    assert "mem_m1" not in ids
    assert store.count(physical) == 1


def test_shared_resolver_reused_across_projections(store) -> None:
    """Review finding: default projections must not rebuild the provider
    per write — one process-level resolver is shared."""
    from rosclaw.storage.versioned_projection import shared_projection_resolver

    first = shared_projection_resolver()
    second = shared_projection_resolver()
    assert first is second
    committer_a = ActiveProjectionCommitter(store)
    committer_b = ActiveProjectionCommitter(store)
    assert committer_a._resolver is committer_b._resolver
