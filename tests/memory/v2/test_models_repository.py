"""Tests for Memory 2.0 models, repository, and migration (PR-MEM-1)."""

from __future__ import annotations

import time

import pytest

from rosclaw.memory.models import (
    SCHEMA_VERSION,
    MemoryEvidence,
    MemoryItem,
    MemoryStatus,
    MemoryType,
)
from rosclaw.memory.repository import MemoryRepository
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore, SQLiteKnowledgeStore


@pytest.fixture
def repo() -> MemoryRepository:
    client = InMemoryKnowledgeStore()
    client.connect()
    return MemoryRepository(client)


def _item(memory_type: str, **overrides) -> MemoryItem:
    base = {
        "robot_id": "rh56_rps_robot",
        "title": f"{memory_type} title",
        "document": f"{memory_type} document with evidence.",
        "evidence_refs": ["evt_001"],
    }
    base.update(overrides)
    return MemoryItem(memory_type=memory_type, **base)


def test_all_ten_memory_types_roundtrip(repo: MemoryRepository) -> None:
    """§5.9: every memory type has schema support and stores correctly."""
    for memory_type in MemoryType:
        item = _item(memory_type.value)
        memory_id = repo.store(item)
        loaded = repo.get(memory_id)
        assert loaded is not None
        assert loaded.memory_type == memory_type.value
        assert loaded.schema_version == SCHEMA_VERSION
        assert loaded.content_hash


def test_memory_is_traceable_to_evidence(repo: MemoryRepository) -> None:
    item = _item(
        "failure", evidence_refs=["evt_a", "evt_b"], artifact_refs=["artifact://mcap/x.mcap"]
    )
    memory_id = repo.store(item)
    trace = repo.trace(memory_id)
    assert trace["found"] and trace["traceable"]
    assert trace["evidence_count"] == 3  # 2 events + 1 mcap artifact
    source_ids = {ev["source_event_id"] for ev in trace["evidence"] if ev["source_event_id"]}
    assert source_ids == {"evt_a", "evt_b"}


def test_active_memory_without_evidence_is_rejected(repo: MemoryRepository) -> None:
    with pytest.raises(ValueError, match="has no evidence"):
        repo.store(_item("semantic", evidence_refs=[], artifact_refs=[]))
    assert repo.count() == 0


def test_store_is_idempotent_on_content_hash(repo: MemoryRepository) -> None:
    first = repo.store(_item("episodic"))
    second = repo.store(_item("episodic"))  # same content → same hash
    assert first == second
    assert repo.count({"robot_id": "rh56_rps_robot"}) == 1


def test_merge_appends_evidence_without_rewriting(repo: MemoryRepository) -> None:
    target = _item("skill", title="scissors success", evidence_refs=["evt_1"])
    target_id = repo.store(target)
    incoming = _item("skill", title="scissors success v2", evidence_refs=["evt_2"])
    assert repo.merge_into(target_id, incoming)
    merged = repo.get(target_id)
    assert merged is not None
    assert set(merged.evidence_refs) == {"evt_1", "evt_2"}
    assert merged.title == "scissors success"  # curated record preserved
    evidence = repo.evidence_for(target_id)
    assert len(evidence) == 2


def test_supersede_marks_old_and_stores_new(repo: MemoryRepository) -> None:
    old_id = repo.store(_item("body", title="thermal v1"))
    new_item = _item("body", title="thermal v2")
    new_id = repo.supersede(old_id, new_item)
    assert repo.get(old_id).status == MemoryStatus.SUPERSEDED.value  # type: ignore[union-attr]
    assert repo.get(new_id) is not None
    assert repo.get(new_id).metadata.get("supersedes") == old_id  # type: ignore[union-attr]


def test_supersede_carries_evidence_forward(repo: MemoryRepository) -> None:
    """The active record must hold the union of all evidence ever seen;
    otherwise redistilling an older session looks like 'new' evidence."""
    old_id = repo.store(_item("failure", title="left_rock failed", evidence_refs=["evt_old"]))
    new_item = _item("failure", title="left_rock failed again", evidence_refs=["evt_new"])
    new_id = repo.supersede(old_id, new_item)
    active = repo.get(new_id)
    assert active is not None
    assert set(active.evidence_refs) == {"evt_old", "evt_new"}


def test_pin_survives(repo: MemoryRepository) -> None:
    memory_id = repo.store(_item("body"))
    assert repo.pin(memory_id)
    assert repo.get(memory_id).pinned  # type: ignore[union-attr]


def test_evidence_model_roundtrip() -> None:
    ev = MemoryEvidence(
        memory_id="mem_1",
        evidence_type="telemetry_window",
        source_event_id="evt_9",
        sha256="abc",
    )
    restored = MemoryEvidence.from_record(ev.to_record())
    assert restored.memory_id == "mem_1"
    assert restored.evidence_type == "telemetry_window"
    assert restored.sha256 == "abc"


def test_numeric_zero_values_survive_roundtrip() -> None:
    item = _item(
        "episodic",
        confidence=0.0,
        importance=0.0,
        novelty=0.0,
        quality_score=0.0,
        event_time=0.0,
        created_at=0.0,
        updated_at=0.0,
    )
    restored = MemoryItem.from_record(item.to_record())
    assert restored.confidence == 0.0
    assert restored.importance == 0.0
    assert restored.novelty == 0.0
    assert restored.quality_score == 0.0
    assert restored.event_time == 0.0
    assert restored.created_at == 0.0
    assert restored.updated_at == 0.0

    evidence = MemoryEvidence(
        memory_id=item.memory_id,
        evidence_type="practice_event",
        confidence=0.0,
        created_at=0.0,
    )
    evidence_restored = MemoryEvidence.from_record(evidence.to_record())
    assert evidence_restored.confidence == 0.0
    assert evidence_restored.created_at == 0.0


def test_content_dedup_is_tenant_scoped(repo: MemoryRepository) -> None:
    alpha = _item("episodic", tenant_id="alpha")
    beta = _item("episodic", tenant_id="beta")
    assert alpha.content_hash != beta.content_hash
    alpha_id = repo.store(alpha)
    beta_id = repo.store(beta)
    assert alpha_id != beta_id
    assert repo.count() == 2


def test_store_deduplicates_pre_tenant_hash_within_scope(repo: MemoryRepository) -> None:
    old = _item("episodic", tenant_id="alpha")
    old.content_hash = old.compute_legacy_content_hash()
    repo._client.insert("memory_items", old.to_record())

    duplicate = _item("episodic", tenant_id="alpha")
    assert repo.store(duplicate) == old.memory_id

    other_tenant = _item("episodic", tenant_id="beta")
    assert repo.store(other_tenant) != old.memory_id


def test_sqlite_backend_stores_memory_items(tmp_path) -> None:
    """The new tables exist and work on the real SQLite backend too."""
    client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    client.connect()
    try:
        repo = MemoryRepository(client)
        memory_id = repo.store(_item("failure"))
        trace = repo.trace(memory_id)
        assert trace["traceable"]
        assert client.count("memory_items") == 1
        assert client.count("memory_evidence") == 1
    finally:
        client.disconnect()


def test_migrate_experience_graph_idempotent(repo: MemoryRepository) -> None:
    client = repo._client
    for i in range(5):
        client.insert(
            "experience_graph",
            {
                "id": f"exp_{i}",
                "event_type": "skill.invoke",
                "robot_id": "rh56_rps_robot",
                "timestamp": time.time() - i,
                "instruction": f"do thing {i % 2}",  # duplicate content on purpose
                "outcome": "failure" if i == 0 else "success",
            },
        )
    stats1 = repo.migrate_experience_graph()
    assert stats1["scanned"] == 5
    assert stats1["migrated"] + stats1["deduplicated"] == 5
    stats2 = repo.migrate_experience_graph()
    assert stats2["migrated"] == 0  # rerun is a pure no-op
    failures = repo.query({"memory_type": "failure"})
    assert len(failures) == 1
    assert failures[0].evidence_refs == ["exp_0"]
