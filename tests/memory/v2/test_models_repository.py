"""Tests for Memory 2.0 models, repository, and migration (PR-MEM-1)."""

from __future__ import annotations

import time

import pytest

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore, SQLiteKnowledgeStore
from rosclaw.memory.v2.models import (
    SCHEMA_VERSION,
    MemoryEvidence,
    MemoryItem,
    MemoryStatus,
    MemoryType,
)
from rosclaw.memory.v2.repository import MemoryRepository


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
