"""Tests for embedder warmup from existing knowledge tables."""

from __future__ import annotations

import pytest

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.storage.vector import TfidfEmbedder


@pytest.fixture
def vector_client(tmp_path):
    db_path = tmp_path / "warmup.sqlite"
    client = SQLiteKnowledgeStore(db_path=str(db_path), vector_enabled=True)
    client.connect()
    return client


def test_warmup_indexes_existing_knowledge_rows(vector_client):
    # Seed knowledge_graph before vector store is initialized.
    vector_client.insert(
        "knowledge_graph",
        {
            "id": "kg1",
            "subject": "ur5e",
            "predicate": "has_capability",
            "object": "pick_and_place",
            "confidence": 1.0,
            "timestamp": 1.0,
        },
    )
    vector_client.insert(
        "knowledge_graph",
        {
            "id": "kg2",
            "subject": "panda",
            "predicate": "has_capability",
            "object": "force_control",
            "confidence": 1.0,
            "timestamp": 2.0,
        },
    )

    counts = vector_client.warmup_embedder()
    assert counts.get("knowledge_graph") == 2

    vector_store = vector_client._vector_store
    rows = vector_store._all_rows("knowledge_graph")
    ids = {r["record_id"] for r in rows}
    assert ids == {"kg1", "kg2"}


def test_warmup_fits_tfidf_on_existing_rows(vector_client):
    vector_client.insert(
        "experience_graph",
        {
            "id": "exp1",
            "event_type": "praxis",
            "robot_id": "r1",
            "timestamp": 1.0,
            "instruction": "grasp the red cup",
            "outcome": "success",
        },
    )
    vector_client.insert(
        "experience_graph",
        {
            "id": "exp2",
            "event_type": "praxis",
            "robot_id": "r1",
            "timestamp": 2.0,
            "instruction": "place the blue book",
            "outcome": "success",
        },
    )

    counts = vector_client.warmup_embedder(tables=["experience_graph"])
    assert counts.get("experience_graph") == 2
    assert isinstance(vector_client._embedder, TfidfEmbedder)
    assert vector_client._embedder._vocab is not None
    assert len(vector_client._embedder._vocab) > 0


def test_warmup_is_no_op_when_vector_disabled(tmp_path):
    client = SQLiteKnowledgeStore(db_path=str(tmp_path / "no_vec.sqlite"), vector_enabled=False)
    client.connect()
    try:
        counts = client.warmup_embedder()
        assert counts == {}
        assert client._embedder is None
    finally:
        client.disconnect()


def test_warmup_skips_missing_tables(vector_client):
    counts = vector_client.warmup_embedder(tables=["nonexistent_table"])
    assert counts.get("nonexistent_table") is None
    assert vector_client._embedder_warmed is True


def test_vector_query_after_warmup_finds_historical_rows(vector_client):
    vector_client.insert(
        "heuristic_rules",
        {
            "id": "hr1",
            "condition": "gripper slip while lifting",
            "action": "increase grip force",
            "priority": 2,
        },
    )

    vector_client.warmup_embedder(tables=["heuristic_rules"])
    results = vector_client.similar("heuristic_rules", "gripper slipped during lift", limit=5)
    assert any(r["id"] == "hr1" for r in results)
