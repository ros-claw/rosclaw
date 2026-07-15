"""Tests for MemoryInterface vector/hybrid search integration."""

from __future__ import annotations

import pytest

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore


@pytest.fixture
def vector_client(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    client = SQLiteKnowledgeStore(db_path=str(db_path), vector_enabled=True)
    client.connect()
    return client


def test_vector_backend_is_used_when_available(vector_client):
    mem = MemoryInterface("ur5e_lab_01", seekdb_client=vector_client)
    mem.initialize()
    try:
        mem.store_experience(
            event_id="exp1",
            event_type="praxis",
            instruction="pick up the red cup",
            outcome="success",
            tags=["grasp", "cup"],
        )
        mem.store_experience(
            event_id="exp2",
            event_type="praxis",
            instruction="place the book on the shelf",
            outcome="success",
            tags=["place", "book"],
        )
        results = mem.find_similar_experiences("grasp a red cup", limit=2)
        assert len(results) >= 1
        # The vector path should surface the cup experience first.
        assert results[0]["id"] == "exp1"
        assert "vector_score" in results[0]
    finally:
        mem.stop()
        vector_client.disconnect()


def test_keyword_fallback_when_vector_disabled():
    mem = MemoryInterface("ur5e_lab_01")
    mem.initialize()
    try:
        mem.store_experience(
            event_id="exp1",
            event_type="praxis",
            instruction="pick up the red cup",
            outcome="success",
            tags=["grasp", "cup"],
        )
        results = mem.find_similar_experiences("red cup", limit=2)
        assert len(results) >= 1
        assert results[0]["id"] == "exp1"
        # No vector_score because InMemoryKnowledgeStore lacks vector search.
        assert "vector_score" not in results[0]
    finally:
        mem.stop()


def test_experience_is_indexed_after_store(vector_client):
    mem = MemoryInterface("ur5e_lab_01", seekdb_client=vector_client)
    mem.initialize()
    try:
        mem.store_experience(
            event_id="exp3",
            event_type="praxis",
            instruction="slide the box to the left",
            outcome="failure",
            error_details="box tipped over",
            tags=["slide", "box"],
        )
        vector_store = vector_client._vector_store
        rows = vector_store._all_rows("experience_graph")
        assert any(r["record_id"] == "exp3" for r in rows)
    finally:
        mem.stop()
        vector_client.disconnect()
