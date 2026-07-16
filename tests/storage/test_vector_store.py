"""Tests for rosclaw.storage.vector."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.storage.vector import (
    SQLiteVectorStore,
    TfidfEmbedder,
    _cosine_similarity,
    _rrf_fusion,
    _tokenize,
)


def test_tokenize_lower_alphanumeric() -> None:
    assert _tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tfidf_embedder_basic() -> None:
    embedder = TfidfEmbedder(max_features=100)
    embedder.fit(["hello world", "hello there"])
    vec = embedder.encode("hello")
    assert len(vec) == len(embedder._vocab or {})
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_tfidf_auto_fit() -> None:
    embedder = TfidfEmbedder()
    vec = embedder.encode("robot grasp")
    assert len(vec) > 0
    assert embedder.dim is not None


def test_cosine_similarity_identical() -> None:
    v = [1.0, 2.0, 3.0]
    assert np.isclose(_cosine_similarity(v, v), 1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_rrf_fusion_orders() -> None:
    vector_rank = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    keyword_rank = [("b", 1.0), ("c", 0.9), ("d", 0.8)]
    fused = _rrf_fusion(vector_rank, keyword_rank)
    ids = [r[0] for r in fused]
    assert "b" in ids[:2]
    assert "a" in ids
    assert "d" in ids


def test_sqlite_vector_store_upsert_and_search(tmp_path) -> None:
    store = SQLiteKnowledgeStore(db_path=str(tmp_path / "knowledge.sqlite"))
    store.connect()
    vector_store = SQLiteVectorStore(store)

    embedder = TfidfEmbedder(max_features=100)
    embedder.fit(
        [
            "pick red cube",
            "place red cube on table",
            "push blue cylinder",
        ]
    )

    vector_store.upsert("skills", "s1", "pick red cube", embedder.encode("pick red cube"))
    vector_store.upsert(
        "skills", "s2", "place red cube on table", embedder.encode("place red cube on table")
    )
    vector_store.upsert("skills", "s3", "push blue cylinder", embedder.encode("push blue cylinder"))

    query = embedder.encode("grab red cube")
    results = vector_store.search("skills", query, limit=2)
    ids = [r["id"] for r in results]
    assert "s1" in ids
    assert len(results) <= 2


def test_sqlite_hybrid_search(tmp_path) -> None:
    embedder = TfidfEmbedder(max_features=100)
    embedder.fit(
        [
            "pick red cube",
            "place red cube on table",
            "push blue cylinder",
        ]
    )
    store = SQLiteKnowledgeStore(
        db_path=str(tmp_path / "knowledge.sqlite"),
        vector_enabled=True,
        embedder=embedder,
    )
    store.connect()
    store._ensure_vector_store()

    store._vector_store.upsert("skills", "s1", "pick red cube", embedder.encode("pick red cube"))
    store._vector_store.upsert(
        "skills", "s2", "place red cube on table", embedder.encode("place red cube on table")
    )
    store._vector_store.upsert(
        "skills", "s3", "push blue cylinder", embedder.encode("push blue cylinder")
    )

    results = store.similar("skills", "grab red cube", limit=2)
    ids = [r["id"] for r in results]
    assert "s1" in ids
    assert all("score" in r for r in results)


def test_sqlite_similar_disabled_fallback(tmp_path) -> None:
    store = SQLiteKnowledgeStore(db_path=str(tmp_path / "knowledge.sqlite"), vector_enabled=False)
    store.connect()
    store.insert("skills", {"id": "s1", "name": "pick red cube", "description": "grasp red cube"})
    store.insert("skills", {"id": "s2", "name": "push blue cylinder"})
    results = store.similar("skills", "red cube grab", limit=2)
    ids = [r["id"] for r in results]
    assert "s1" in ids
    assert all("score" in r for r in results)


def test_vector_store_update_existing_record(tmp_path) -> None:
    store = SQLiteKnowledgeStore(db_path=str(tmp_path / "knowledge.sqlite"))
    store.connect()
    vector_store = SQLiteVectorStore(store)
    vector_store.upsert("skills", "s1", "old text", [1.0, 0.0])
    vector_store.upsert("skills", "s1", "new text", [0.0, 1.0])
    results = vector_store.search("skills", [0.0, 1.0], limit=1)
    assert results[0]["text"] == "new text"


def test_empty_query_returns_empty() -> None:
    store = SQLiteKnowledgeStore(db_path=":memory:")
    store.connect()
    results = store.similar("skills", "", limit=5)
    assert results == []


def test_warmup_embedder_indexes_existing_rows(tmp_path) -> None:
    store = SQLiteKnowledgeStore(db_path=str(tmp_path / "knowledge.sqlite"), vector_enabled=True)
    store.connect()
    store.insert(
        "experience_graph",
        {
            "id": "exp1",
            "event_type": "praxis",
            "robot_id": "r1",
            "timestamp": 1.0,
            "instruction": "pick red cube",
            "outcome": "success",
        },
    )
    store.insert(
        "experience_graph",
        {
            "id": "exp2",
            "event_type": "praxis",
            "robot_id": "r1",
            "timestamp": 2.0,
            "instruction": "place red cube on table",
            "outcome": "success",
        },
    )
    counts = store.warmup_embedder(tables=["experience_graph"])
    assert counts.get("experience_graph") == 2

    results = store.similar("experience_graph", "grab red cube", limit=2)
    ids = [r["id"] for r in results]
    assert "exp1" in ids


def test_warmup_embedder_is_idempotent(tmp_path) -> None:
    store = SQLiteKnowledgeStore(db_path=str(tmp_path / "knowledge.sqlite"), vector_enabled=True)
    store.connect()
    store.insert(
        "experience_graph",
        {
            "id": "exp1",
            "event_type": "praxis",
            "robot_id": "r1",
            "timestamp": 1.0,
            "instruction": "pick red cube",
        },
    )
    counts1 = store.warmup_embedder(tables=["experience_graph"])
    assert counts1.get("experience_graph") == 1
    counts2 = store.warmup_embedder(tables=["experience_graph"])
    assert counts2 == {}
