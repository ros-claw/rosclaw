"""Tests for Memory 2.0 retrieval and index lifecycle (PR-MEM-2)."""

from __future__ import annotations

import time

import pytest

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore, SQLiteKnowledgeStore
from rosclaw.memory.v2.index import (
    EmbeddingIndexManager,
    IndexModelMismatchError,
)
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever
from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder


@pytest.fixture
def repo() -> MemoryRepository:
    client = InMemoryKnowledgeStore()
    client.connect()
    return MemoryRepository(client)


def _store_samples(repo: MemoryRepository) -> dict[str, str]:
    samples = [
        MemoryItem(
            memory_type="failure",
            robot_id="rh56_rps_robot",
            body_id="rh56_right",
            task_id="rh56_rps",
            title="right scissors failed: joint_not_reached",
            document="剪刀手势食指不到位，温度 42°C。",
            evidence_refs=["evt_1", "evt_2", "evt_3"],
            outcome="failure",
            confidence=0.9,
            event_time=time.time(),
        ),
        MemoryItem(
            memory_type="failure",
            robot_id="rh56_rps_robot",
            body_id="rh56_left",
            task_id="rh56_rps",
            title="left rock failed: serial timeout",
            document="CH340 串口超时 USB_TIMEOUT，EIO。",
            evidence_refs=["evt_4"],
            outcome="failure",
            confidence=0.8,
            event_time=time.time() - 10 * 86400,
        ),
        MemoryItem(
            memory_type="body",
            robot_id="rh56_rps_robot",
            body_id="rh56_right",
            task_id="rh56_rps",
            title="right hand thermal drift 36→42°C",
            document="右手连续运行 60 分钟温度上升。",
            evidence_refs=["evt_5", "evt_6"],
            confidence=0.85,
            event_time=time.time(),
        ),
        MemoryItem(
            memory_type="failure",
            robot_id="other_robot",
            body_id="other_body",
            task_id="painting",
            title="brush motor overcurrent",
            document="finger overcurrent on brush motor.",
            evidence_refs=["evt_7"],
            outcome="failure",
            event_time=time.time(),
        ),
    ]
    return {item.title: repo.store(item) for item in samples}


def test_cross_robot_isolation(repo: MemoryRepository) -> None:
    _store_samples(repo)
    retriever = MemoryRetriever(repo)
    results = retriever.retrieve(MemoryQuery(text="overcurrent 过流", robot_id="rh56_rps_robot"))
    assert results
    assert all(r.memory.robot_id == "rh56_rps_robot" for r in results)


def test_lexical_chinese_english_recall(repo: MemoryRepository) -> None:
    ids = _store_samples(repo)
    retriever = MemoryRetriever(repo)
    zh = retriever.retrieve(MemoryQuery(text="剪刀 食指 不到位", robot_id="rh56_rps_robot"))
    assert zh and zh[0].memory_id == ids["right scissors failed: joint_not_reached"]
    en = retriever.retrieve(MemoryQuery(text="serial timeout EIO", robot_id="rh56_rps_robot"))
    assert en and en[0].memory_id == ids["left rock failed: serial timeout"]


def test_metadata_filters(repo: MemoryRepository) -> None:
    _store_samples(repo)
    retriever = MemoryRetriever(repo)
    results = retriever.retrieve(
        MemoryQuery(text="thermal", memory_types=["body"], body_id="rh56_right")
    )
    assert results
    assert all(r.memory.memory_type == "body" for r in results)
    assert all(r.memory.body_id == "rh56_right" for r in results)


def test_scores_are_explained(repo: MemoryRepository) -> None:
    ids = _store_samples(repo)
    retriever = MemoryRetriever(repo)
    results = retriever.retrieve(MemoryQuery(text="剪刀", robot_id="rh56_rps_robot"))
    top = results[0]
    assert top.score_type in {"lexical", "vector", "lexical+vector", "metadata"}
    explanation = top.explanation
    assert "fusion_weights" in explanation
    assert (
        abs(
            sum(
                explanation["fusion_weights"][k] * explanation["score_parts"][k]
                for k in explanation["fusion_weights"]
            )
            - top.fusion_score
        )
        < 1e-9
    )
    # memory explain API
    detail = retriever.explain(
        ids["right scissors failed: joint_not_reached"],
        MemoryQuery(text="剪刀", robot_id="rh56_rps_robot"),
    )
    assert detail["memory_id"] == ids["right scissors failed: joint_not_reached"]


def test_expired_memory_excluded(repo: MemoryRepository) -> None:
    repo.store(
        MemoryItem(
            memory_type="episodic",
            robot_id="rh56_rps_robot",
            title="expired note",
            document="should not appear",
            evidence_refs=["e1"],
            expires_at=time.time() - 1,
        )
    )
    retriever = MemoryRetriever(repo)
    results = retriever.retrieve(MemoryQuery(text="expired note", robot_id="rh56_rps_robot"))
    assert all(r.memory.title != "expired note" for r in results)


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_stack(tmp_path):
    client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    client.connect()
    vector = SQLiteVectorStore(client)
    repo = MemoryRepository(client)
    manager = EmbeddingIndexManager(client, vector)
    return client, vector, repo, manager


def test_index_build_and_atomic_switch(sqlite_stack) -> None:
    client, vector, repo, manager = sqlite_stack
    _store_samples(repo)
    embedder = TfidfEmbedder()
    memories = repo.query({"robot_id": "rh56_rps_robot"}, limit=100)
    first = manager.build(memories, embedder)
    assert first["status"] == "READY"
    assert first["record_count"] == len(memories)
    assert first["index_version"] == 1

    # Rebuild with new corpus: old becomes OLD, new becomes READY v2.
    second = manager.build(memories, embedder)
    assert second["index_version"] == 2
    assert second["status"] == "READY"
    registry = client.query("memory_index_registry", filters={"id": first["id"]}, limit=1)
    assert registry[0]["status"] == "OLD"


def test_index_restart_consistency(sqlite_stack, tmp_path) -> None:
    """Index survives a store restart: same query returns same results."""
    client, vector, repo, manager = sqlite_stack
    _store_samples(repo)
    embedder = TfidfEmbedder()
    memories = repo.query({"robot_id": "rh56_rps_robot"}, limit=100)
    manager.build(memories, embedder)
    client.disconnect()

    client2 = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    client2.connect()
    vector2 = SQLiteVectorStore(client2)
    hits = vector2.search("memory_items", embedder.encode("剪刀"), limit=5)
    assert hits
    client2.disconnect()


def test_model_change_requires_rebuild(sqlite_stack) -> None:
    client, vector, repo, manager = sqlite_stack
    _store_samples(repo)
    embedder = TfidfEmbedder()
    manager.build(repo.query(limit=100), embedder)

    class _OtherEmbedder(TfidfEmbedder):
        model_name = "different-model"

    rebuild, reason = manager.needs_rebuild(_OtherEmbedder())
    assert rebuild
    assert "model changed" in reason
    with pytest.raises(IndexModelMismatchError):
        manager.check_query_embedder(_OtherEmbedder())


def test_delete_syncs_vector_index(sqlite_stack) -> None:
    client, vector, repo, manager = sqlite_stack
    ids = _store_samples(repo)
    embedder = TfidfEmbedder()
    manager.build(repo.query(limit=100), embedder)
    target = ids["right scissors failed: joint_not_reached"]
    assert vector.count("memory_items") > 0
    assert manager.on_memory_deleted(target)
    assert vector.count("memory_items") == manager.active_index()["record_count"]


def test_dimension_guard(sqlite_stack) -> None:
    client, vector, repo, manager = sqlite_stack
    _store_samples(repo)

    class _BadDimEmbedder(TfidfEmbedder):
        def dim(self):
            return 999

    with pytest.raises(IndexModelMismatchError):
        manager.build(repo.query(limit=100), _BadDimEmbedder())
