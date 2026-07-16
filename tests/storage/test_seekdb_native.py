"""Real native SeekDB tests (PR-SDB-1) — no mocks, real embedded engine.

Skipped when pyseekdb is not installed (optional dependency).
"""
# ruff: noqa: E402 - imports follow pytest.importorskip by design

from __future__ import annotations

import time

import pytest

pyseekdb = pytest.importorskip(  # noqa: E402
    "pyseekdb", reason="pyseekdb optional dependency not installed"
)

from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore
from rosclaw.storage.seekdb_projection import SeekDBProjection

TEST_PATH = "/tmp/seekdb_native_tests"


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "seekdb"
    store = SeekDBEmbeddedStore(path=str(path), database="test_db")
    store.connect()
    yield store
    store.disconnect()


def _sample_records() -> list[dict]:
    return [
        {
            "id": f"mem_{i:03d}",
            "memory_type": "failure",
            "robot_id": "rh56_rps_robot",
            "body_id": "rh56_right",
            "task_id": "rh56_rps",
            "title": f"failure case {i}: scissors overcurrent",
            "document": f"剪刀手势失败 {i}，食指过流 overcurrent，温度 {40 + i % 5}°C。",
            "status": "active",
            "event_time": time.time(),
        }
        for i in range(20)
    ] + [
        {
            "id": "mem_other",
            "memory_type": "failure",
            "robot_id": "other_robot",
            "body_id": "other_body",
            "task_id": "painting",
            "title": "other robot scissors failure",
            "document": "scissors failure on another robot.",
            "status": "active",
            "event_time": time.time(),
        }
    ]


def test_crud_roundtrip(store: SeekDBEmbeddedStore) -> None:
    records = _sample_records()
    for record in records:
        store.insert("memory_items", record)
    assert store.count("memory_items") == len(records)
    rh56 = store.query("memory_items", {"robot_id": "rh56_rps_robot"}, limit=50)
    assert len(rh56) == 20
    assert store.update("memory_items", "mem_000", {"status": "superseded"})
    row = store.query("memory_items", {"id": "mem_000"}, limit=1)[0]
    assert row["status"] == "superseded"
    assert store.delete("memory_items", "mem_000")
    assert store.count("memory_items") == len(records) - 1


def test_native_vector_search(store: SeekDBEmbeddedStore) -> None:
    store.insert_many("memory_items", _sample_records())
    hits = store.similar("memory_items", "剪刀 过流", limit=5)
    assert hits, "native vector search returned nothing"
    assert all(hit["robot_id"] in {"rh56_rps_robot", "other_robot"} for hit in hits)
    # Metadata pre-filter isolates robots.
    rh56_hits = store.similar(
        "memory_items", "scissors failure", filters={"robot_id": "rh56_rps_robot"}, limit=5
    )
    assert rh56_hits
    assert all(hit["robot_id"] == "rh56_rps_robot" for hit in rh56_hits)


def test_native_hybrid_search(store: SeekDBEmbeddedStore) -> None:
    store.insert_many("memory_items", _sample_records())
    hits = store.hybrid_search("memory_items", "剪刀", limit=5)
    assert hits, "native hybrid search returned nothing"


def test_restart_persistence(tmp_path) -> None:
    path = tmp_path / "seekdb"
    store = SeekDBEmbeddedStore(path=str(path), database="persist_db")
    store.connect()
    store.insert_many("memory_items", _sample_records())
    store.disconnect()

    store2 = SeekDBEmbeddedStore(path=str(path), database="persist_db")
    store2.connect()
    assert store2.count("memory_items") == len(_sample_records())
    hits = store2.similar("memory_items", "剪刀", limit=3)
    assert hits
    store2.disconnect()


def test_embedding_info(store: SeekDBEmbeddedStore) -> None:
    info = store.embedding_info("memory_items")
    assert info["dimension"] == 384
    assert info["model_name"] == "all-MiniLM-L6-v2"


def test_projection_rebuild(tmp_path) -> None:
    from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore

    sqlite_client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    sqlite_client.connect()
    repo = MemoryRepository(sqlite_client)
    for record in _sample_records()[:10]:
        repo.store(
            MemoryItem(
                memory_type=record["memory_type"],
                robot_id=record["robot_id"],
                title=record["title"],
                document=record["document"],
                evidence_refs=["evt_1"],
            )
        )

    store = SeekDBEmbeddedStore(path=str(tmp_path / "seekdb"), database="proj_db")
    projection = SeekDBProjection(store)
    result = projection.rebuild(repo)
    assert result["rebuilt"] == 10
    hits = store.similar("memory_items", "剪刀 过流", limit=5)
    assert hits
    sqlite_client.disconnect()


def test_repository_dual_write_via_projection(tmp_path) -> None:
    from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore

    sqlite_client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    sqlite_client.connect()
    store = SeekDBEmbeddedStore(path=str(tmp_path / "seekdb"), database="dual_db")
    projection = SeekDBProjection(store)
    repo = MemoryRepository(sqlite_client, projection=projection)

    item = MemoryItem(
        memory_type="failure",
        robot_id="rh56_rps_robot",
        title="dual write test failure",
        document="双手剪刀失败 dual write。",
        evidence_refs=["evt_1"],
    )
    repo.store(item)
    store.refresh_index("memory_items")
    assert store.count("memory_items") == 1
    sqlite_client.disconnect()


def test_native_benchmark_vs_sqlite_scan(tmp_path) -> None:
    """Native HNSW vector search must beat SQLite full-table scan at 10k records."""
    from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
    from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder

    base = _sample_records()
    records = [
        {
            **record,
            "id": f"{record['id']}_{copy}",
            "document": f"{record['document']} variant {copy}",
        }
        for copy in range(480)
        for record in base
    ]  # 10,080 records
    # Native SeekDB (HNSW index)
    store = SeekDBEmbeddedStore(path=str(tmp_path / "seekdb"), database="bench_db")
    store.connect()
    for offset in range(0, len(records), 1000):
        store.insert_many("memory_items", records[offset : offset + 1000])
    t0 = time.perf_counter()
    for _ in range(10):
        store.similar("memory_items", "剪刀 过流", limit=5)
    native_ms = (time.perf_counter() - t0) / 10 * 1000

    # SQLite TF-IDF fallback (full table scan)
    sqlite_client = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    sqlite_client.connect()
    vector = SQLiteVectorStore(sqlite_client)
    embedder = TfidfEmbedder().fit([r["document"] for r in records])
    vector.upsert_many(
        "memory_items", [(r["id"], r["document"], embedder.encode(r["document"])) for r in records]
    )
    query_vec = embedder.encode("剪刀 过流")
    t0 = time.perf_counter()
    for _ in range(10):
        vector.search("memory_items", query_vec, limit=5)
    sqlite_ms = (time.perf_counter() - t0) / 10 * 1000

    print(
        f"\nnative hnsw: {native_ms:.1f}ms  sqlite tfidf scan: {sqlite_ms:.1f}ms  ({len(records)} records)"
    )
    assert native_ms < sqlite_ms, f"native {native_ms}ms not faster than sqlite {sqlite_ms}ms"
    sqlite_client.disconnect()
