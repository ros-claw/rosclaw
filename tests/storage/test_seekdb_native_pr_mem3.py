"""PR-MEM-3 native SeekDB integration tests (数据库优化v3 §13) — REAL engine.

* both hybrid legs apply the same metadata filter (cross-robot/body
  leakage must be 0 even when the BM25 leg alone would match)
* order_by raises UnsupportedOperationError (no fake global ordering)
* count(filters) and delete_where paginate without a 100k cap
* exception narrowing: only a genuine not-found creates a collection
* collection restart persistence (embedded)
"""

# ruff: noqa: E402 - imports follow pytest.importorskip by design

from __future__ import annotations

import contextlib

import pytest

pyseekdb = pytest.importorskip("pyseekdb")

from rosclaw.storage.seekdb_native import (
    SeekDBEmbeddedStore,
    UnsupportedOperationError,
)

_CLEAN_TABLES = ("memory_items", "bulk", "t")


@pytest.fixture()
def store(shared_embedded_seekdb_target):
    # One embedded target per process — shared via tests/conftest.py
    # (same pattern as test_seekdb_native.py).
    s = SeekDBEmbeddedStore(
        path=shared_embedded_seekdb_target["path"],
        database=shared_embedded_seekdb_target["database"],
    )
    s.connect()
    for table in _CLEAN_TABLES:
        with contextlib.suppress(Exception):
            s.delete_where(table, {})
    try:
        yield s
    finally:
        if not s.is_connected():
            s.connect()
        for table in _CLEAN_TABLES:
            with contextlib.suppress(Exception):
                s.delete_where(table, {})
        s.disconnect()


def _seed_leakage_corpus(s):
    """Left memory matches the text query on BOTH legs; right memory only
    matches BM25 (it names the same error) — pre-fix it leaked through."""
    s.insert(
        "memory_items",
        {
            "id": "mem_left",
            "robot_id": "rh56_left",
            "body_id": "rh56_left_01",
            "title": "left thumb_rot joint_not_reached",
            "document": "left hand thumb_rot failed joint_not_reached 左手拇指根未到位",
            "memory_type": "failure",
        },
    )
    s.insert(
        "memory_items",
        {
            "id": "mem_right",
            "robot_id": "rh56_right",
            "body_id": "rh56_right_01",
            "title": "right thumb joint_not_reached",
            "document": "right hand thumb failed joint_not_reached 右手拇指未到位",
            "memory_type": "failure",
        },
    )
    s.refresh_index("memory_items")


def test_hybrid_filter_applies_to_bm25_and_vector(store):
    _seed_leakage_corpus(store)
    rows = store.hybrid_search(
        "memory_items",
        "joint_not_reached 未到位",
        filters={"robot_id": "rh56_left"},
        limit=5,
    )
    ids = {r["id"] for r in rows}
    assert ids == {"mem_left"}, f"cross-robot leakage through a hybrid leg: {ids}"


def test_left_hand_query_never_returns_right_hand_memory(store):
    _seed_leakage_corpus(store)
    rows = store.hybrid_search(
        "memory_items",
        "thumb 拇指 未到位",
        filters={"body_id": "rh56_left_01"},
        limit=5,
    )
    assert all(r.get("body_id") == "rh56_left_01" for r in rows)
    assert {r["id"] for r in rows} == {"mem_left"}


def test_robot_a_query_never_returns_robot_b_memory(store):
    _seed_leakage_corpus(store)
    rows = store.hybrid_search(
        "memory_items",
        "joint_not_reached",
        filters={"robot_id": "rh56_right"},
        limit=5,
    )
    assert {r["id"] for r in rows} == {"mem_right"}


def test_unfiltered_hybrid_still_returns_both(store):
    _seed_leakage_corpus(store)
    rows = store.hybrid_search("memory_items", "joint_not_reached 未到位", limit=5)
    assert {r["id"] for r in rows} == {"mem_left", "mem_right"}


def test_order_by_raises(store):
    store.insert("t", {"id": "a", "ts": 1})
    with pytest.raises(UnsupportedOperationError):
        store.query("t", order_by="-ts", limit=1)


def test_count_and_delete_where_paginate(store):
    for i in range(2300):
        store.insert("bulk", {"id": f"r{i:04d}", "grp": "x" if i % 2 else "y"})
    assert store.count("bulk", {"grp": "x"}) == 1150
    deleted = store.delete_where("bulk", {"grp": "x"})
    assert deleted == 1150
    assert store.count("bulk") == 1150
    assert store.count("bulk", {"grp": "x"}) == 0


def test_exception_narrowing_classifier():
    # The observed pyseekdb message shapes (real engine, this rig):
    from rosclaw.storage.seekdb_native import _is_collection_not_found

    assert _is_collection_not_found(
        RuntimeError("execute sql failed OB_TABLE_NOT_EXIST(1146): Table '%s.%s' doesn't exist")
    )
    assert _is_collection_not_found(
        ValueError("Collection ('t') not found: Table('c$v1$t') not exists")
    )
    assert _is_collection_not_found(ValueError("Collection 't' does not exist"))
    # Unknown DATABASE (1049) is NOT a missing collection:
    assert not _is_collection_not_found(
        RuntimeError("connect failed OB_ERR_BAD_DATABASE(1049): Unknown database '%.*s'")
    )
    # Auth/permission/network must never look like a missing collection:
    assert not _is_collection_not_found(RuntimeError("access denied for user 'root'"))
    assert not _is_collection_not_found(OSError("connection refused"))


def test_unknown_database_surfaces_not_creates(shared_embedded_seekdb_target):
    # Point a client at a database that does not exist: the engine's
    # 1049 must surface, never be mistaken for a missing collection.
    import pyseekdb

    client = pyseekdb.Client(
        path=shared_embedded_seekdb_target["path"],
        database="definitely_not_a_database_v3",
    )
    with pytest.raises(Exception) as excinfo:
        client.get_collection("anything")
    assert "1049" in str(excinfo.value) or "Unknown database" in str(excinfo.value)


def test_collection_restart_persistence(shared_embedded_seekdb_target):
    # Same target, fresh client instance (the only embedded "restart"
    # pylibseekdb allows in one process — SDB-01 acceptance pattern).
    path = shared_embedded_seekdb_target["path"]
    db = shared_embedded_seekdb_target["database"]
    s1 = SeekDBEmbeddedStore(path=path, database=db)
    s1.connect()
    s1.insert("memory_items", {"id": "persist1", "document": "重启持久化 persistence"})
    s1.disconnect()
    s2 = SeekDBEmbeddedStore(path=path, database=db)
    s2.connect()
    rows = s2.query("memory_items", {"id": "persist1"}, limit=1)
    assert len(rows) == 1 and rows[0]["id"] == "persist1"
    s2.disconnect()


def test_query_embeddings_manual_vector(store):
    _seed_leakage_corpus(store)
    # A manual embedding vector must drive the KNN leg (§8.1); a zero
    # vector of the collection's dimension is accepted and still respects
    # the hard filter.
    dim = store.embedding_info("memory_items").get("dimension") or 384
    rows = store.hybrid_search(
        "memory_items",
        "joint_not_reached",
        filters={"robot_id": "rh56_left"},
        limit=3,
        query_embedding=[0.0] * dim,
    )
    assert {r["id"] for r in rows} == {"mem_left"}
