"""PR-SDB-2 versioned collection manager integration tests — REAL engine.

* build -> verify -> shadow -> activate -> rollback on embedded SeekDB
* physical collections never shared across profiles (§17.5)
* honest failures: activate without READY, rollback without OLD
* exact-entity hard body filter (左手 query -> left-body rows only)
* describe() exposes every truth field (§13)
"""

# ruff: noqa: E402 - imports follow pytest.importorskip by design

from __future__ import annotations

import contextlib
from dataclasses import replace

import pytest

pyseekdb = pytest.importorskip("pyseekdb")

from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore
from rosclaw.storage.versioned_collections import (
    ACTIVE,
    OLD,
    READY,
    VersionedCollectionManager,
    physical_name,
)
from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider

RECORDS = [
    {
        "id": "mem_l1",
        "body_id": "rh56_left_01",
        "memory_type": "failure",
        "joint_name": None,
        "failure_type": "joint_not_reached",
        "title": "left left_scissors failed: joint_not_reached",
        "document": "[ZH]\nRH56 左手执行剪刀时未到位\n\n[EN]\nleft hand scissors failed",
    },
    {
        "id": "mem_r1",
        "body_id": "rh56_right_01",
        "memory_type": "failure",
        "joint_name": None,
        "failure_type": "joint_not_reached",
        "title": "right scissors failed: joint_not_reached",
        "document": "[ZH]\nRH56 右手执行剪刀时未到位\n\n[EN]\nright hand scissors failed",
    },
]


@pytest.fixture()
def store(shared_embedded_seekdb_target):
    s = SeekDBEmbeddedStore(
        path=shared_embedded_seekdb_target["path"],
        database=shared_embedded_seekdb_target["database"],
    )
    s.connect()
    yield s
    s.disconnect()


@pytest.fixture()
def mgr(store):
    provider = FakeProvider(FAKE_PROFILE)
    manager = VersionedCollectionManager(store, provider)
    yield manager
    # cleanup: registry rows + physical collections
    try:
        names = [
            row.get("physical_collection")
            for logical in ("mem_v", "mem_x")
            for row in manager.registry(logical)
            if row.get("physical_collection")
        ]
        store.delete_where("projection_registry", {"logical_name": "mem_v"})
        store.delete_where("projection_registry", {"logical_name": "mem_x"})
        for name in names:
            with contextlib.suppress(Exception):
                store._client.delete_collection(name)
    except Exception:
        pass


def test_build_verify_activate_rollback(mgr):
    row = mgr.build("mem_v", RECORDS, analyzer="ngram")
    assert row["status"] == READY
    assert row["record_count"] == 2
    verify = mgr.verify("mem_v", analyzer="ngram")
    assert verify["ok"] and verify["actual"] == 2

    rows = mgr.shadow_query("mem_v", "剪刀 未到位", analyzer="ngram", limit=2)
    assert len(rows) == 2

    activated = mgr.activate("mem_v", analyzer="ngram")
    assert activated["status"] == ACTIVE
    assert (mgr.active("mem_v") or {}).get("physical_collection") == activated[
        "physical_collection"
    ]

    with pytest.raises(RuntimeError, match="no OLD build"):
        mgr.rollback("mem_v")


def test_activate_without_ready_build_fails(mgr):
    with pytest.raises(RuntimeError, match="no READY/OLD build"):
        mgr.activate("mem_x", analyzer="ngram")


def test_empty_build_is_refused(mgr):
    with pytest.raises(ValueError, match="empty versioned collection"):
        mgr.build("mem_x", [], analyzer="ngram")


def test_switch_between_generations_and_rollback(mgr, store):
    mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr.activate("mem_v", analyzer="ngram")
    mgr.build("mem_v", RECORDS, analyzer="ik")
    mgr.activate("mem_v", analyzer="ik")
    active = mgr.active("mem_v")
    assert active["analyzer"] == "ik"
    restored = mgr.rollback("mem_v")
    assert restored["analyzer"] == "ngram"
    assert (mgr.active("mem_v") or {}).get("analyzer") == "ngram"
    statuses = {r["analyzer"]: r["status"] for r in mgr.registry("mem_v")}
    assert statuses == {"ngram": ACTIVE, "ik": OLD}


def test_rollback_can_restore_previous_different_profile(mgr, store):
    first = mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr.activate("mem_v", analyzer="ngram")
    other_profile = replace(
        FAKE_PROFILE,
        profile_id="other_9d_v1",
        model_id="other/model",
        model_revision="rev2",
        dimension=9,
    )
    other = VersionedCollectionManager(store, FakeProvider(other_profile))
    second = other.build("mem_v", RECORDS, analyzer="ngram")
    other.activate("mem_v", analyzer="ngram")

    restored = other.rollback("mem_v")

    assert restored["physical_collection"] == first["physical_collection"]
    assert restored["embedding_profile_id"] == FAKE_PROFILE.profile_id
    assert second["physical_collection"] != restored["physical_collection"]


def test_rebuild_same_profile_never_mutates_active_collection(mgr, store):
    first = mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr.activate("mem_v", analyzer="ngram")
    second = mgr.build("mem_v", RECORDS, analyzer="ngram")

    assert first["physical_collection"] != second["physical_collection"]
    assert (mgr.active("mem_v") or {})["physical_collection"] == first["physical_collection"]
    assert store.count(first["physical_collection"]) == len(RECORDS)
    assert store.count(second["physical_collection"]) == len(RECORDS)

    mgr.activate("mem_v", analyzer="ngram")
    assert (mgr.active("mem_v") or {})["physical_collection"] == second["physical_collection"]
    mgr.rollback("mem_v")
    assert (mgr.active("mem_v") or {})["physical_collection"] == first["physical_collection"]
    mgr.activate("mem_v", analyzer="ngram")
    assert (mgr.active("mem_v") or {})["physical_collection"] == first["physical_collection"]


def test_verify_missing_collection_does_not_recreate_it(mgr, store):
    row = mgr.build("mem_v", RECORDS, analyzer="ngram")
    name = row["physical_collection"]
    store._client.delete_collection(name)

    result = mgr.verify("mem_v", analyzer="ngram")

    assert result["ok"] is False
    assert result["reason"] == "physical collection missing"
    assert not store._client.has_collection(name)


def test_physical_names_never_shared():
    a = physical_name("memory_items", FAKE_PROFILE, "ngram")
    assert a == "memory_items__fake_8d_v1__ngram"
    assert "384" not in a and "1024" not in a


def test_exact_hand_query_hard_filters_body(mgr):
    mgr.build("mem_v", RECORDS, analyzer="ngram")
    rows = mgr.shadow_query("mem_v", "左手 剪刀 未到位", analyzer="ngram", limit=3)
    assert rows, "expected rows after hard body filter"
    assert all(r.get("body_id") == "rh56_left_01" for r in rows)


def test_embedding_failure_falls_back_to_filtered_bm25(mgr):
    mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr._provider = FakeProvider(FAKE_PROFILE, fail=True)

    rows = mgr.shadow_query("mem_v", "左手 剪刀 未到位", analyzer="ngram", limit=3)

    assert rows
    assert all(row.get("body_id") == "rh56_left_01" for row in rows)


def test_describe_truth_fields(mgr):
    mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr.activate("mem_v", analyzer="ngram")
    desc = mgr.describe("mem_v")
    for field in (
        "backend",
        "active_collection",
        "analyzer",
        "vector_source",
        "score_semantics",
        "reranker",
        "fallback_state",
        "runtime_query_integration",
    ):
        assert field in desc
    emb = desc["embedding"]
    assert emb["model_revision"] == FAKE_PROFILE.model_revision
    assert emb["dimension"] == FAKE_PROFILE.dimension
    assert "similarity" not in desc["score_semantics"].lower() or "NOT" in desc["score_semantics"]


def test_describe_never_labels_active_collection_as_requested_other_profile(mgr, store):
    mgr.build("mem_v", RECORDS, analyzer="ngram")
    mgr.activate("mem_v", analyzer="ngram")
    other_profile = replace(
        FAKE_PROFILE,
        profile_id="other_9d_v1",
        model_id="other/model",
        model_revision="rev2",
        dimension=9,
    )
    other = VersionedCollectionManager(store, FakeProvider(other_profile))

    desc = other.describe("mem_v")

    assert desc["embedding"]["profile_id"] == FAKE_PROFILE.profile_id
    assert desc["embedding"]["model_id"] == FAKE_PROFILE.model_id
    assert desc["embedding"]["dimension"] == FAKE_PROFILE.dimension
    assert desc["requested_provider_profile_id"] == other_profile.profile_id
    assert desc["provider_matches_active"] is False
