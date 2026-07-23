"""Real-engine integration for the runtime retrieval facade (PR-MEM-5, v4 §13).

Runs against the shared embedded SeekDB engine: build → activate → serve.
"""

from __future__ import annotations

import argparse
import json

import pytest

from rosclaw.memory.v2.cli import _close, _open_stack, cmd_memory_v2_query
from rosclaw.memory.v2.retrieval import MemoryQuery
from rosclaw.memory.v2.runtime_retrieval import (
    MODE_ACTIVE_BM25,
    EmbeddingProviderResolver,
    RetrievalPurpose,
    build_retrieval_facade,
)
from rosclaw.memory.v2.runtime_retrieval.fallback import hybrid_mode_for
from rosclaw.storage.versioned_collections import VersionedCollectionManager
from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider

pyseekdb = pytest.importorskip("pyseekdb", reason="native SeekDB engine not installed")


def _stack_args(path: str) -> argparse.Namespace:
    return argparse.Namespace(v2_path=path, backend="seekdb_embedded", seekdb_url=None)


def _records() -> list[dict]:
    base = {
        "robot_id": "r1",
        "memory_type": "failure",
        "status": "active",
        "outcome": "failure",
        "failure_type": "joint_not_reached",
    }
    return [
        {
            **base,
            "id": "mem_it_left_middle",
            "body_id": "rh56_left_01",
            "joint_name": "middle",
            "title": "左手 middle joint_not_reached",
            "document": "左手 middle 未达到目标位置，joint_not_reached，热退化工况",
        },
        {
            **base,
            "id": "mem_it_right_middle",
            "body_id": "rh56_right_01",
            "joint_name": "middle",
            "title": "右手 middle joint_not_reached",
            "document": "右手 middle 未达到目标位置，joint_not_reached，健康工况",
        },
    ]


@pytest.fixture()
def active_collection(shared_embedded_seekdb_target):
    path = shared_embedded_seekdb_target["path"]
    client, _, _, _, _ = _open_stack(_stack_args(path), with_vector=False)
    try:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        mgr = VersionedCollectionManager(client, FakeProvider(FAKE_PROFILE))
        mgr.build("memory_items", _records(), analyzer="ik")
        activated = mgr.activate("memory_items", analyzer="ik")
        yield client, str(activated["physical_collection"])
    finally:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        for name in client.list_collections():
            if name.startswith("memory_items__fake_8d_v1__"):
                client._client.delete_collection(name)
        _close(client)


def test_facade_hybrid_on_real_engine(active_collection) -> None:
    client, physical = active_collection
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )
    facade = build_retrieval_facade(native_store=client, provider_resolver=resolver)
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert response.retrieval_mode == hybrid_mode_for(FAKE_PROFILE.profile_id)
    assert response.fallback is False
    assert response.physical_collection == physical
    assert response.candidates
    top = response.candidates[0]
    assert top.memory_id == "mem_it_left_middle"
    assert top.body_match is True
    assert top.joint_match is True
    # The wrong-body memory is demoted or filtered out of top-1, never boosted.
    right = [c for c in response.candidates if c.memory_id == "mem_it_right_middle"]
    assert not right or right[0].fusion_rank > top.fusion_rank


def test_how_purpose_abstains_cross_body_on_real_engine(active_collection) -> None:
    client, physical = active_collection
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )
    # Only right-body memories remain in the ACTIVE physical collection.
    client.delete(physical, "mem_it_left_middle")
    client.refresh_index(physical)
    facade = build_retrieval_facade(native_store=client, provider_resolver=resolver)
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HOW_INTERVENTION,
    )
    left = [c for c in response.candidates if c.memory_id == "mem_it_left_middle"]
    cross = [c for c in response.candidates if c.body_match is False]
    assert left == []
    assert cross == []


def test_cli_and_facade_topk_match(shared_embedded_seekdb_target, capsys) -> None:
    """v4 §13: Runtime facade and CLI query --v2 return the same Top-K."""
    path = shared_embedded_seekdb_target["path"]
    client, _, _, _, _ = _open_stack(_stack_args(path), with_vector=False)
    try:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        mgr = VersionedCollectionManager(client, FakeProvider(FAKE_PROFILE))
        mgr.build("memory_items", _records(), analyzer="ik")
        mgr.activate("memory_items", analyzer="ik")

        # Facade with the DEFAULT resolver: the fake profile has no production
        # provider, so both paths serve the declared BM25-on-ACTIVE mode.
        facade = build_retrieval_facade(native_store=client)
        facade_response = facade.retrieve(
            MemoryQuery(text="middle joint_not_reached", limit=5),
            purpose=RetrievalPurpose.HUMAN_SEARCH,
        )
        assert facade_response.retrieval_mode == MODE_ACTIVE_BM25

        args = argparse.Namespace(
            **{
                **vars(_stack_args(path)),
                "query": "middle joint_not_reached",
                "limit": 5,
                "type": None,
                "tenant_id": None,
                "project_id": None,
                "site_id": None,
                "robot_id": None,
                "body_id": None,
                "task_id": None,
                "safety_filter": False,
                "purpose": "human_search",
                "reranker": False,
                "no_vector": False,
            }
        )
        assert cmd_memory_v2_query(args) == 0
        cli_output = json.loads(capsys.readouterr().out)
        cli_ids = [row["memory_id"] for row in cli_output["results"]]
        facade_ids = [c.memory_id for c in facade_response.candidates]
        assert cli_ids == facade_ids
        assert cli_output["retrieval"]["retrieval_mode"] == facade_response.retrieval_mode
    finally:
        for table in ("memory_items", "projection_registry"):
            client.delete_where(table, {})
        for name in client.list_collections():
            if name.startswith("memory_items__fake_8d_v1__"):
                client._client.delete_collection(name)
        _close(client)
