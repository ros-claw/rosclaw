"""Facade fallback chain + purpose policy tests (PR-MEM-5, v4 §13)."""

from __future__ import annotations

import time

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery
from rosclaw.memory.v2.runtime_retrieval import (
    MODE_ABSTAIN,
    MODE_ACTIVE_BM25,
    MODE_SQLITE_LEXICAL,
    EmbeddingProviderResolver,
    RetrievalPurpose,
    build_retrieval_facade,
)
from rosclaw.memory.v2.runtime_retrieval.fallback import hybrid_mode_for
from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider
from tests.memory.v2.runtime_retrieval.test_resolvers import (
    FakeNativeStore,
    _pointer_and_build_rows,
)

PHYSICAL = "memory_items__fake_8d_v1__ik__gabc"


def _failure_row(memory_id: str, body: str, joint: str | None = None) -> dict:
    return {
        "id": memory_id,
        "memory_type": "failure",
        "robot_id": "r1",
        "body_id": body,
        "joint_name": joint,
        "failure_type": "joint_not_reached",
        "gesture_name": "rock",
        "title": f"{body} {joint} failure",
        "document": f"{body} 的 {joint} 未达到目标位置 joint_not_reached",
        "status": "active",
        "outcome": "failure",
        "confidence": 0.9,
        "importance": 0.75,
        "event_time": time.time(),
        "evidence_refs": '["evt_1"]',
        "tags": '["failure"]',
        "metadata": "{}",
    }


def _resolver(provider: FakeProvider | None = None) -> EmbeddingProviderResolver:
    provider = provider or FakeProvider(FAKE_PROFILE)
    return EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: provider,
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )


def test_active_hybrid_serves_and_discloses_truth() -> None:
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_left_1", "rh56_left_01", "middle")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert response.retrieval_mode == hybrid_mode_for(FAKE_PROFILE.profile_id)
    assert response.fallback is False
    assert response.fallback_reason is None
    assert response.physical_collection == PHYSICAL
    assert response.embedding_profile_id == FAKE_PROFILE.profile_id
    assert [c.memory_id for c in response.candidates] == ["mem_left_1"]
    candidate = response.candidates[0]
    assert candidate.body_match is True
    assert candidate.joint_match is True
    assert candidate.exact_entity_match is True
    assert candidate.cross_body_reference is False
    # The query named one hand: the search carried the hard body filter.
    assert store.hybrid_calls[0]["filters"]["body_id"] == "rh56_left_01"
    assert store.hybrid_calls[0]["query_embedding"] is not None


def test_provider_failure_bm25_fallback_on_active() -> None:
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_right_1", "rh56_right_01", "thumb")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    failing = FakeProvider(FAKE_PROFILE, fail=True)
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver(failing))
    response = facade.retrieve(
        MemoryQuery(text="thumb failure", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert response.retrieval_mode == MODE_ACTIVE_BM25
    assert response.fallback is True
    assert response.fallback_reason.startswith("embedding_provider_unavailable")
    assert response.physical_collection == PHYSICAL
    assert [c.memory_id for c in response.candidates] == ["mem_right_1"]
    # BM25 leg used fulltext_search, never another model's vectors.
    assert store.fulltext_calls
    assert not store.hybrid_calls


def test_missing_active_pointer_falls_to_sqlite_lexical(tmp_path) -> None:
    store = FakeNativeStore(registry_rows=[])  # no pointer at all
    sqlite = SQLiteKnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    sqlite.connect()
    repo = MemoryRepository(sqlite)
    repo.store(
        MemoryItem(
            memory_type="failure",
            robot_id="r1",
            title=" lexical-only fallback target",
            document="sqlite lexical path document",
            evidence_refs=["evt_1"],
        )
    )
    facade = build_retrieval_facade(native_store=store, sqlite_store=sqlite)
    response = facade.retrieve(
        MemoryQuery(text="fallback target", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert response.retrieval_mode == MODE_SQLITE_LEXICAL
    assert response.fallback is True
    assert response.fallback_reason == "no_active_pointer"
    assert response.candidates
    assert response.candidates[0].physical_collection is None


def test_all_stores_down_abstains_honestly() -> None:
    facade = build_retrieval_facade(native_store=None, sqlite_store=None)
    response = facade.retrieve(MemoryQuery(text="anything", limit=5))
    assert response.retrieval_mode == MODE_ABSTAIN
    assert response.candidates == []
    assert response.fallback is True
    assert response.fallback_reason


def test_how_intervention_never_retries_cross_body() -> None:
    """v4 §7.4: starved same-body query on the HOW path must NOT surface the
    other body's memory — ABSTAIN upstream, never borrow."""
    store = FakeNativeStore(
        # Only right-body memories exist.
        collections={PHYSICAL: [_failure_row("mem_right_1", "rh56_right_01", "middle")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HOW_INTERVENTION,
    )
    assert response.candidates == []
    assert store.hybrid_calls  # the constrained query ran...
    assert all(
        call["filters"].get("body_id") == "rh56_left_01" for call in store.hybrid_calls
    )  # ...but no unfiltered retry happened


def test_human_search_annotates_cross_body_reference() -> None:
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_right_1", "rh56_right_01", "middle")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert [c.memory_id for c in response.candidates] == ["mem_right_1"]
    assert response.candidates[0].cross_body_reference is True
    assert response.candidates[0].body_match is False
    # The second call was the unfiltered retry.
    assert len(store.hybrid_calls) == 2
    assert "body_id" not in store.hybrid_calls[1]["filters"]


def test_know_reasoning_also_forbids_cross_body_retry() -> None:
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_right_1", "rh56_right_01", "middle")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle", limit=5),
        purpose=RetrievalPurpose.KNOW_REASONING,
    )
    assert response.candidates == []
    assert len(store.hybrid_calls) == 1


def test_missing_entity_attribution_is_not_a_match() -> None:
    """v4 §4.4: joint_name=None means unattributed, never 'matches anything'."""
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_left_na", "rh56_left_01", None)]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HUMAN_SEARCH,
    )
    assert response.candidates[0].joint_match is None
    assert response.candidates[0].exact_entity_match is False


def test_reranker_required_disclosed_for_how_purpose() -> None:
    store = FakeNativeStore(
        collections={PHYSICAL: [_failure_row("mem_left_1", "rh56_left_01", "middle")]},
        registry_rows=_pointer_and_build_rows(physical=PHYSICAL),
    )
    facade = build_retrieval_facade(native_store=store, provider_resolver=_resolver())
    response = facade.retrieve(
        MemoryQuery(text="左手 middle joint_not_reached", limit=5),
        purpose=RetrievalPurpose.HOW_INTERVENTION,
    )
    assert response.reranker_required is True
    assert response.reranker_applied is False  # no reranker wired — disclosed, not hidden


# ---------------------------------------------------------------------------
# KNOW / HOW integration points (v4 §13: know/how use the active collection)
# ---------------------------------------------------------------------------


def _stub_facade_response(rows: list[dict], physical: str = PHYSICAL):
    """Build a facade-like response object from ACTIVE-shaped rows."""
    from rosclaw.memory.v2.models import MemoryItem
    from rosclaw.memory.v2.runtime_retrieval.result import (
        RetrievalCandidate,
        RetrievalResponse,
    )

    candidates = []
    for rank, row in enumerate(rows, start=1):
        candidates.append(
            RetrievalCandidate(
                memory_id=row["id"],
                memory_type=row.get("memory_type", "failure"),
                vector_rank=None,
                bm25_rank=None,
                fusion_rank=rank,
                rerank_score=None,
                exact_entity_match=True,
                body_match=True,
                joint_match=True,
                failure_type_match=True,
                physical_collection=physical,
                embedding_profile_id=FAKE_PROFILE.profile_id,
                score_semantics="test semantics",
                item=MemoryItem.from_record(row),
            )
        )
    return RetrievalResponse(
        retrieval_mode="fake_hybrid",
        logical_collection="memory_items",
        physical_collection=physical,
        embedding_profile_id=FAKE_PROFILE.profile_id,
        purpose=RetrievalPurpose.HOW_INTERVENTION,
        fallback=False,
        fallback_reason=None,
        candidates=candidates,
    )


class _StubFacade:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[tuple] = []

    def retrieve(self, query, *, purpose):
        self.calls.append((query, purpose))
        return self._response


def test_memory_interface_find_analogy_uses_active_collection() -> None:
    from rosclaw.memory.interface import MemoryInterface

    row = _failure_row("mem_left_hint", "rh56_left_01", "middle")
    import json

    row["metadata"] = json.dumps({"recovery_hint": "增加回合间冷却"})
    facade = _StubFacade(_stub_facade_response([row]))
    memory = MemoryInterface("r1", retrieval_facade=facade)
    analogy = memory.find_analogy("middle joint_not_reached", limit=1)
    assert analogy is not None
    assert analogy["id"] == "mem_left_hint"
    assert analogy["action_suggestion"] == "增加回合间冷却"
    assert analogy["source"] == "memory_v2_active"
    assert analogy["physical_collection"] == PHYSICAL
    # HOW path: purpose must be how_intervention (cross-body forbidden).
    _, purpose = facade.calls[0]
    assert purpose is RetrievalPurpose.HOW_INTERVENTION


def test_know_match_symptom_uses_active_collection() -> None:
    import json

    from rosclaw.know.interface import KnowledgeInterface
    from rosclaw.memory.interface import MemoryInterface

    row = _failure_row("mem_left_hint", "rh56_left_01", "middle")
    row["metadata"] = json.dumps({"recovery_hint": "冷却并复位"})
    facade = _StubFacade(_stub_facade_response([row]))
    memory = MemoryInterface("r1", retrieval_facade=facade)
    know = KnowledgeInterface(robot_id="r1", memory_interface=memory)
    know._initialized = True
    match = know.match_symptom("middle joint_not_reached")
    assert match is not None
    assert match["source"] == "memory_v2_active"
    assert match["memory_id"] == "mem_left_hint"
    assert match["fix"] == "冷却并复位"
    assert match["similarity"] is None  # fused rank is not a similarity
    _, purpose = facade.calls[0]
    assert purpose is RetrievalPurpose.KNOW_REASONING


def test_memory_interface_falls_back_to_legacy_when_facade_empty() -> None:
    from rosclaw.memory.interface import MemoryInterface

    facade = _StubFacade(_stub_facade_response([]))
    memory = MemoryInterface("r1", retrieval_facade=facade)
    memory._client.connect()
    memory.store_experience(
        event_id="evt_legacy_1",
        event_type="calibration",
        instruction="legacy calibration experience",
        outcome="success",
        tags=["legacytag"],
    )
    results = memory.find_similar_experiences("legacytag calibration", limit=3)
    assert results  # legacy experience_graph path still serves
    assert results[0].get("source") != "memory_v2_active"
