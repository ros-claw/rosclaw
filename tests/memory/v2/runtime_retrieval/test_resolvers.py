"""Unit tests for the ACTIVE resolver + provider resolver (PR-MEM-5, v4 §13)."""

from __future__ import annotations

import time

import pytest

from rosclaw.embedding.protocol import EmbeddingProfile
from rosclaw.memory.v2.runtime_retrieval.active_resolver import (
    ACTIVE_COLLECTION_MISSING,
    ACTIVE_DESCRIPTOR_INCOMPLETE,
    ACTIVE_DIMENSION_MISMATCH,
    NO_ACTIVE_POINTER,
    ActiveCollectionResolver,
    ActiveIndexUnavailableError,
)
from rosclaw.memory.v2.runtime_retrieval.provider_resolver import (
    PROVIDER_IDENTITY_MISMATCH,
    PROVIDER_PROFILE_UNKNOWN,
    EmbeddingProviderResolver,
    ProviderUnavailableError,
)
from rosclaw.storage.versioned_collections import VersionedCollectionManager
from tests.embedding.test_embedding_providers import FAKE_PROFILE, FakeProvider

OTHER_PROFILE = EmbeddingProfile(
    profile_id="other_8d_v1",
    model_id="other/model",
    model_revision="rev9",
    dimension=8,
    normalize=True,
    distance="cosine",
    query_instruction=None,
    document_instruction=None,
    max_tokens=32,
    provider_type="fake",
)


def _pointer_and_build_rows(
    *,
    logical: str = "memory_items",
    physical: str = "memory_items__fake_8d_v1__ik__gabc",
    profile: EmbeddingProfile = FAKE_PROFILE,
    dimension: int | None = None,
    activated_at: float | None = None,
) -> list[dict]:
    activated = activated_at if activated_at is not None else time.time()
    pointer_id = VersionedCollectionManager._pointer_id(logical)
    return [
        {
            "id": "projection_build_1",
            "row_kind": "build",
            "build_id": "abc",
            "logical_name": logical,
            "physical_collection": physical,
            "embedding_profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "dimension": dimension if dimension is not None else profile.dimension,
            "normalize": profile.normalize,
            "distance": profile.distance,
            "analyzer": "ik",
            "record_count": 3,
            "status": "ACTIVE",
            "created_at": activated - 10,
            "activated_at": activated,
        },
        {
            "id": pointer_id,
            "row_kind": "active_pointer",
            "logical_name": logical,
            "active_physical_collection": physical,
            "active_activated_at": activated,
            "previous_physical_collection": None,
            "previous_activated_at": None,
            "updated_at": activated,
        },
    ]


class FakeNativeStore:
    """Minimal native-store double exposing the surfaces the resolvers use."""

    def __init__(
        self,
        *,
        collections: dict[str, list[dict]] | None = None,
        registry_rows: list[dict] | None = None,
        dimension: int = 8,
    ) -> None:
        self.collections = collections or {}
        self.registry_rows = registry_rows or []
        self.dimension = dimension
        self._client = self
        self.hybrid_calls: list[dict] = []
        self.fulltext_calls: list[dict] = []
        self.search_error: Exception | None = None

    def query(self, table: str, filters: dict | None = None, limit: int = 100) -> list[dict]:
        assert table == "projection_registry"

        def match(row: dict) -> bool:
            return all(row.get(key) == value for key, value in (filters or {}).items())

        return [row for row in self.registry_rows if match(row)][:limit]

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def embedding_info(self, table: str) -> dict:
        return {"dimension": self.dimension}

    def _filtered(self, table: str, filters: dict | None, limit: int) -> list[dict]:
        def match(row: dict) -> bool:
            return all(row.get(key) == value for key, value in (filters or {}).items())

        return [dict(row) for row in self.collections.get(table, []) if match(row)][:limit]

    def hybrid_search(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
        candidate_window: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        if self.search_error is not None:
            raise self.search_error
        self.hybrid_calls.append(
            {"table": table, "filters": filters, "query_embedding": query_embedding}
        )
        return self._filtered(table, filters, limit)

    def fulltext_search(
        self,
        table: str,
        query_text: str,
        filters: dict | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if self.search_error is not None:
            raise self.search_error
        self.fulltext_calls.append({"table": table, "filters": filters})
        return self._filtered(table, filters, limit)


# ---------------------------------------------------------------------------
# ActiveCollectionResolver
# ---------------------------------------------------------------------------


def test_active_pointer_resolved() -> None:
    physical = "memory_items__fake_8d_v1__ik__gabc"
    store = FakeNativeStore(
        collections={physical: [{"id": "m1"}]},
        registry_rows=_pointer_and_build_rows(),
    )
    descriptor = ActiveCollectionResolver(store).resolve("memory_items")
    assert descriptor.physical_collection == physical
    assert descriptor.embedding_profile_id == FAKE_PROFILE.profile_id
    assert descriptor.model_id == FAKE_PROFILE.model_id
    assert descriptor.model_revision == FAKE_PROFILE.model_revision
    assert descriptor.dimension == FAKE_PROFILE.dimension
    assert descriptor.analyzer == "ik"


def test_missing_pointer_raises_declared_reason() -> None:
    store = FakeNativeStore(registry_rows=[])
    with pytest.raises(ActiveIndexUnavailableError) as excinfo:
        ActiveCollectionResolver(store).resolve("memory_items")
    assert excinfo.value.reason == NO_ACTIVE_POINTER


def test_missing_physical_collection_raises() -> None:
    store = FakeNativeStore(
        collections={},  # engine does not have the registered collection
        registry_rows=_pointer_and_build_rows(),
    )
    with pytest.raises(ActiveIndexUnavailableError) as excinfo:
        ActiveCollectionResolver(store).resolve("memory_items")
    assert excinfo.value.reason == ACTIVE_COLLECTION_MISSING


def test_dimension_mismatch_raises() -> None:
    physical = "memory_items__fake_8d_v1__ik__gabc"
    store = FakeNativeStore(
        collections={physical: [{"id": "m1"}]},
        registry_rows=_pointer_and_build_rows(dimension=8),
        dimension=16,  # engine reports a different dimension than the registry
    )
    with pytest.raises(ActiveIndexUnavailableError) as excinfo:
        ActiveCollectionResolver(store).resolve("memory_items")
    assert excinfo.value.reason == ACTIVE_DIMENSION_MISMATCH


def test_incomplete_descriptor_raises() -> None:
    rows = _pointer_and_build_rows()
    rows[0]["model_revision"] = None  # corrupt registry row
    store = FakeNativeStore(
        collections={"memory_items__fake_8d_v1__ik__gabc": [{"id": "m1"}]},
        registry_rows=rows,
    )
    with pytest.raises(ActiveIndexUnavailableError) as excinfo:
        ActiveCollectionResolver(store).resolve("memory_items")
    assert excinfo.value.reason == ACTIVE_DESCRIPTOR_INCOMPLETE


# ---------------------------------------------------------------------------
# EmbeddingProviderResolver
# ---------------------------------------------------------------------------


def _descriptor(**overrides):
    from rosclaw.memory.v2.runtime_retrieval.active_resolver import ActiveIndexDescriptor

    base = {
        "logical_name": "memory_items",
        "physical_collection": "memory_items__fake_8d_v1__ik__gabc",
        "embedding_profile_id": FAKE_PROFILE.profile_id,
        "model_id": FAKE_PROFILE.model_id,
        "model_revision": FAKE_PROFILE.model_revision,
        "dimension": FAKE_PROFILE.dimension,
        "analyzer": "ik",
        "normalize": True,
        "distance": "cosine",
        "activated_at": time.time(),
    }
    base.update(overrides)
    return ActiveIndexDescriptor(**base)


def test_provider_matches_active_profile() -> None:
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )
    provider = resolver.resolve(_descriptor())
    assert provider.profile.profile_id == FAKE_PROFILE.profile_id


def test_provider_unknown_profile_rejected() -> None:
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={},
    )
    with pytest.raises(ProviderUnavailableError) as excinfo:
        resolver.resolve(_descriptor())
    assert excinfo.value.reason == PROVIDER_PROFILE_UNKNOWN


def test_model_dimension_mismatch_fails() -> None:
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )
    with pytest.raises(ProviderUnavailableError) as excinfo:
        resolver.resolve(_descriptor(dimension=1024))
    assert excinfo.value.reason == PROVIDER_IDENTITY_MISMATCH


def test_model_revision_mismatch_fails() -> None:
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(FAKE_PROFILE),
        profiles={FAKE_PROFILE.profile_id: FAKE_PROFILE},
    )
    with pytest.raises(ProviderUnavailableError) as excinfo:
        resolver.resolve(_descriptor(model_revision="deadbeef"))
    assert excinfo.value.reason == PROVIDER_IDENTITY_MISMATCH


def test_other_model_never_substituted() -> None:
    """A provider for a different model must not serve the ACTIVE collection
    even when the descriptor's profile id exists (v4 §3.4)."""
    resolver = EmbeddingProviderResolver(
        provider_factory=lambda profile_id, **kwargs: FakeProvider(OTHER_PROFILE),
        profiles={OTHER_PROFILE.profile_id: OTHER_PROFILE},
    )
    with pytest.raises(ProviderUnavailableError) as excinfo:
        resolver.resolve(
            _descriptor(
                embedding_profile_id=OTHER_PROFILE.profile_id,
                model_id=FAKE_PROFILE.model_id,  # ACTIVE was built by another model
            )
        )
    assert excinfo.value.reason == PROVIDER_IDENTITY_MISMATCH
