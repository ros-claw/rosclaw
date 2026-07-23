"""Canonical ACTIVE index resolution (数据库优化v4 §3.3).

The resolver reads the ``projection_registry`` canonical pointer, finds the
matching build row, and validates that the physical collection still exists
with the recorded dimension.  A missing pointer is NOT a silent signal to
fall back to the logical collection — it raises
:class:`ActiveIndexUnavailableError` with a machine-readable reason so the facade
can enter the declared fallback chain and report ``fallback_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveIndexDescriptor:
    """Everything a query path must know about the serving index (v4 §3.3)."""

    logical_name: str
    physical_collection: str
    embedding_profile_id: str
    model_id: str
    model_revision: str
    dimension: int
    analyzer: str
    normalize: bool
    distance: str
    activated_at: float


# Machine-readable reason codes (v4 §3.3 / §3.7).
NO_ACTIVE_POINTER = "no_active_pointer"
ACTIVE_BUILD_ROW_MISSING = "active_build_row_missing"
ACTIVE_COLLECTION_MISSING = "active_collection_missing"
ACTIVE_DIMENSION_MISMATCH = "active_dimension_mismatch"
ACTIVE_DESCRIPTOR_INCOMPLETE = "active_descriptor_incomplete"


class ActiveIndexUnavailableError(RuntimeError):
    """The canonical ACTIVE index cannot serve queries; carries ``reason``."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail or reason
        super().__init__(self.detail)


class ActiveCollectionResolver:
    """Resolve the canonical ACTIVE physical collection for a logical name."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def resolve(self, logical_name: str) -> ActiveIndexDescriptor:
        # Local import: versioned_collections owns the registry layout.
        from rosclaw.storage.versioned_collections import VersionedCollectionManager

        manager = VersionedCollectionManager(self._store, None)
        active = manager.active(logical_name)
        if active is None:
            raise ActiveIndexUnavailableError(
                NO_ACTIVE_POINTER,
                f"no canonical ACTIVE pointer for logical collection {logical_name!r}",
            )
        physical = active.get("physical_collection")
        required = (
            "embedding_profile_id",
            "model_id",
            "model_revision",
            "dimension",
            "analyzer",
        )
        if not physical or any(active.get(key) in (None, "") for key in required):
            raise ActiveIndexUnavailableError(
                ACTIVE_DESCRIPTOR_INCOMPLETE,
                f"ACTIVE registry row for {logical_name!r} is missing required fields",
            )
        client = getattr(self._store, "_client", None)
        if client is None:
            raise ActiveIndexUnavailableError(
                ACTIVE_COLLECTION_MISSING, "native store is not connected"
            )
        if not client.has_collection(physical):
            raise ActiveIndexUnavailableError(
                ACTIVE_COLLECTION_MISSING,
                f"registered physical collection is missing from the engine: {physical}",
            )
        actual_dimension = self._store.embedding_info(physical).get("dimension")
        expected_dimension = int(active["dimension"])
        if actual_dimension != expected_dimension:
            raise ActiveIndexUnavailableError(
                ACTIVE_DIMENSION_MISMATCH,
                f"{physical}: engine dimension {actual_dimension} != registry {expected_dimension}",
            )
        return ActiveIndexDescriptor(
            logical_name=logical_name,
            physical_collection=str(physical),
            embedding_profile_id=str(active["embedding_profile_id"]),
            model_id=str(active["model_id"]),
            model_revision=str(active["model_revision"]),
            dimension=expected_dimension,
            analyzer=str(active["analyzer"]),
            normalize=bool(active.get("normalize", True)),
            distance=str(active.get("distance") or "cosine"),
            activated_at=float(active.get("activated_at") or 0.0),
        )
