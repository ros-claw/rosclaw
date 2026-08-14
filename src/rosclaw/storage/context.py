"""Data Plane context (PR-DF-03 / ADR-0009 / ADR-0010).

``DataPlaneContext`` is the single, explicit holder for the Runtime's data
plane.  It replaces the historical pattern of a bare ``seekdb = ...`` local
variable being threaded through Memory / Skill / Knowledge / How / Auto —
a variable whose backend semantics were impossible to determine at the use
site.

The Runtime builds the context exactly once in ``_create_data_plane()`` and
modules receive the parts they need by dependency injection.  Field types
are kept as ``Any`` at the boundary where the concrete classes are optional
imports (native SeekDB, the retrieval facade) so that importing this module
never pulls heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.memory.seekdb_client import StructuredStore


@dataclass
class DataPlaneContext:
    """The Runtime's data plane, initialized once (ADR-0009 §2).

    Attributes:
        structured_store: canonical structured source of truth
            (SQLite edge / SeekDB SQL production / InMemory tests).
        retrieval_store: the SeekDB native retrieval store when the
            structured store itself is native (None otherwise).
        outbox: the transactional outbox for non-safety-realtime writes,
            when enabled.
        memory_projection: Memory -> retrieval-index projection
            (rebuildable; never the source of truth).
        memory_retrieval: the unified MemoryRetrievalFacade serving
            Memory/KNOW/HOW/AUTO queries, when available.
        practice_sink: the Practice event sink (HTTP bridge), when configured.
    """

    structured_store: StructuredStore | None = None
    retrieval_store: Any | None = None
    outbox: Any | None = None
    memory_projection: Any | None = None
    memory_retrieval: Any | None = None
    practice_sink: Any | None = None

    def capabilities(self) -> dict[str, Any]:
        """Observability snapshot for storage status/doctor (DF-07/DF-15)."""
        from rosclaw.storage.factory import StoreFactory

        store = self.structured_store
        caps = StoreFactory.capabilities(store) if store is not None else {}
        return {
            "structured_backend": type(store).__name__ if store is not None else None,
            **caps,
            "retrieval_store": type(self.retrieval_store).__name__
            if self.retrieval_store is not None
            else None,
            "outbox_enabled": self.outbox is not None,
            "memory_projection": self.memory_projection is not None,
            "memory_retrieval": self.memory_retrieval is not None,
            "practice_sink": type(self.practice_sink).__name__
            if self.practice_sink is not None
            else None,
        }
