"""ROSClaw storage layer: factory, outbox, vector, and migration utilities."""

from __future__ import annotations

from rosclaw.storage.factory import StoreFactory
from rosclaw.storage.outbox import OutboxStore, OutboxWorker
from rosclaw.storage.vector import (
    Embedder,
    SQLiteVectorStore,
    TfidfEmbedder,
    VectorStore,
)

# ADR-0010 compatibility alias (PR-DF-01)
StorageFactory = StoreFactory

__all__ = [
    "StoreFactory",
    "StorageFactory",
    "OutboxStore",
    "OutboxWorker",
    "Embedder",
    "VectorStore",
    "TfidfEmbedder",
    "SQLiteVectorStore",
]
