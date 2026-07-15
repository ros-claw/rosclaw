"""
ROSClaw Memory - Experience Grounding Engine

Interface to SeekDB (Shared Knowledge Plane).
Stores and retrieves robot experiences, skills, and world knowledge.
"""

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import (
    InMemoryKnowledgeStore,
    SeekDBClient,
    SeekDBMemoryClient,
    SeekDBMySQLClient,
    SeekDBSQLiteClient,
    SQLiteKnowledgeStore,
)
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent

# Backward-compatible aliases for documentation
SQLiteSeekDB = SQLiteKnowledgeStore
MemorySeekDB = InMemoryKnowledgeStore
# Legacy aliases kept for compatibility (emit DeprecationWarning on instantiation)
_ = SeekDBMemoryClient
_ = SeekDBSQLiteClient

__all__ = [
    "MemoryInterface",
    "SeekDBClient",
    "InMemoryKnowledgeStore",
    "SeekDBMemoryClient",
    "MemorySeekDB",
    "SeekDBMySQLClient",
    "SQLiteKnowledgeStore",
    "SeekDBSQLiteClient",
    "SQLiteSeekDB",
    "PraxisEvent",
    "FailureMemory",
    "ArtifactRef",
]
