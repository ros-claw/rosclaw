"""
ROSClaw Memory - Experience Grounding Engine

Interface to SeekDB (Shared Knowledge Plane).
Stores and retrieves robot experiences, skills, and world knowledge.
"""

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import (
    SeekDBClient,
    SeekDBMemoryClient,
    SeekDBMySQLClient,
    SeekDBSQLiteClient,
)
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent

# Backward-compatible aliases for documentation
SQLiteSeekDB = SeekDBSQLiteClient
MemorySeekDB = SeekDBMemoryClient

__all__ = [
    "MemoryInterface",
    "SeekDBClient",
    "SeekDBMemoryClient",
    "MemorySeekDB",
    "SeekDBMySQLClient",
    "SeekDBSQLiteClient",
    "SQLiteSeekDB",
    "PraxisEvent",
    "FailureMemory",
    "ArtifactRef",
]
