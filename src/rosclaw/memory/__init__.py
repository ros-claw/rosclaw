"""
ROSClaw Memory - Experience Grounding Engine

Interface to SeekDB (Shared Knowledge Plane).
Stores and retrieves robot experiences, skills, and world knowledge.
"""

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import (
    ROSCLAW_STRUCTURED_SCHEMAS,
    SEEKDB_SCHEMAS,
    InMemoryKnowledgeStore,
    InMemoryStructuredStore,
    SeekDBClient,
    SeekDBMemoryClient,
    SeekDBMySQLClient,
    SeekDBSQLiteClient,
    SeekDBSQLStore,
    SQLiteKnowledgeStore,
    SQLiteStructuredStore,
    StructuredStore,
)
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent

# Backward-compatible aliases for documentation
SQLiteSeekDB = SQLiteStructuredStore
MemorySeekDB = InMemoryStructuredStore

__all__ = [
    "SeekDBClient",
    "InMemoryKnowledgeStore",
    "SQLiteKnowledgeStore",
    "SeekDBMySQLClient",
    "SEEKDB_SCHEMAS",
    "ROSCLAW_STRUCTURED_SCHEMAS",
    "MemoryInterface",
    "StructuredStore",
    "InMemoryStructuredStore",
    "SeekDBMemoryClient",
    "MemorySeekDB",
    "SeekDBSQLStore",
    "SQLiteStructuredStore",
    "SeekDBSQLiteClient",
    "SQLiteSeekDB",
    "PraxisEvent",
    "FailureMemory",
    "ArtifactRef",
]

# ---- Memory 2.0 (PR-MEM-1..DF-16B; moved up from memory/v2 in DF-24.3) ----

from rosclaw.memory.consolidate import ConsolidateResult, MemoryConsolidator  # noqa: E402
from rosclaw.memory.distill import (  # noqa: E402
    DistillResult,
    SessionContext,
    build_candidates,
    distill_events,
    distill_session_dir,
    load_session_events,
)  # noqa: E402
from rosclaw.memory.gate import MemoryDecision, MemoryWriteGate  # noqa: E402
from rosclaw.memory.models import (  # noqa: E402
    SCHEMA_VERSION,
    EvidenceType,
    GateDecision,
    MemoryEvidence,
    MemoryItem,
    MemoryStatus,
    MemoryType,
)
from rosclaw.memory.repository import MemoryRepository  # noqa: E402

__all__ = __all__ + [
    "SCHEMA_VERSION",
    "ConsolidateResult",
    "DistillResult",
    "EvidenceType",
    "GateDecision",
    "MemoryConsolidator",
    "MemoryDecision",
    "MemoryEvidence",
    "MemoryItem",
    "MemoryRepository",
    "MemoryStatus",
    "MemoryType",
    "MemoryWriteGate",
    "SessionContext",
    "build_candidates",
    "distill_events",
    "distill_session_dir",
    "load_session_events",
]
