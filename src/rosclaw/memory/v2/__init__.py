"""ROSClaw Memory 2.0 — structured embodied memory (PR-MEM-1).

Public surface:

* :class:`MemoryItem` / :class:`MemoryEvidence` — the unified model;
* :class:`MemoryRepository` — typed CRUD with evidence, idempotent writes;
* :class:`MemoryWriteGate` — STORE/MERGE/UPDATE/IGNORE/QUARANTINE decisions;
* :func:`distill_session_dir` / :func:`distill_events` — practice distillation;
* :class:`MemoryConsolidator` — dedup, decay, supersession, TTL, pinning.
"""

from rosclaw.memory.v2.consolidate import ConsolidateResult, MemoryConsolidator
from rosclaw.memory.v2.distill import (
    DistillResult,
    SessionContext,
    build_candidates,
    distill_events,
    distill_session_dir,
    load_session_events,
)
from rosclaw.memory.v2.gate import MemoryDecision, MemoryWriteGate
from rosclaw.memory.v2.models import (
    SCHEMA_VERSION,
    EvidenceType,
    GateDecision,
    MemoryEvidence,
    MemoryItem,
    MemoryStatus,
    MemoryType,
)
from rosclaw.memory.v2.repository import MemoryRepository

__all__ = [
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
