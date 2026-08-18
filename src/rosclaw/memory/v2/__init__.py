"""DEPRECATED package shim (PR-DF-24.3): ``rosclaw.memory.v2`` moved to
``rosclaw.memory``.  The DATA schema name stays ``memory.v2`` — source
layout version ≠ protocol version (DF-16.3).  Modules register into
``sys.modules`` so both paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

cli = _sys.modules[__name__ + ".cli"] = _importlib.import_module("rosclaw.memory.cli")
consolidate = _sys.modules[__name__ + ".consolidate"] = _importlib.import_module("rosclaw.memory.consolidate")
distill = _sys.modules[__name__ + ".distill"] = _importlib.import_module("rosclaw.memory.distill")
distillation_service = _sys.modules[__name__ + ".distillation_service"] = _importlib.import_module("rosclaw.memory.distillation_service")
document = _sys.modules[__name__ + ".document"] = _importlib.import_module("rosclaw.memory.document")
gate = _sys.modules[__name__ + ".gate"] = _importlib.import_module("rosclaw.memory.gate")
index = _sys.modules[__name__ + ".index"] = _importlib.import_module("rosclaw.memory.index")
models = _sys.modules[__name__ + ".models"] = _importlib.import_module("rosclaw.memory.models")
repository = _sys.modules[__name__ + ".repository"] = _importlib.import_module("rosclaw.memory.repository")
retrieval = _sys.modules[__name__ + ".retrieval"] = _importlib.import_module("rosclaw.memory.retrieval")
tokenizer = _sys.modules[__name__ + ".tokenizer"] = _importlib.import_module("rosclaw.memory.tokenizer")
from rosclaw.memory import (  # noqa: F401,E402
    SCHEMA_VERSION,
    ConsolidateResult,
    DistillResult,
    EvidenceType,
    GateDecision,
    MemoryConsolidator,
    MemoryDecision,
    MemoryEvidence,
    MemoryItem,
    MemoryRepository,
    MemoryStatus,
    MemoryType,
    MemoryWriteGate,
    SessionContext,
    build_candidates,
    distill_events,
    distill_session_dir,
    load_session_events,
)
