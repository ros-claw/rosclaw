"""Legacy Knowledge runtime — canonical import location (PR-DF-09 / ADR-0010).

``rosclaw.knowledge`` is the canonical Knowledge package.  The pre-v2
runtime (KnowledgeInterface, TaskCard, EmbodimentCard, VerifierCard, the
knowledge graph and evidence ingest) lives on in ``rosclaw.know`` for at
least one full minor release; import it through HERE so the eventual
physical move is a one-line change for consumers:

    from rosclaw.knowledge.legacy import KnowledgeInterface

New code must not gain dependencies on these symbols — they exist for the
rollback-only local Knowledge path (v2 mode == "disabled").
"""

from __future__ import annotations

from rosclaw.know.embodiment_card import EmbodimentCard
from rosclaw.know.interface import KnowledgeInterface
from rosclaw.know.task_card import TaskCard
from rosclaw.know.verifier_card import VerifierCard

# Canonical-in-spirit alias for the legacy entry point (ADR-0010 §4):
# two "official-looking" Knowledge runtimes must not coexist unnamed.
LegacyKnowledgeRuntime = KnowledgeInterface

__all__ = [
    "KnowledgeInterface",
    "LegacyKnowledgeRuntime",
    "TaskCard",
    "EmbodimentCard",
    "VerifierCard",
]
