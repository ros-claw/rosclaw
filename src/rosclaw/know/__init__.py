"""DEPRECATED package (PR-DF-24.1): legacy Knowledge runtime.

The physical home moved: ``rosclaw.know`` → ``rosclaw.knowledge.legacy``
(per the Phase-II DF-16.1 split; the DF-09 alias module anticipated this
move).  This package stays as the compatibility shim for at least one
full minor release — import legacy symbols via
``rosclaw.knowledge.legacy`` so the move is invisible to consumers.
"""

from rosclaw.knowledge.legacy import *  # noqa: F401,F403
from rosclaw.knowledge.legacy import (
    AssetsLoader,
    EmbodimentCard,
    KnowledgeBatchEngine,
    KnowledgeInterface,
    LegacyKnowledgeRuntime,
    TaskCard,
    VerifierCard,
    task_pack_for,
)

__all__ = [
    "AssetsLoader",
    "EmbodimentCard",
    "KnowledgeBatchEngine",
    "KnowledgeInterface",
    "LegacyKnowledgeRuntime",
    "TaskCard",
    "VerifierCard",
    "task_pack_for",
]
