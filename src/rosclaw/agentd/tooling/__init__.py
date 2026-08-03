"""Tool/Capability Catalog (PR-05, 大纲 §7).

The catalog is the single seam between "capabilities that exist" and "tools
the model may see". Deterministic hard filters (safety) run before any
relevance ranking, and safety conditions never enter the model-facing score.
"""

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.agentd.tooling.evidence import EvidenceEnvelope, wrap_observation
from rosclaw.agentd.tooling.resolver import (
    MAX_INJECTED_TOOLS,
    FilterContext,
    ToolResolver,
)

__all__ = [
    "EvidenceEnvelope",
    "FilterContext",
    "MAX_INJECTED_TOOLS",
    "ToolCatalog",
    "ToolResolver",
    "wrap_observation",
]
