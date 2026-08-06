"""Token estimation and layer truncation policy (PR-NA-021).

Deterministic estimator (~4 chars/token heuristic, documented, replaceable).
Truncation priority: L0/L7/L1/L2 are *never* truncated (if they alone exceed
budget the compile fails closed). L4 current-task summary and L3 tool
candidates rank next; L6 org, L5 memory and L8 conversation are trimmed
first — history and low-relevance memory become references, not prose.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Deterministic, provider-agnostic estimate. Never returns 0 for text."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


# Layer names in *ascending* sacrifice order: the first entries are trimmed
# first when over budget. constitution/safety/embodiment/dynamic_self are
# absent — they are protected and must always fit.
TRIM_ORDER = (
    "untrusted_inputs",  # L8 history first
    "memory",  # L5
    "organization",  # L6
    "capabilities",  # L3 (keep top-N candidates)
    "mission",  # L4 (keep current task + anomalies)
)

PROTECTED_LAYERS = ("constitution", "embodiment", "dynamic_self", "safety")
