"""ROSClaw versioned cross-process contracts.

Maturity: **experimental** (ADR-0000). These schemas are the stable language
between rosclaw-agentd, workers, team peers, and the operator broker. They
carry no secrets — credentials are always references (``*_ref``).

Subpackages:
- ``rosclaw.contracts.agent``  — mission, task graph, context bundle, decision
- ``rosclaw.contracts.team``   — member card, role lease, world delta
"""

from rosclaw.contracts.common import (
    ContractError,
    ContractModel,
    UnsupportedVersionError,
    ValidationError,
    canonical_json,
    content_hash,
    new_id,
)

__all__ = [
    "ContractError",
    "ContractModel",
    "UnsupportedVersionError",
    "ValidationError",
    "canonical_json",
    "content_hash",
    "new_id",
]

MATURITY = "experimental"
