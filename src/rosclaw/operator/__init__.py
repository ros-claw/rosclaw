"""Operator plane: broker contracts (daemon-side) + agentd MissionGrants.

Two complementary layers (ADR-0006):

- ``protocol`` / ``store`` — upstream daemon-side operator-broker consent
  plane: trusted operator proposals and decisions for guarded actions.
- ``broker`` — agentd-side approval cards and MissionGrants (public scope
  only; private signatures never leave this package).

Maturity: **experimental**.
"""

from rosclaw.operator.broker import GrantDeniedError, OperatorBroker
from rosclaw.operator.protocol import (
    OPERATOR_PROPOSAL_SCHEMA_VERSION,
    OperatorDecision,
    OperatorProposal,
    ProposalState,
)
from rosclaw.operator.store import OperatorProposalError, OperatorProposalStore

__all__ = [
    "OPERATOR_PROPOSAL_SCHEMA_VERSION",
    "GrantDeniedError",
    "OperatorBroker",
    "OperatorDecision",
    "OperatorProposal",
    "OperatorProposalError",
    "OperatorProposalStore",
    "ProposalState",
]

MATURITY = "experimental"
