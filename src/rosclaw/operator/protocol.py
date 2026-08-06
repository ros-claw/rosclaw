"""Versioned, transport-neutral Operator Broker proposal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rosclaw.kernel import ActionEnvelope

if TYPE_CHECKING:
    from rosclaw.daemon.protocol import PeerCredentials

OPERATOR_PROPOSAL_SCHEMA_VERSION = "rosclaw.operator.proposal.v1"


class ProposalState(StrEnum):
    """Durable lifecycle states for one exact operator proposal."""

    CREATED = "CREATED"
    PRESENTED = "PRESENTED"
    ACCEPTED = "ACCEPTED"
    PERMIT_ISSUED = "PERMIT_ISSUED"
    SUBMITTED = "SUBMITTED"
    TERMINAL = "TERMINAL"
    DECLINED = "DECLINED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class OperatorDecision(StrEnum):
    ACCEPT = "ACCEPT"
    DECLINE = "DECLINE"


_TERMINAL_PROPOSAL_STATES = frozenset(
    {
        ProposalState.TERMINAL,
        ProposalState.DECLINED,
        ProposalState.CANCELLED,
        ProposalState.EXPIRED,
        ProposalState.INVALIDATED,
    }
)


@dataclass
class OperatorProposal:
    """One exact action awaiting a decision from a trusted operator process."""

    request_id: str
    action: ActionEnvelope
    action_intent_hash: str
    display: dict[str, Any]
    origin_peer: PeerCredentials
    created_at: datetime
    expires_at: datetime
    challenge_nonce: str
    daemon_instance_id: str
    state: ProposalState = ProposalState.CREATED
    operator_principal: str | None = None
    decision_channel: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    audited_transition_count: int = 0
    # 二次复核 R1/P0-6：创建方（agentd）携带的不透明绑定引用
    # {agent_request_id, mission_id}——daemon 不解释，原样进入
    # challenge/receipt，agentd 侧精确比对。
    client_reference: dict[str, str] = field(default_factory=dict)
    # daemon 签名的 DecisionReceiptV1（决定后填充，公开可读）。
    decision_receipt: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_PROPOSAL_STATES

    def public_dict(self) -> dict[str, Any]:
        """Return Agent-readable state without the decision challenge or Permit."""

        return self._to_dict(include_operator_challenge=False)

    def operator_dict(self) -> dict[str, Any]:
        """Return the trusted Operator Broker view including the one-time challenge."""

        return self._to_dict(include_operator_challenge=True)

    def _to_dict(self, *, include_operator_challenge: bool) -> dict[str, Any]:
        result = {
            "schema_version": OPERATOR_PROPOSAL_SCHEMA_VERSION,
            "request_id": self.request_id,
            "action_id": self.action.action_id,
            "action_intent_hash": self.action_intent_hash,
            "body_id": self.action.body_id,
            "body_snapshot_hash": self.action.body_snapshot_hash,
            "capability_id": self.action.capability_id,
            "execution_mode": self.action.execution_mode.value,
            "risk_class": self.action.risk_class,
            "deadline_at": self.action.to_dict()["deadline_at"],
            "display": dict(self.display),
            "origin": {
                "uid": self.origin_peer.uid,
                "pid": self.origin_peer.pid,
            },
            "daemon_instance_id": self.daemon_instance_id,
            "state": self.state.value,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "operator_principal": self.operator_principal,
            "decision_channel": self.decision_channel,
            "decision_reason": self.decision_reason,
            "decided_at": _iso(self.decided_at),
            "failure": (
                {"code": self.failure_code, "message": self.failure_message}
                if self.failure_code
                else None
            ),
            "transitions": [dict(item) for item in self.transitions],
            "client_reference": dict(self.client_reference),
            "decision_receipt": (
                dict(self.decision_receipt) if self.decision_receipt else None
            ),
        }
        if include_operator_challenge:
            result["challenge_nonce"] = self.challenge_nonce
        return result


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "OPERATOR_PROPOSAL_SCHEMA_VERSION",
    "OperatorDecision",
    "OperatorProposal",
    "ProposalState",
]
