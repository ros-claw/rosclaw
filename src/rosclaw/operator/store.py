"""Bounded in-memory store for daemon-owned pending operator proposals."""

from __future__ import annotations

import copy
import json
import secrets
import threading
import uuid
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING, Any

from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode
from rosclaw.operator.protocol import OperatorProposal, ProposalState

if TYPE_CHECKING:
    from rosclaw.daemon.protocol import PeerCredentials

MIN_PROPOSAL_TTL_SEC = 5.0
MAX_PROPOSAL_TTL_SEC = 300.0
MAX_DISPLAY_BYTES = 32_768


class OperatorProposalError(ValueError):
    """Structured proposal lifecycle validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class OperatorProposalStore:
    """Retain a bounded set of exact, UID-bound proposals for one daemon generation."""

    def __init__(self, *, capacity: int = 256) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._proposals: dict[str, OperatorProposal] = {}
        self._action_requests: dict[str, str] = {}
        self._lock = threading.RLock()

    def create(
        self,
        action: ActionEnvelope,
        *,
        display: dict[str, Any],
        origin_peer: PeerCredentials,
        daemon_instance_id: str,
        ttl_sec: float,
        now: datetime | None = None,
    ) -> OperatorProposal:
        """Create one immutable proposal and strip all caller approval claims."""

        ttl = self._ttl(ttl_sec)
        if action.execution_mode not in {ExecutionMode.REAL, ExecutionMode.SHADOW}:
            raise OperatorProposalError(
                "PROPOSAL_REAL_ACTION_REQUIRED",
                "Operator proposals may contain only explicit REAL or SHADOW actions "
                "(FTC-100: SHADOW proposals exercise the full permission chain "
                "with actuation hard-blocked)",
            )
        if not action.body_snapshot_hash.strip():
            raise OperatorProposalError(
                "PROPOSAL_BODY_SNAPSHOT_REQUIRED",
                "REAL/SHADOW operator proposals require a body snapshot hash",
            )
        normalized_display = self._display(display)
        current = (now or datetime.now(UTC)).astimezone(UTC)
        if action.deadline_at is None or current >= action.deadline_at:
            raise OperatorProposalError(
                "ACTION_DEADLINE_EXPIRED",
                "Action deadline expired before proposal creation",
            )
        expires_at = min(current + timedelta(seconds=ttl), action.deadline_at)
        normalized_action = ActionEnvelope.from_dict(copy.deepcopy(action.to_dict()))
        normalized_action.authorization = AuthorizationContext()
        with self._lock:
            existing_request = self._action_requests.get(normalized_action.action_id)
            if existing_request is not None:
                existing = self._proposals[existing_request]
                if existing.action_intent_hash != _action_intent_hash(normalized_action):
                    raise OperatorProposalError(
                        "PROPOSAL_ACTION_ID_CONFLICT",
                        "action_id is already bound to a different operator proposal",
                    )
                return existing
            if len(self._proposals) >= self.capacity:
                self._evict_terminal_locked()
            if len(self._proposals) >= self.capacity:
                raise OperatorProposalError(
                    "PROPOSAL_STORE_FULL",
                    "Operator proposal store is full",
                )
            request_id = f"proposal_{uuid.uuid4().hex}"
            proposal = OperatorProposal(
                request_id=request_id,
                action=normalized_action,
                action_intent_hash=_action_intent_hash(normalized_action),
                display=normalized_display,
                origin_peer=origin_peer,
                created_at=current,
                expires_at=expires_at,
                challenge_nonce=secrets.token_urlsafe(32),
                daemon_instance_id=daemon_instance_id,
                transitions=[{"state": ProposalState.CREATED.value, "at": _iso(current)}],
            )
            self._proposals[request_id] = proposal
            self._action_requests[normalized_action.action_id] = request_id
            return proposal

    def get(
        self,
        request_id: str,
        *,
        now: datetime | None = None,
    ) -> OperatorProposal:
        with self._lock:
            proposal = self._proposals.get(request_id)
            if proposal is None:
                raise OperatorProposalError(
                    "PROPOSAL_NOT_FOUND", f"No operator proposal {request_id!r} exists"
                )
            self._expire_locked(proposal, now or datetime.now(UTC))
            return proposal

    def pending(self, *, now: datetime | None = None) -> list[OperatorProposal]:
        current = now or datetime.now(UTC)
        with self._lock:
            for proposal in self._proposals.values():
                self._expire_locked(proposal, current)
            return [
                proposal
                for proposal in self._proposals.values()
                if proposal.state in {ProposalState.CREATED, ProposalState.PRESENTED}
            ]

    def all(self, *, now: datetime | None = None) -> list[OperatorProposal]:
        """Return a stable snapshot after applying lazy expiry transitions."""

        current = now or datetime.now(UTC)
        with self._lock:
            for proposal in self._proposals.values():
                self._expire_locked(proposal, current)
            return list(self._proposals.values())

    def transition(
        self,
        proposal: OperatorProposal,
        state: ProposalState,
        *,
        now: datetime | None = None,
        operator_principal: str | None = None,
        decision_channel: str | None = None,
        decision_reason: str | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> OperatorProposal:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        with self._lock:
            stored = self._proposals.get(proposal.request_id)
            if stored is not proposal:
                raise OperatorProposalError(
                    "PROPOSAL_NOT_FOUND", "Operator proposal no longer belongs to this store"
                )
            stored.state = ProposalState(state)
            stored.operator_principal = operator_principal or stored.operator_principal
            stored.decision_channel = decision_channel or stored.decision_channel
            stored.decision_reason = decision_reason or stored.decision_reason
            if state in {
                ProposalState.ACCEPTED,
                ProposalState.DECLINED,
                ProposalState.CANCELLED,
                ProposalState.INVALIDATED,
            }:
                stored.decided_at = current
            stored.failure_code = failure_code
            stored.failure_message = failure_message
            stored.transitions.append({"state": state.value, "at": _iso(current)})
            return stored

    def status(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(UTC)
        with self._lock:
            for proposal in self._proposals.values():
                self._expire_locked(proposal, current)
            counts = {state.value.lower(): 0 for state in ProposalState}
            for proposal in self._proposals.values():
                counts[proposal.state.value.lower()] += 1
            return {"total": len(self._proposals), "capacity": self.capacity, **counts}

    @staticmethod
    def _ttl(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperatorProposalError("INVALID_PROPOSAL_TTL", "ttl_sec must be numeric")
        ttl = float(value)
        if not isfinite(ttl) or not MIN_PROPOSAL_TTL_SEC <= ttl <= MAX_PROPOSAL_TTL_SEC:
            raise OperatorProposalError(
                "INVALID_PROPOSAL_TTL",
                f"ttl_sec must be between {MIN_PROPOSAL_TTL_SEC:g} and {MAX_PROPOSAL_TTL_SEC:g}",
            )
        return ttl

    @staticmethod
    def _display(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise OperatorProposalError("INVALID_PROPOSAL_DISPLAY", "display must be an object")
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise OperatorProposalError(
                "INVALID_PROPOSAL_DISPLAY", "display must be finite JSON"
            ) from exc
        if len(encoded) > MAX_DISPLAY_BYTES:
            raise OperatorProposalError(
                "INVALID_PROPOSAL_DISPLAY",
                f"display exceeds {MAX_DISPLAY_BYTES} bytes",
            )
        return copy.deepcopy(value)

    def _expire_locked(self, proposal: OperatorProposal, now: datetime) -> None:
        current = now.astimezone(UTC)
        if proposal.state in {ProposalState.CREATED, ProposalState.PRESENTED} and (
            current >= proposal.expires_at
            or proposal.action.deadline_at is None
            or current >= proposal.action.deadline_at
        ):
            proposal.state = ProposalState.EXPIRED
            proposal.decided_at = current
            proposal.failure_code = "PROPOSAL_EXPIRED"
            proposal.failure_message = "Operator proposal expired before a trusted decision"
            proposal.transitions.append({"state": ProposalState.EXPIRED.value, "at": _iso(current)})

    def _evict_terminal_locked(self) -> None:
        for request_id, proposal in tuple(self._proposals.items()):
            if not proposal.terminal:
                continue
            self._proposals.pop(request_id, None)
            self._action_requests.pop(proposal.action.action_id, None)
            if len(self._proposals) < self.capacity:
                return


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _action_intent_hash(action: ActionEnvelope) -> str:
    # Lazy import keeps the transport-neutral operator contracts independent
    # from rosclaw.daemon's eager package exports during CLI startup.
    from rosclaw.daemon.permits import action_intent_hash

    return action_intent_hash(action)


__all__ = [
    "MAX_PROPOSAL_TTL_SEC",
    "MIN_PROPOSAL_TTL_SEC",
    "OperatorProposalError",
    "OperatorProposalStore",
]
