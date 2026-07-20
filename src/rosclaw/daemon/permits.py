"""Daemon-owned, body-bound permits for REAL action authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.kernel import ActionEnvelope, AuthorizationContext


@dataclass(frozen=True)
class ExecutionPermit:
    """A server-issued, bounded authorization for physical execution."""

    permit_id: str
    principal_id: str
    peer_uid: int
    body_id: str
    body_snapshot_hash: str
    capabilities: tuple[str, ...]
    action_intent_hash: str
    expires_at: datetime
    max_uses: int = 1

    def __post_init__(self) -> None:
        required = {
            "permit_id": self.permit_id,
            "principal_id": self.principal_id,
            "body_id": self.body_id,
            "body_snapshot_hash": self.body_snapshot_hash,
            "action_intent_hash": self.action_intent_hash,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"ExecutionPermit requires: {', '.join(missing)}")
        if not self.capabilities or any(not item.strip() for item in self.capabilities):
            raise ValueError("ExecutionPermit requires at least one non-empty capability")
        if "*" in self.capabilities:
            raise ValueError("ExecutionPermit capabilities must be explicit; wildcard is forbidden")
        digest = self.action_intent_hash.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("ExecutionPermit.action_intent_hash must be a sha256 digest")
        if self.peer_uid < 0:
            raise ValueError("ExecutionPermit.peer_uid must be non-negative")
        if self.max_uses < 1:
            raise ValueError("ExecutionPermit.max_uses must be positive")
        if self.expires_at.tzinfo is None:
            raise ValueError("ExecutionPermit.expires_at must be timezone-aware")


@dataclass(frozen=True)
class PermitDecision:
    """Result of daemon-side authorization."""

    allowed: bool
    code: str
    message: str
    authorization: AuthorizationContext


class PermitAuthority:
    """In-process permit authority that never trusts caller ``approved`` flags."""

    def __init__(self) -> None:
        self._permits: dict[str, ExecutionPermit] = {}
        self._uses: dict[str, int] = {}
        self._consumed_actions: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, permit: ExecutionPermit) -> None:
        """Register a permit from a trusted daemon/operator integration."""

        with self._lock:
            if permit.permit_id in self._permits:
                raise ValueError(f"Permit {permit.permit_id!r} is already registered")
            self._permits[permit.permit_id] = permit
            self._uses[permit.permit_id] = 0

    def authorize(
        self,
        action: ActionEnvelope,
        peer: PeerCredentials,
    ) -> PermitDecision:
        """Consume a matching permit and return daemon-authored authorization."""

        caller = action.authorization
        denied = AuthorizationContext(
            principal_id=str(caller.principal_id),
            approved=False,
            approval_id=caller.approval_id,
            scopes=[],
        )
        permit_id = str(caller.approval_id or "")
        if not permit_id:
            return PermitDecision(
                False,
                "AUTHORIZATION_REQUIRED",
                "REAL execution requires a daemon-issued permit.",
                denied,
            )

        with self._lock:
            permit = self._permits.get(permit_id)
            if permit is None:
                return PermitDecision(
                    False,
                    "AUTHORIZATION_REQUIRED",
                    f"Permit {permit_id!r} is not registered by rosclawd.",
                    denied,
                )
            if datetime.now(UTC) >= permit.expires_at:
                return PermitDecision(
                    False,
                    "PERMIT_EXPIRED",
                    f"Permit {permit_id!r} has expired.",
                    denied,
                )
            if peer.uid != permit.peer_uid:
                return PermitDecision(
                    False,
                    "PERMIT_PEER_MISMATCH",
                    "Permit is bound to a different Unix peer UID.",
                    denied,
                )
            if caller.principal_id != permit.principal_id:
                return PermitDecision(
                    False,
                    "PERMIT_PRINCIPAL_MISMATCH",
                    "Permit principal does not match the action authorization context.",
                    denied,
                )
            if action.body_id != permit.body_id:
                return PermitDecision(
                    False,
                    "PERMIT_BODY_MISMATCH",
                    "Permit is bound to a different body.",
                    denied,
                )
            if action.body_snapshot_hash != permit.body_snapshot_hash:
                return PermitDecision(
                    False,
                    "PERMIT_SNAPSHOT_MISMATCH",
                    "Permit is bound to a different immutable body snapshot.",
                    denied,
                )
            if action.capability_id not in permit.capabilities:
                return PermitDecision(
                    False,
                    "PERMIT_SCOPE_MISMATCH",
                    "Permit does not authorize the requested capability.",
                    denied,
                )
            if not hmac.compare_digest(
                action_intent_hash(action),
                permit.action_intent_hash,
            ):
                return PermitDecision(
                    False,
                    "PERMIT_INTENT_MISMATCH",
                    "Permit is bound to different action arguments or execution constraints.",
                    denied,
                )

            consumed_permit = self._consumed_actions.get(action.action_id)
            if consumed_permit == permit_id:
                return self._allow(permit)
            uses = self._uses.get(permit_id, 0)
            if uses >= permit.max_uses:
                return PermitDecision(
                    False,
                    "PERMIT_EXHAUSTED",
                    f"Permit {permit_id!r} has no remaining uses.",
                    denied,
                )
            self._uses[permit_id] = uses + 1
            self._consumed_actions[action.action_id] = permit_id
            return self._allow(permit)

    def status(self) -> dict[str, int]:
        """Return non-sensitive permit counters."""

        with self._lock:
            active = sum(
                1
                for permit_id, permit in self._permits.items()
                if datetime.now(UTC) < permit.expires_at
                and self._uses.get(permit_id, 0) < permit.max_uses
            )
            return {
                "registered": len(self._permits),
                "active": active,
                "consumed_actions": len(self._consumed_actions),
            }

    @staticmethod
    def _allow(permit: ExecutionPermit) -> PermitDecision:
        return PermitDecision(
            True,
            "AUTHORIZED",
            (
                "Daemon-issued permit matched the peer, body, snapshot, capability, "
                "and action intent."
            ),
            AuthorizationContext(
                principal_id=permit.principal_id,
                approved=True,
                approval_id=permit.permit_id,
                scopes=list(permit.capabilities),
            ),
        )


def action_intent_hash(action: ActionEnvelope) -> str:
    """Hash the physical intent and constraints, excluding caller approval claims."""

    payload = action.to_dict()
    intent = {
        "body_id": payload["body_id"],
        "body_snapshot_hash": payload["body_snapshot_hash"],
        "capability_id": payload["capability_id"],
        "arguments": payload["arguments"],
        "execution_mode": payload["execution_mode"],
        "risk_class": payload["risk_class"],
        "deadline_at": payload["deadline_at"],
        "expected_effect": payload["expected_effect"],
        "verification_policy": payload["verification_policy"],
    }
    encoded = json.dumps(
        intent,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "ExecutionPermit",
    "PermitAuthority",
    "PermitDecision",
    "action_intent_hash",
]
