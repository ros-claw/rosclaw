"""Daemon-owned, body-bound permits for REAL action authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.kernel import ActionEnvelope, AuthorizationContext

if TYPE_CHECKING:
    from rosclaw.daemon.ledger import DaemonLedger

PERMIT_SCHEMA_VERSION = "rosclaw.daemon.execution_permit.v1"


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
        missing = [
            name
            for name, value in required.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise ValueError(f"ExecutionPermit requires: {', '.join(missing)}")
        if (
            not isinstance(self.capabilities, tuple)
            or not self.capabilities
            or any(not isinstance(item, str) or not item.strip() for item in self.capabilities)
        ):
            raise ValueError("ExecutionPermit requires at least one non-empty capability")
        if "*" in self.capabilities:
            raise ValueError("ExecutionPermit capabilities must be explicit; wildcard is forbidden")
        digest = self.action_intent_hash.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("ExecutionPermit.action_intent_hash must be a sha256 digest")
        if isinstance(self.peer_uid, bool) or not isinstance(self.peer_uid, int):
            raise ValueError("ExecutionPermit.peer_uid must be an integer")
        if self.peer_uid < 0:
            raise ValueError("ExecutionPermit.peer_uid must be non-negative")
        if isinstance(self.max_uses, bool) or not isinstance(self.max_uses, int):
            raise ValueError("ExecutionPermit.max_uses must be an integer")
        if self.max_uses < 1:
            raise ValueError("ExecutionPermit.max_uses must be positive")
        if self.expires_at.tzinfo is None:
            raise ValueError("ExecutionPermit.expires_at must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PERMIT_SCHEMA_VERSION,
            "permit_id": self.permit_id,
            "principal_id": self.principal_id,
            "peer_uid": self.peer_uid,
            "body_id": self.body_id,
            "body_snapshot_hash": self.body_snapshot_hash,
            "capabilities": list(self.capabilities),
            "action_intent_hash": self.action_intent_hash,
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "max_uses": self.max_uses,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionPermit:
        if value.get("schema_version") != PERMIT_SCHEMA_VERSION:
            raise ValueError("ExecutionPermit schema_version is invalid")
        capabilities = value.get("capabilities")
        if not isinstance(capabilities, list) or not all(
            isinstance(item, str) for item in capabilities
        ):
            raise ValueError("ExecutionPermit capabilities must be a string list")
        expires_at = value.get("expires_at")
        if not isinstance(expires_at, str):
            raise ValueError("ExecutionPermit expires_at must be an ISO timestamp")
        try:
            parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("ExecutionPermit expires_at must be an ISO timestamp") from exc
        return cls(
            permit_id=_strict_string(value.get("permit_id"), "permit_id"),
            principal_id=_strict_string(value.get("principal_id"), "principal_id"),
            peer_uid=_strict_int(value.get("peer_uid"), "peer_uid"),
            body_id=_strict_string(value.get("body_id"), "body_id"),
            body_snapshot_hash=_strict_string(
                value.get("body_snapshot_hash"),
                "body_snapshot_hash",
            ),
            capabilities=tuple(capabilities),
            action_intent_hash=_strict_string(
                value.get("action_intent_hash"),
                "action_intent_hash",
            ),
            expires_at=parsed_expiry,
            max_uses=_strict_int(value.get("max_uses"), "max_uses"),
        )


@dataclass(frozen=True)
class PermitDecision:
    """Result of daemon-side authorization."""

    allowed: bool
    code: str
    message: str
    authorization: AuthorizationContext


class PermitAuthority:
    """Daemon permit authority that never trusts caller ``approved`` flags."""

    def __init__(self, *, ledger: DaemonLedger | None = None) -> None:
        self._permits: dict[str, ExecutionPermit] = {}
        self._uses: dict[str, int] = {}
        self._consumed_actions: dict[str, str] = {}
        self._lock = threading.RLock()
        self.ledger = ledger
        if ledger is not None:
            self._restore_from_ledger()

    def register(self, permit: ExecutionPermit) -> None:
        """Register a permit from a trusted daemon/operator integration."""

        with self._lock:
            existing = self._permits.get(permit.permit_id)
            if existing is not None:
                if self.ledger is not None and existing == permit:
                    return
                raise ValueError(f"Permit {permit.permit_id!r} is already registered")
            if self.ledger is not None:
                self.ledger.append(
                    "PERMIT_REGISTERED",
                    entity_kind="PERMIT",
                    entity_id=permit.permit_id,
                    payload={"permit": permit.to_dict()},
                )
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
            if consumed_permit is not None:
                return PermitDecision(
                    False,
                    "PERMIT_ACTION_ID_CONFLICT",
                    "Action id was already consumed under a different daemon permit.",
                    denied,
                )
            uses = self._uses.get(permit_id, 0)
            if uses >= permit.max_uses:
                return PermitDecision(
                    False,
                    "PERMIT_EXHAUSTED",
                    f"Permit {permit_id!r} has no remaining uses.",
                    denied,
                )
            if self.ledger is not None:
                self.ledger.append(
                    "PERMIT_CONSUMED",
                    entity_kind="PERMIT",
                    entity_id=permit_id,
                    payload={
                        "permit_id": permit_id,
                        "action_id": action.action_id,
                        "peer_uid": peer.uid,
                        "action_intent_hash": action_intent_hash(action),
                    },
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

    def _restore_from_ledger(self) -> None:
        from rosclaw.daemon.ledger import LedgerIntegrityError

        assert self.ledger is not None
        for event in self.ledger.events(entity_kind="PERMIT"):
            if event.event_type == "PERMIT_REGISTERED":
                raw_permit = event.payload.get("permit")
                if not isinstance(raw_permit, dict):
                    raise LedgerIntegrityError("persisted permit registration is invalid")
                try:
                    permit = ExecutionPermit.from_dict(raw_permit)
                except (TypeError, ValueError) as exc:
                    raise LedgerIntegrityError("persisted permit registration is invalid") from exc
                if permit.permit_id != event.entity_id:
                    raise LedgerIntegrityError(
                        "persisted permit id does not match its ledger entity"
                    )
                existing = self._permits.get(permit.permit_id)
                if existing is not None:
                    raise LedgerIntegrityError("persisted permit has duplicate registration")
                self._permits[permit.permit_id] = permit
                self._uses.setdefault(permit.permit_id, 0)
                continue
            if event.event_type == "PERMIT_CONSUMED":
                try:
                    permit_id = _strict_string(event.payload.get("permit_id"), "permit_id")
                    action_id = _strict_string(event.payload.get("action_id"), "action_id")
                    peer_uid = _strict_int(event.payload.get("peer_uid"), "peer_uid")
                    intent_hash = _strict_string(
                        event.payload.get("action_intent_hash"),
                        "action_intent_hash",
                    )
                except ValueError as exc:
                    raise LedgerIntegrityError("persisted permit consumption is invalid") from exc
                if permit_id != event.entity_id or permit_id not in self._permits:
                    raise LedgerIntegrityError("persisted permit consumption is invalid")
                existing_permit = self._consumed_actions.get(action_id)
                if existing_permit is not None:
                    raise LedgerIntegrityError(
                        "persisted action consumes more than one permit event"
                    )
                permit = self._permits[permit_id]
                if peer_uid != permit.peer_uid or intent_hash != permit.action_intent_hash:
                    raise LedgerIntegrityError(
                        "persisted permit consumption does not match its permit binding"
                    )
                next_uses = self._uses.get(permit_id, 0) + 1
                if next_uses > permit.max_uses:
                    raise LedgerIntegrityError(
                        "persisted permit consumption exceeds permit max_uses"
                    )
                self._consumed_actions[action_id] = permit_id
                self._uses[permit_id] = next_uses
                continue
            raise LedgerIntegrityError(f"unsupported permit ledger event: {event.event_type!r}")


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


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ExecutionPermit.{name} must be an integer")
    return value


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ExecutionPermit.{name} must be a non-empty string")
    return value


__all__ = [
    "ExecutionPermit",
    "PERMIT_SCHEMA_VERSION",
    "PermitAuthority",
    "PermitDecision",
    "action_intent_hash",
]
