"""In-memory Agent Session ownership and heartbeat leases."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.kernel import ActionEnvelope

SESSION_SCHEMA_VERSION = "rosclaw.daemon.session.v1"
DEFAULT_SESSION_TTL_MS = 10_000
MIN_SESSION_TTL_MS = 300
MAX_SESSION_TTL_MS = 3_600_000


class SessionError(ValueError):
    """A structured Agent Session validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SessionState(StrEnum):
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    CLOSED = "CLOSED"


@dataclass
class AgentSession:
    """One UID-bound Agent ownership lease."""

    session_id: str
    actor_id: str
    agent_framework: str
    peer_uid: int
    body_scope: tuple[str, ...]
    capability_scope: tuple[str, ...]
    ttl_ms: int
    created_at: datetime
    last_heartbeat: datetime
    expires_at: datetime
    expires_monotonic: float
    state: SessionState = SessionState.ACTIVE
    loss_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "agent_framework": self.agent_framework,
            "peer_uid": self.peer_uid,
            "body_scope": list(self.body_scope),
            "capability_scope": list(self.capability_scope),
            "ttl_ms": self.ttl_ms,
            "state": self.state.value,
            "created_at": _iso(self.created_at),
            "last_heartbeat": _iso(self.last_heartbeat),
            "expires_at": _iso(self.expires_at),
            "loss_reason": self.loss_reason,
        }


class SessionManager:
    """Own Agent Sessions without depending on the durable ledger."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        agent_framework: str,
        body_scope: list[str] | tuple[str, ...],
        capability_scope: list[str] | tuple[str, ...],
        ttl_ms: int,
        peer: PeerCredentials,
    ) -> AgentSession:
        session_id = _identifier(session_id, "session_id")
        actor_id = _identifier(actor_id, "actor_id")
        agent_framework = _identifier(agent_framework, "agent_framework")
        normalized_bodies = _scope(body_scope, "body_scope")
        normalized_capabilities = _scope(capability_scope, "capability_scope")
        ttl_ms = _ttl(ttl_ms)
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                expected = (
                    actor_id,
                    agent_framework,
                    peer.uid,
                    normalized_bodies,
                    normalized_capabilities,
                    ttl_ms,
                )
                actual = (
                    existing.actor_id,
                    existing.agent_framework,
                    existing.peer_uid,
                    existing.body_scope,
                    existing.capability_scope,
                    existing.ttl_ms,
                )
                if existing.state is SessionState.ACTIVE and actual == expected:
                    return existing
                raise SessionError(
                    "SESSION_ID_CONFLICT",
                    "session_id is already bound to another or terminal Agent Session",
                )
            now = datetime.now(UTC)
            session = AgentSession(
                session_id=session_id,
                actor_id=actor_id,
                agent_framework=agent_framework,
                peer_uid=peer.uid,
                body_scope=normalized_bodies,
                capability_scope=normalized_capabilities,
                ttl_ms=ttl_ms,
                created_at=now,
                last_heartbeat=now,
                expires_at=now + timedelta(milliseconds=ttl_ms),
                expires_monotonic=time.monotonic() + ttl_ms / 1000.0,
            )
            self._sessions[session_id] = session
            return session

    def adopt_action(self, action: ActionEnvelope, peer: PeerCredentials) -> AgentSession:
        """Create a bounded compatibility Session for a legacy action request."""

        return self.create_session(
            session_id=action.session_id,
            actor_id=action.actor_id,
            agent_framework=action.agent_framework,
            body_scope=[action.body_id],
            capability_scope=[action.capability_id],
            ttl_ms=max(DEFAULT_SESSION_TTL_MS, action.lease_ttl_ms),
            peer=peer,
        )

    def require_action(self, action: ActionEnvelope, peer: PeerCredentials) -> AgentSession:
        with self._lock:
            session = self._sessions.get(action.session_id)
            if session is None:
                raise SessionError("SESSION_NOT_FOUND", "Agent Session does not exist")
            self._require_owner(session, peer)
            if session.state is not SessionState.ACTIVE:
                raise SessionError("SESSION_NOT_ACTIVE", "Agent Session is not active")
            if time.monotonic() >= session.expires_monotonic:
                session.state = SessionState.LOST
                session.loss_reason = "heartbeat_timeout"
                raise SessionError("SESSION_EXPIRED", "Agent Session heartbeat expired")
            if session.actor_id != action.actor_id:
                raise SessionError("SESSION_ACTOR_MISMATCH", "Action actor does not own Session")
            if action.body_id not in session.body_scope:
                raise SessionError("SESSION_BODY_SCOPE_MISMATCH", "Body is outside Session scope")
            if action.capability_id not in session.capability_scope:
                raise SessionError(
                    "SESSION_CAPABILITY_SCOPE_MISMATCH",
                    "Capability is outside Session scope",
                )
            return session

    def heartbeat(self, session_id: str, peer: PeerCredentials) -> AgentSession:
        with self._lock:
            session = self._get(session_id)
            self._require_owner(session, peer)
            if session.state is not SessionState.ACTIVE:
                raise SessionError("SESSION_NOT_ACTIVE", "Agent Session is not active")
            if time.monotonic() >= session.expires_monotonic:
                session.state = SessionState.LOST
                session.loss_reason = "heartbeat_timeout"
                raise SessionError("SESSION_EXPIRED", "Agent Session heartbeat expired")
            now = datetime.now(UTC)
            session.last_heartbeat = now
            session.expires_at = now + timedelta(milliseconds=session.ttl_ms)
            session.expires_monotonic = time.monotonic() + session.ttl_ms / 1000.0
            return session

    def close_session(
        self,
        session_id: str,
        peer: PeerCredentials,
        *,
        reason: str = "client_closed",
    ) -> AgentSession:
        with self._lock:
            session = self._get(session_id)
            self._require_owner(session, peer)
            if session.state is SessionState.ACTIVE:
                session.state = SessionState.CLOSED
                session.loss_reason = reason[:256]
            return session

    def expire_sessions(self, *, now_monotonic: float | None = None) -> list[AgentSession]:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        expired: list[AgentSession] = []
        with self._lock:
            for session in self._sessions.values():
                if session.state is SessionState.ACTIVE and now >= session.expires_monotonic:
                    session.state = SessionState.LOST
                    session.loss_reason = "heartbeat_timeout"
                    expired.append(session)
        return expired

    def get_session(self, session_id: str, peer: PeerCredentials) -> AgentSession:
        with self._lock:
            session = self._get(session_id)
            self._require_owner(session, peer)
            return session

    def status(self) -> dict[str, int]:
        with self._lock:
            counts = {state.value.lower(): 0 for state in SessionState}
            for session in self._sessions.values():
                counts[session.state.value.lower()] += 1
            return {"total": len(self._sessions), **counts}

    def _get(self, session_id: str) -> AgentSession:
        normalized = _identifier(session_id, "session_id")
        session = self._sessions.get(normalized)
        if session is None:
            raise SessionError("SESSION_NOT_FOUND", "Agent Session does not exist")
        return session

    @staticmethod
    def _require_owner(session: AgentSession, peer: PeerCredentials) -> None:
        if peer.uid != session.peer_uid:
            raise SessionError(
                "SESSION_OWNERSHIP_MISMATCH",
                "Authenticated Unix peer does not own this Agent Session",
            )


def _identifier(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SessionError(
            "INVALID_SESSION",
            f"{field} must contain 1..256 printable characters",
        )
    return value


def _scope(value: list[str] | tuple[str, ...], field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SessionError("INVALID_SESSION", f"{field} must be a non-empty string list")
    normalized = tuple(sorted({_identifier(item, field) for item in value}))
    if "*" in normalized:
        raise SessionError("INVALID_SESSION", f"{field} must use explicit identifiers")
    return normalized


def _ttl(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionError("INVALID_SESSION", "ttl_ms must be an integer")
    if not MIN_SESSION_TTL_MS <= value <= MAX_SESSION_TTL_MS:
        raise SessionError(
            "INVALID_SESSION",
            f"ttl_ms must be between {MIN_SESSION_TTL_MS} and {MAX_SESSION_TTL_MS}",
        )
    return value


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_SESSION_TTL_MS",
    "SESSION_SCHEMA_VERSION",
    "AgentSession",
    "SessionError",
    "SessionManager",
    "SessionState",
]
