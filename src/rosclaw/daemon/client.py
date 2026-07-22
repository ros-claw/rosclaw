"""Northbound client for the local rosclawd control socket."""

from __future__ import annotations

import os
import socket
import stat
import time
from pathlib import Path
from typing import Any

from rosclaw.daemon.protocol import (
    DaemonProtocolError,
    decode_response,
    get_peer_credentials,
    make_request,
    receive_frame,
    send_frame,
)
from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.kernel import ActionEnvelope

DEFAULT_DAEMON_TIMEOUT_SEC = 5.0


class DaemonClientError(RuntimeError):
    """Base class for structured rosclawd client failures."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class DaemonUnavailableError(DaemonClientError):
    """The daemon socket is absent or not accepting connections."""


class DaemonSecurityError(DaemonClientError):
    """The daemon socket path violates a local security invariant."""


class DaemonRequestError(DaemonClientError):
    """rosclawd rejected a well-formed request."""


def get_daemon_socket_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit, environment, or workspace daemon socket path."""

    configured = path or os.environ.get("ROSCLAW_DAEMON_SOCKET")
    if configured:
        return Path(configured).expanduser()
    return get_rosclaw_home() / "run" / "rosclawd.sock"


class DaemonClient:
    """Bounded request/response client; every call uses a fresh Unix socket."""

    def __init__(
        self,
        *,
        socket_path: str | Path | None = None,
        timeout_sec: float = DEFAULT_DAEMON_TIMEOUT_SEC,
        expected_daemon_uid: int | None = None,
    ):
        self.socket_path = get_daemon_socket_path(socket_path)
        self.timeout_sec = max(0.01, float(timeout_sec))
        configured_uid = os.environ.get("ROSCLAW_DAEMON_UID")
        if expected_daemon_uid is None and configured_uid:
            try:
                expected_daemon_uid = int(configured_uid)
            except ValueError as exc:
                raise ValueError("ROSCLAW_DAEMON_UID must be a non-negative integer") from exc
        if expected_daemon_uid is not None and expected_daemon_uid < 0:
            raise ValueError("expected_daemon_uid must be non-negative")
        self.expected_daemon_uid = expected_daemon_uid
        self._action_leases: dict[str, tuple[str, float, float]] = {}

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Call one allowlisted daemon method."""

        socket_metadata = self._validate_socket_path()
        request = make_request(method, params)
        request_id = str(request["request_id"])
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout_sec or self.timeout_sec)
        try:
            connection.connect(str(self.socket_path))
            daemon_peer = get_peer_credentials(connection)
            if daemon_peer.uid != socket_metadata.st_uid:
                raise DaemonSecurityError(
                    "DAEMON_UID_MISMATCH",
                    (
                        "rosclawd peer UID does not own its socket "
                        f"({daemon_peer.uid} != {socket_metadata.st_uid})"
                    ),
                    details={
                        "socket_path": str(self.socket_path),
                        "peer_uid": daemon_peer.uid,
                        "socket_uid": socket_metadata.st_uid,
                    },
                )
            if self.expected_daemon_uid is not None and daemon_peer.uid != self.expected_daemon_uid:
                raise DaemonSecurityError(
                    "UNEXPECTED_DAEMON_UID",
                    (
                        f"rosclawd peer UID {daemon_peer.uid} does not match configured "
                        f"UID {self.expected_daemon_uid}"
                    ),
                    details={
                        "socket_path": str(self.socket_path),
                        "peer_uid": daemon_peer.uid,
                        "expected_uid": self.expected_daemon_uid,
                    },
                )
            send_frame(connection, request)
            payload = receive_frame(connection)
        except DaemonProtocolError as exc:
            raise DaemonRequestError(exc.code, exc.message) from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise DaemonUnavailableError(
                "DAEMON_UNAVAILABLE",
                f"rosclawd is unavailable at {self.socket_path}: {exc}",
                details={"socket_path": str(self.socket_path)},
            ) from exc
        finally:
            connection.close()

        try:
            result = decode_response(payload, request_id=request_id)
        except DaemonProtocolError as exc:
            raise DaemonRequestError(exc.code, exc.message) from exc
        result.setdefault("daemon_peer", daemon_peer.to_dict())
        return result

    def get_runtime_status(self) -> dict[str, Any]:
        return self.call("runtime.status")

    def request_action(self, action: ActionEnvelope | dict[str, Any]) -> dict[str, Any]:
        payload = action.to_dict() if isinstance(action, ActionEnvelope) else action
        result = self.call("action.request", {"action": payload})
        action_id = result.get("action_id")
        session_id = result.get("session_id")
        lease = result.get("action_lease")
        if isinstance(action_id, str) and isinstance(session_id, str) and isinstance(lease, dict):
            interval_ms = lease.get("renew_interval_ms", 3_000)
            if isinstance(interval_ms, int) and not isinstance(interval_ms, bool):
                interval_sec = max(0.1, interval_ms / 1000.0)
                self._action_leases[action_id] = (
                    session_id,
                    time.monotonic() + interval_sec,
                    interval_sec,
                )
        return result

    def get_action_status(self, action_id: str) -> dict[str, Any]:
        return self.call("action.status", {"action_id": action_id})

    def get_execution_receipt(self, action_id: str) -> dict[str, Any]:
        return self.call("action.receipt", {"action_id": action_id})

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        return self.call("action.cancel", {"action_id": action_id})

    def create_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        agent_framework: str,
        body_scope: list[str],
        capability_scope: list[str],
        ttl_ms: int = 10_000,
    ) -> dict[str, Any]:
        return self.call(
            "session.create",
            {
                "session_id": session_id,
                "actor_id": actor_id,
                "agent_framework": agent_framework,
                "body_scope": body_scope,
                "capability_scope": capability_scope,
                "ttl_ms": ttl_ms,
            },
        )

    def heartbeat_session(self, session_id: str) -> dict[str, Any]:
        return self.call("session.heartbeat", {"session_id": session_id})

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.call("session.status", {"session_id": session_id})

    def close_session(self, session_id: str, *, reason: str = "client_closed") -> dict[str, Any]:
        return self.call("session.close", {"session_id": session_id, "reason": reason})

    def renew_action_lease(self, action_id: str, session_id: str) -> dict[str, Any]:
        return self.call(
            "action.lease.renew",
            {"action_id": action_id, "session_id": session_id},
        )

    def arm_runtime(self, reason: str) -> dict[str, Any]:
        return self.call("runtime.arm", {"reason": reason})

    def issue_execution_permit(
        self,
        action: ActionEnvelope | dict[str, Any],
        *,
        principal_id: str,
        target_peer_uid: int,
        expires_in_sec: float = 60.0,
        reason: str,
    ) -> dict[str, Any]:
        """Request an audited permit; rosclawd accepts only its service UID."""

        payload = action.to_dict() if isinstance(action, ActionEnvelope) else action
        return self.call(
            "permit.issue",
            {
                "action": payload,
                "principal_id": principal_id,
                "target_peer_uid": target_peer_uid,
                "expires_in_sec": expires_in_sec,
                "reason": reason,
            },
        )

    def disarm_runtime(self, reason: str) -> dict[str, Any]:
        return self.call("runtime.disarm", {"reason": reason})

    def get_worker_status(self, worker_id: str | None = None) -> dict[str, Any]:
        return self.call("worker.status", {"worker_id": worker_id} if worker_id else {})

    def control_worker(self, operation: str, worker_id: str) -> dict[str, Any]:
        if operation not in {"start", "stop", "restart"}:
            raise ValueError("operation must be start, stop, or restart")
        return self.call(f"worker.{operation}", {"worker_id": worker_id})

    def wait_for_action(
        self,
        action_id: str,
        *,
        timeout_sec: float = 30.0,
        poll_interval_sec: float = 0.01,
    ) -> dict[str, Any]:
        """Poll until an action reaches a terminal scheduler state."""

        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            status = self.get_action_status(action_id)
            if status.get("state") in {"FINISHED", "CANCELLED"}:
                self._action_leases.pop(action_id, None)
                return status
            lease = self._action_leases.get(action_id)
            if lease is not None and time.monotonic() >= lease[1]:
                session_id, _next_renewal, interval_sec = lease
                self.renew_action_lease(action_id, session_id)
                self._action_leases[action_id] = (
                    session_id,
                    time.monotonic() + interval_sec,
                    interval_sec,
                )
            if time.monotonic() >= deadline:
                raise DaemonRequestError(
                    "ACTION_WAIT_TIMEOUT",
                    f"Timed out waiting for action {action_id!r}.",
                )
            time.sleep(max(0.001, poll_interval_sec))

    def emergency_stop(
        self,
        reason: str,
        *,
        source: str = "daemon.client",
        timeout_sec: float = 1.0,
    ) -> dict[str, Any]:
        return self.call(
            "safety.emergency_stop",
            {
                "reason": reason,
                "source": source,
                "timeout_sec": timeout_sec,
            },
            timeout_sec=max(self.timeout_sec, timeout_sec + 1.0),
        )

    def acknowledge_recovery(self, reason: str) -> dict[str, Any]:
        """Persist operator review; the daemon accepts only its service UID."""

        return self.call("runtime.recovery.acknowledge", {"reason": reason})

    def shutdown(self) -> dict[str, Any]:
        """Request shutdown; rosclawd permits this only to its own service UID."""

        return self.call("runtime.shutdown")

    def _validate_socket_path(self) -> os.stat_result:
        encoded_length = len(os.fsencode(self.socket_path))
        if encoded_length >= 104:
            raise DaemonSecurityError(
                "SOCKET_PATH_TOO_LONG",
                f"Unix socket path is too long ({encoded_length} bytes): {self.socket_path}",
            )
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError as exc:
            raise DaemonUnavailableError(
                "DAEMON_UNAVAILABLE",
                f"rosclawd socket does not exist: {self.socket_path}",
                details={"socket_path": str(self.socket_path)},
            ) from exc
        if not stat.S_ISSOCK(metadata.st_mode):
            raise DaemonSecurityError(
                "UNSAFE_SOCKET_PATH",
                f"Refusing non-socket rosclawd path: {self.socket_path}",
            )
        if stat.S_IMODE(metadata.st_mode) & stat.S_IWOTH:
            raise DaemonSecurityError(
                "WORLD_WRITABLE_SOCKET",
                f"Refusing world-writable rosclawd socket: {self.socket_path}",
            )
        parent = self.socket_path.parent
        parent_metadata = parent.lstat()
        parent_mode = stat.S_IMODE(parent_metadata.st_mode)
        if not stat.S_ISDIR(parent_metadata.st_mode) or parent.is_symlink():
            raise DaemonSecurityError(
                "UNSAFE_SOCKET_DIRECTORY",
                f"Refusing unsafe rosclawd socket directory: {parent}",
            )
        if parent_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise DaemonSecurityError(
                "WRITABLE_SOCKET_DIRECTORY",
                f"Refusing group/world-writable rosclawd socket directory: {parent}",
            )
        if parent_metadata.st_uid not in {0, metadata.st_uid}:
            raise DaemonSecurityError(
                "SOCKET_DIRECTORY_OWNER_MISMATCH",
                (f"rosclawd socket directory is not owned by root or the socket owner: {parent}"),
            )
        return metadata


__all__ = [
    "DEFAULT_DAEMON_TIMEOUT_SEC",
    "DaemonClient",
    "DaemonClientError",
    "DaemonRequestError",
    "DaemonSecurityError",
    "DaemonUnavailableError",
    "get_daemon_socket_path",
]
