"""Authenticated Unix-socket server for the rosclawd control plane."""

from __future__ import annotations

import contextlib
import errno
import grp
import os
import socket
import stat
import threading
from pathlib import Path
from typing import Any

from rosclaw.daemon.client import get_daemon_socket_path
from rosclaw.daemon.protocol import (
    DaemonProtocolError,
    PeerCredentials,
    decode_request,
    get_peer_credentials,
    make_response,
    receive_frame,
    send_frame,
)
from rosclaw.daemon.service import ControlPlaneError, DaemonControlPlane
from rosclaw.kernel import ActionEnvelope

_ALLOWED_METHODS = frozenset(
    {
        "runtime.status",
        "runtime.arm",
        "runtime.disarm",
        "runtime.recovery.acknowledge",
        "runtime.shutdown",
        "permit.issue",
        "action.request",
        "action.status",
        "action.receipt",
        "action.cancel",
        "action.lease.renew",
        "session.create",
        "session.heartbeat",
        "session.status",
        "session.close",
        "worker.status",
        "worker.start",
        "worker.stop",
        "worker.restart",
        "safety.emergency_stop",
    }
)


class RosclawDaemon:
    """Own one local socket and a daemon-side physical control plane."""

    def __init__(
        self,
        *,
        service: DaemonControlPlane,
        socket_path: str | Path | None = None,
        socket_mode: int = 0o600,
        socket_group: str | None = None,
        request_timeout_sec: float = 5.0,
        max_clients: int = 32,
    ):
        mode = stat.S_IMODE(socket_mode)
        if socket_mode != mode or mode & 0o007 or mode & 0o600 != 0o600:
            raise ValueError(
                "socket_mode must grant owner read/write and must not grant world access"
            )
        self.service = service
        self.socket_path = get_daemon_socket_path(socket_path)
        self.socket_mode = mode
        self.socket_group = socket_group
        self.request_timeout_sec = max(0.1, request_timeout_sec)
        self.max_clients = max(1, int(max_clients))
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._client_threads: set[threading.Thread] = set()
        self._client_slots = threading.BoundedSemaphore(self.max_clients)
        self._shutdown = threading.Event()
        self._lock = threading.RLock()
        self._socket_inode: int | None = None

    def start(self) -> None:
        """Bind a protected filesystem socket and start accepting clients."""

        with self._lock:
            if self._listener is not None:
                return
            group_id = self._socket_group_id()
            self._prepare_socket_path(group_id)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(self.socket_path))
                os.chmod(self.socket_path, self.socket_mode)
                if group_id is not None:
                    os.chown(self.socket_path, -1, group_id)
                listener.listen(32)
                listener.settimeout(0.2)
                self._socket_inode = self.socket_path.stat().st_ino
            except Exception:
                listener.close()
                self._unlink_owned_socket()
                raise
            self._listener = listener
            self._shutdown.clear()
            try:
                self.service.start()
                self._accept_thread = threading.Thread(
                    target=self._accept_loop,
                    name="rosclawd-accept",
                    daemon=True,
                )
                self._accept_thread.start()
            except Exception:
                self._listener = None
                self._accept_thread = None
                listener.close()
                with contextlib.suppress(Exception):
                    self.service.close()
                self._unlink_owned_socket()
                raise

    def serve_forever(self) -> None:
        """Run until SIGTERM, KeyboardInterrupt, or an authorized shutdown request."""

        try:
            self.start()
            while not self._shutdown.wait(0.2):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def request_shutdown(self) -> None:
        self._shutdown.set()
        listener = self._listener
        if listener is not None:
            with contextlib.suppress(OSError):
                listener.close()

    def stop(self) -> None:
        """Stop accepting, close the control plane, and remove only our socket."""

        with self._lock:
            listener = self._listener
            if listener is None and self._socket_inode is None:
                return
            self._shutdown.set()
            self._listener = None
            if listener is not None:
                with contextlib.suppress(OSError):
                    listener.close()
        current = threading.current_thread()
        if self._accept_thread is not None and self._accept_thread is not current:
            self._accept_thread.join(timeout=2.0)
        with self._lock:
            client_threads = list(self._client_threads)
        for thread in client_threads:
            if thread is not current:
                thread.join(timeout=self.request_timeout_sec + 1.0)
        self.service.close()
        self._unlink_owned_socket()

    def _prepare_socket_path(self, group_id: int | None) -> None:
        encoded_length = len(os.fsencode(self.socket_path))
        if encoded_length >= 104:
            raise RuntimeError(
                f"rosclawd socket path is too long ({encoded_length} bytes): {self.socket_path}"
            )
        parent = self.socket_path.parent
        if parent.exists() and parent.is_symlink():
            raise RuntimeError(f"rosclawd refusing symlink runtime directory: {parent}")
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_metadata = parent.lstat()
        parent_mode = stat.S_IMODE(parent_metadata.st_mode)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise RuntimeError(f"rosclawd runtime path is not a directory: {parent}")
        if parent_metadata.st_uid not in {0, os.geteuid()}:
            raise RuntimeError(
                "rosclawd runtime directory is owned by untrusted UID "
                f"{parent_metadata.st_uid}: {parent}"
            )
        if parent_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"rosclawd runtime directory must not be group/world writable: {parent}"
            )
        if group_id is not None and (
            parent_metadata.st_gid != group_id or not parent_mode & stat.S_IXGRP
        ):
            raise RuntimeError(
                "rosclawd group socket requires a runtime directory owned by "
                f"group {self.socket_group!r} with group traverse permission: {parent}"
            )

        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"rosclawd refusing to replace non-socket path: {self.socket_path}")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(str(self.socket_path))
        except FileNotFoundError:
            return
        except ConnectionRefusedError:
            self.socket_path.unlink()
        except TimeoutError as exc:
            raise RuntimeError(
                f"rosclawd refusing to replace an unresponsive socket: {self.socket_path}"
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.ECONNREFUSED, errno.ENOENT}:
                with contextlib.suppress(FileNotFoundError):
                    self.socket_path.unlink()
            else:
                raise RuntimeError(
                    f"rosclawd could not authenticate existing socket {self.socket_path}: {exc}"
                ) from exc
        else:
            raise RuntimeError(f"rosclawd is already listening at {self.socket_path}")
        finally:
            probe.close()

    def _socket_group_id(self) -> int | None:
        if not self.socket_group:
            return None
        try:
            return grp.getgrnam(self.socket_group).gr_gid
        except KeyError as exc:
            raise RuntimeError(
                f"rosclawd socket group does not exist: {self.socket_group!r}"
            ) from exc

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self._client_slots.acquire(blocking=False):
                connection.close()
                continue
            thread = threading.Thread(
                target=self._handle_client,
                args=(connection,),
                name="rosclawd-client",
                daemon=True,
            )
            try:
                with self._lock:
                    self._client_threads.add(thread)
                thread.start()
            except Exception:
                with self._lock:
                    self._client_threads.discard(thread)
                connection.close()
                self._client_slots.release()

    def _handle_client(self, connection: socket.socket) -> None:
        request_id = "unknown"
        shutdown_after_response = False
        try:
            connection.settimeout(self.request_timeout_sec)
            peer = get_peer_credentials(connection)
            payload = receive_frame(connection)
            request = decode_request(payload)
            request_id = request.request_id
            result, shutdown_after_response = self._dispatch(
                request.method,
                request.params,
                peer,
            )
            response = make_response(request_id, result=result)
        except (DaemonProtocolError, ControlPlaneError) as exc:
            response = make_response(
                request_id,
                error={"code": exc.code, "message": exc.message},
            )
        except Exception:
            response = make_response(
                request_id,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "rosclawd failed to process the request",
                },
            )
        try:
            send_frame(connection, response)
        except (BrokenPipeError, ConnectionError, OSError, DaemonProtocolError):
            pass
        finally:
            connection.close()
            with self._lock:
                self._client_threads.discard(threading.current_thread())
            self._client_slots.release()
        if shutdown_after_response:
            self.request_shutdown()

    def _dispatch(
        self,
        method: str,
        params: dict[str, Any],
        peer: PeerCredentials,
    ) -> tuple[dict[str, Any], bool]:
        if method not in _ALLOWED_METHODS:
            raise ControlPlaneError(
                "METHOD_NOT_ALLOWED",
                f"rosclawd does not expose method {method!r}",
            )
        if method == "runtime.status":
            return self.service.get_runtime_status(peer), False
        if method == "runtime.arm":
            return self.service.arm_runtime(
                _required_id(params, "reason", max_length=1024), peer
            ), False
        if method == "runtime.disarm":
            return (
                self.service.disarm_runtime(
                    _required_id(params, "reason", max_length=1024),
                    peer,
                ),
                False,
            )
        if method == "runtime.recovery.acknowledge":
            reason = _required_id(params, "reason", max_length=1024)
            return self.service.acknowledge_recovery(reason, peer), False
        if method == "runtime.shutdown":
            if peer.uid != os.geteuid():
                raise ControlPlaneError(
                    "PERMISSION_DENIED",
                    "Only the rosclawd service UID may request daemon shutdown",
                )
            return {"shutdown_requested": True}, True
        if method == "permit.issue":
            return (
                self.service.issue_execution_permit(
                    _required_action(params),
                    principal_id=_required_id(params, "principal_id"),
                    target_peer_uid=_required_int(params, "target_peer_uid"),
                    expires_in_sec=_required_number(params, "expires_in_sec"),
                    reason=_required_id(params, "reason", max_length=1024),
                    peer=peer,
                ),
                False,
            )
        if method == "action.request":
            return self.service.request_action(_required_action(params), peer), False
        if method == "action.status":
            return (
                self.service.get_action_status(
                    _required_id(params, "action_id"),
                    peer,
                ),
                False,
            )
        if method == "action.receipt":
            return (
                self.service.get_execution_receipt(
                    _required_id(params, "action_id"),
                    peer,
                ),
                False,
            )
        if method == "action.cancel":
            return (
                self.service.cancel_action(
                    _required_id(params, "action_id"),
                    peer,
                ),
                False,
            )
        if method == "action.lease.renew":
            return (
                self.service.renew_action_lease(
                    _required_id(params, "action_id"),
                    _required_id(params, "session_id"),
                    peer,
                ),
                False,
            )
        if method == "session.create":
            return (
                self.service.create_session(
                    session_id=_required_id(params, "session_id"),
                    actor_id=_required_id(params, "actor_id"),
                    agent_framework=_required_id(params, "agent_framework"),
                    body_scope=_required_string_list(params, "body_scope"),
                    capability_scope=_required_string_list(params, "capability_scope"),
                    ttl_ms=_required_int(params, "ttl_ms"),
                    peer=peer,
                ),
                False,
            )
        if method == "session.heartbeat":
            return (
                self.service.heartbeat_session(
                    _required_id(params, "session_id"),
                    peer,
                ),
                False,
            )
        if method == "session.status":
            return (
                self.service.get_session(
                    _required_id(params, "session_id"),
                    peer,
                ),
                False,
            )
        if method == "session.close":
            reason = params.get("reason", "client_closed")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 256:
                raise ControlPlaneError(
                    "INVALID_ARGUMENT",
                    "reason must contain 1 to 256 characters",
                )
            return (
                self.service.close_session(
                    _required_id(params, "session_id"),
                    peer,
                    reason=reason,
                ),
                False,
            )
        if method == "worker.status":
            worker_id = params.get("worker_id")
            if worker_id is not None:
                worker_id = _required_id(params, "worker_id", max_length=128)
            return self.service.get_worker_status(peer, worker_id=worker_id), False
        if method in {"worker.start", "worker.stop", "worker.restart"}:
            return (
                self.service.control_worker(
                    method.removeprefix("worker."),
                    _required_id(params, "worker_id", max_length=128),
                    peer,
                ),
                False,
            )
        if method == "safety.emergency_stop":
            reason = _required_id(params, "reason", max_length=1024)
            source = str(params.get("source", "rosclawd.client"))[:128]
            try:
                timeout_sec = float(params.get("timeout_sec", 1.0))
            except (TypeError, ValueError) as exc:
                raise ControlPlaneError(
                    "INVALID_TIMEOUT",
                    "timeout_sec must be numeric",
                ) from exc
            if not 0.0 <= timeout_sec <= 10.0:
                raise ControlPlaneError(
                    "INVALID_TIMEOUT",
                    "timeout_sec must be between 0 and 10 seconds",
                )
            return (
                self.service.emergency_stop(
                    reason,
                    source=source,
                    timeout_sec=timeout_sec,
                    peer=peer,
                ),
                False,
            )
        raise AssertionError(f"Unhandled allowlisted method: {method}")

    def _unlink_owned_socket(self) -> None:
        inode = self._socket_inode
        if inode is None:
            return
        try:
            metadata = self.socket_path.lstat()
            if stat.S_ISSOCK(metadata.st_mode) and metadata.st_ino == inode:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._socket_inode = None


def _required_id(
    params: dict[str, Any],
    key: str,
    *,
    max_length: int = 256,
) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ControlPlaneError(
            "INVALID_ARGUMENT",
            f"{key} must be a non-empty string of at most {max_length} characters",
        )
    return value


def _required_string_list(params: dict[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ControlPlaneError(
            "INVALID_ARGUMENT",
            f"{key} must be a non-empty string list",
        )
    return value


def _required_int(params: dict[str, Any], key: str) -> int:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlPlaneError("INVALID_ARGUMENT", f"{key} must be an integer")
    return value


def _required_number(params: dict[str, Any], key: str) -> float:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlPlaneError("INVALID_ARGUMENT", f"{key} must be numeric")
    return float(value)


def _required_action(params: dict[str, Any]) -> ActionEnvelope:
    action_payload = params.get("action")
    if not isinstance(action_payload, dict):
        raise ControlPlaneError("INVALID_ACTION", "action must be a JSON object")
    try:
        return ActionEnvelope.from_dict(action_payload)
    except (TypeError, ValueError, KeyError) as exc:
        raise ControlPlaneError("INVALID_ACTION", str(exc)) from exc


__all__ = ["RosclawDaemon"]
