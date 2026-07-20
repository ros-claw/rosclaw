"""Strict, versioned framing for the local rosclawd Unix socket."""

from __future__ import annotations

import json
import socket
import struct
import uuid
from dataclasses import dataclass
from typing import Any

DAEMON_PROTOCOL_VERSION = "rosclaw.daemon.v1"
MAX_FRAME_BYTES = 1024 * 1024
MAX_REQUEST_ID_LENGTH = 128
MAX_METHOD_LENGTH = 96
_FRAME_HEADER_BYTES = 4


class DaemonProtocolError(ValueError):
    """A malformed or unsupported daemon protocol message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PeerCredentials:
    """Kernel-authenticated identity of one Unix-socket peer."""

    pid: int
    uid: int
    gid: int

    def to_dict(self) -> dict[str, int]:
        return {"pid": self.pid, "uid": self.uid, "gid": self.gid}


@dataclass(frozen=True)
class DaemonRequest:
    """Validated request accepted by rosclawd."""

    protocol_version: str
    request_id: str
    method: str
    params: dict[str, Any]


def make_request(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a daemon request object."""

    return {
        "protocol_version": DAEMON_PROTOCOL_VERSION,
        "request_id": request_id or f"request_{uuid.uuid4().hex}",
        "method": method,
        "params": params or {},
    }


def decode_request(payload: dict[str, Any]) -> DaemonRequest:
    """Validate and decode one request object."""

    protocol_version = payload.get("protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise DaemonProtocolError("INVALID_PROTOCOL", "protocol_version is required")
    if protocol_version != DAEMON_PROTOCOL_VERSION:
        raise DaemonProtocolError(
            "UNSUPPORTED_PROTOCOL",
            (
                f"Unsupported daemon protocol {protocol_version!r}; "
                f"expected {DAEMON_PROTOCOL_VERSION!r}"
            ),
        )

    request_id = payload.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > MAX_REQUEST_ID_LENGTH
    ):
        raise DaemonProtocolError(
            "INVALID_REQUEST_ID",
            f"request_id must be 1..{MAX_REQUEST_ID_LENGTH} characters",
        )

    method = payload.get("method")
    if not isinstance(method, str) or not method.strip() or len(method) > MAX_METHOD_LENGTH:
        raise DaemonProtocolError(
            "INVALID_METHOD",
            f"method must be 1..{MAX_METHOD_LENGTH} characters",
        )

    params = payload.get("params")
    if not isinstance(params, dict):
        raise DaemonProtocolError("INVALID_PARAMS", "params must be a JSON object")

    return DaemonRequest(
        protocol_version=protocol_version,
        request_id=request_id,
        method=method,
        params=params,
    )


def make_response(
    request_id: str,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a response containing exactly one result or error."""

    if (result is None) == (error is None):
        raise ValueError("response requires exactly one of result or error")
    response: dict[str, Any] = {
        "protocol_version": DAEMON_PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": error is None,
    }
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return response


def decode_response(payload: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    """Validate a response and return its result object."""

    if payload.get("protocol_version") != DAEMON_PROTOCOL_VERSION:
        raise DaemonProtocolError(
            "INVALID_RESPONSE_PROTOCOL",
            "rosclawd returned an unsupported protocol version",
        )
    if payload.get("request_id") != request_id:
        raise DaemonProtocolError(
            "RESPONSE_ID_MISMATCH",
            "rosclawd response does not match the request_id",
        )
    if payload.get("ok") is True:
        result = payload.get("result")
        if not isinstance(result, dict):
            raise DaemonProtocolError("INVALID_RESPONSE", "response result must be an object")
        return result
    if payload.get("ok") is False:
        error = payload.get("error")
        if not isinstance(error, dict):
            raise DaemonProtocolError("INVALID_RESPONSE", "response error must be an object")
        code = str(error.get("code", "DAEMON_REQUEST_FAILED"))
        message = str(error.get("message", "rosclawd rejected the request"))
        raise DaemonProtocolError(code, message)
    raise DaemonProtocolError("INVALID_RESPONSE", "response ok field must be boolean")


def encode_frame(payload: Any, *, max_bytes: int = MAX_FRAME_BYTES) -> bytes:
    """Encode one length-prefixed JSON frame."""

    try:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DaemonProtocolError("INVALID_MESSAGE", f"Message is not valid JSON: {exc}") from exc
    if not body:
        raise DaemonProtocolError("INVALID_MESSAGE", "Message body must not be empty")
    if len(body) > max_bytes:
        raise DaemonProtocolError(
            "FRAME_TOO_LARGE",
            f"Message is {len(body)} bytes; maximum is {max_bytes}",
        )
    return struct.pack("!I", len(body)) + body


def send_frame(
    connection: socket.socket,
    payload: dict[str, Any],
    *,
    max_bytes: int = MAX_FRAME_BYTES,
) -> None:
    """Send one complete frame."""

    connection.sendall(encode_frame(payload, max_bytes=max_bytes))


def receive_frame(
    connection: socket.socket,
    *,
    max_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Read one complete frame without accepting unbounded input."""

    header = _receive_exact(connection, _FRAME_HEADER_BYTES)
    length = struct.unpack("!I", header)[0]
    if length == 0:
        raise DaemonProtocolError("INVALID_MESSAGE", "Message body must not be empty")
    if length > max_bytes:
        raise DaemonProtocolError(
            "FRAME_TOO_LARGE",
            f"Message declares {length} bytes; maximum is {max_bytes}",
        )
    body = _receive_exact(connection, length)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DaemonProtocolError("INVALID_JSON", f"Invalid JSON frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise DaemonProtocolError("INVALID_MESSAGE", "Top-level message must be a JSON object")
    return payload


def get_peer_credentials(connection: socket.socket) -> PeerCredentials:
    """Return Linux kernel-authenticated peer credentials."""

    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        raise DaemonProtocolError(
            "PEER_CREDENTIALS_UNAVAILABLE",
            "This platform does not expose SO_PEERCRED for Unix sockets",
        )
    raw = connection.getsockopt(socket.SOL_SOCKET, option, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentials(pid=pid, uid=uid, gid=gid)


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise DaemonProtocolError(
                "TRUNCATED_FRAME",
                f"Connection closed with {remaining} frame bytes missing",
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "DAEMON_PROTOCOL_VERSION",
    "MAX_FRAME_BYTES",
    "DaemonProtocolError",
    "DaemonRequest",
    "PeerCredentials",
    "decode_request",
    "decode_response",
    "encode_frame",
    "get_peer_credentials",
    "make_request",
    "make_response",
    "receive_frame",
    "send_frame",
]
