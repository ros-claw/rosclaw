"""ROS Connector - Transport abstractions.

This module defines the protocol and dataclasses used by all ROS transports.
It intentionally does not import rclpy/rospy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RosbridgeEndpoint:
    """Connection endpoint for a rosbridge server."""

    host: str = "127.0.0.1"
    port: int = 9090
    scheme: str = "ws"
    timeout_sec: float = 5.0
    path: str = ""
    query: str = ""

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        suffix = self.path + ("?" + self.query if self.query else "")
        return f"{self.scheme}://{host}:{self.port}{suffix}"

    @classmethod
    def from_url(cls, url: str, timeout_sec: float = 5.0) -> RosbridgeEndpoint:
        """Parse an explicit websocket endpoint, preserving proxy route and IPv6."""
        from urllib.parse import urlsplit

        parsed = urlsplit(url if "://" in url else "ws://" + url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(
                "ROS_ENDPOINT_INVALID: ws/wss host URL required; embedded credentials are unsupported"
            )
        port = parsed.port if parsed.port is not None else 9090
        if not 1 <= port <= 65535:
            raise ValueError("ROS_ENDPOINT_INVALID: port must be 1..65535")
        return cls(
            host=parsed.hostname,
            port=port,
            scheme=parsed.scheme,
            timeout_sec=timeout_sec,
            path=parsed.path,
            query=parsed.query,
        )


@dataclass
class RosTransportResult:
    """Structured result from a ROS transport operation."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    raw: Any | None = None
    request_id: str | None = None

    @property
    def is_ok(self) -> bool:
        return self.ok and self.error is None


class RosTransport(Protocol):
    """Protocol for ROS transport implementations.

    Implementations must not import ROS Python client libraries.
    """

    def connect(self) -> RosTransportResult: ...
    def close(self) -> None: ...
    def request(
        self,
        message: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> RosTransportResult: ...
    def send(self, message: dict[str, Any]) -> RosTransportResult: ...
    def receive(self, timeout_sec: float | None = None) -> RosTransportResult: ...


class RosTransportError(Exception):
    """Base exception for ROS transport errors."""

    def __init__(self, message: str, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id


class RosConnectionError(RosTransportError):
    """Could not establish transport connection."""


class RosTimeoutError(RosTransportError):
    """Operation timed out."""


class RosSerializationError(RosTransportError):
    """Message could not be serialized or parsed."""


@dataclass
class RosbridgeMessage:
    """Canonical rosbridge protocol message envelope."""

    op: str
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"op": self.op}
        if self.request_id:
            msg["id"] = self.request_id
        msg.update(self.extra)
        return msg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RosbridgeMessage:
        op = data.get("op", "")
        request_id = data.get("id")
        extra = {k: v for k, v in data.items() if k not in ("op", "id")}
        return cls(op=op, request_id=request_id, extra=extra)
