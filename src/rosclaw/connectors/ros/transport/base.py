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

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @classmethod
    def from_url(cls, url: str, timeout_sec: float = 5.0) -> RosbridgeEndpoint:
        """Parse a rosbridge URL like ``ws://host:port``."""
        scheme = "ws"
        rest = url
        if "://" in url:
            scheme, rest = url.split("://", 1)
        if ":" in rest:
            host, port_str = rest.rsplit(":", 1)
            port = int(port_str)
        else:
            host = rest
            port = 9090
        return cls(host=host, port=port, scheme=scheme, timeout_sec=timeout_sec)


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
