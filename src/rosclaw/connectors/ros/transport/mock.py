"""ROS Connector - Mock transport for tests and dry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint, RosTransportResult


@dataclass
class MockTransport:
    """In-memory mock ROS transport with programmable responses.

    Useful for unit tests and offline manifest compilation.
    """

    endpoint: RosbridgeEndpoint = field(default_factory=RosbridgeEndpoint)
    responses: list[RosTransportResult] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    dry_run: bool = False

    def connect(self) -> RosTransportResult:
        self.connected = True
        return RosTransportResult(ok=True, data={"mock": True})

    def close(self) -> None:
        self.connected = False

    def send(self, message: dict[str, Any]) -> RosTransportResult:
        self.requests.append(message)
        if self.dry_run:
            return RosTransportResult(ok=True, data={"dry_run": True})
        return self._next_response() or RosTransportResult(ok=True)

    def receive(self, timeout_sec: float | None = None) -> RosTransportResult:
        if self.dry_run:
            return RosTransportResult(ok=False, error="dry_run: no receive")
        return self._next_response() or RosTransportResult(
            ok=False, error="no mock response queued"
        )

    def request(
        self,
        message: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> RosTransportResult:
        self.requests.append(message)
        if self.dry_run:
            return RosTransportResult(ok=True, data={"dry_run": True})
        response = self._next_response()
        if response is None:
            return RosTransportResult(ok=False, error="no mock response queued")
        # Inject request id if caller provided one for traceability.
        request_id = message.get("id")
        if request_id and response.request_id is None:
            response.request_id = request_id
        return response

    def queue_response(self, result: RosTransportResult) -> None:
        self.responses.append(result)

    def queue_json(self, data: dict[str, Any], request_id: str | None = None) -> None:
        self.responses.append(
            RosTransportResult(ok=True, data=data, request_id=request_id)
        )

    def _next_response(self) -> RosTransportResult | None:
        if self.responses:
            return self.responses.pop(0)
        return None

    def call_service(
        self,
        service: str,
        args: dict[str, Any],
        service_type: str | None = None,
        timeout_sec: float | None = None,
    ) -> RosTransportResult:
        return self.request(
            {"op": "call_service", "service": service, "args": args}
        )

    def advertise(
        self,
        topic: str,
        msg_type: str,
        latch: bool = False,
        queue_size: int = 1,
    ) -> RosTransportResult:
        return self.request(
            {"op": "advertise", "topic": topic, "type": msg_type}
        )

    def unadvertise(self, topic: str) -> RosTransportResult:
        return self.request({"op": "unadvertise", "topic": topic})

    def publish(self, topic: str, msg: dict[str, Any]) -> RosTransportResult:
        return self.send({"op": "publish", "topic": topic, "msg": msg})

    def subscribe_once(
        self,
        topic: str,
        msg_type: str | None = None,
        timeout_sec: float = 5.0,
        throttle_rate_ms: int = 0,
    ) -> RosTransportResult:
        return self.request({"op": "subscribe", "topic": topic})

    def subscribe_for_duration(
        self,
        topic: str,
        msg_type: str | None = None,
        duration_sec: float = 1.0,
        throttle_rate_ms: int = 0,
    ) -> RosTransportResult:
        return RosTransportResult(
            ok=True,
            data={"topic": topic, "messages": [], "count": 0},
        )
