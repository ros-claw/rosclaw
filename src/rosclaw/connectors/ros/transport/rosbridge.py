"""ROS Connector - rosbridge WebSocket transport.

Implements RosTransport over the rosbridge WebSocket protocol.
No ROS Python imports.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

from rosclaw.connectors.ros.transport.base import (
    RosbridgeEndpoint,
    RosTransportResult,
)

logger = logging.getLogger("rosclaw.connectors.ros.transport.rosbridge")


class RosbridgeTransport:
    """Thread-safe WebSocket transport for rosbridge.

    Uses lazy connection: the first request triggers ``connect()``.
    All public methods return ``RosTransportResult`` so callers do not
    need to catch transport exceptions.
    """

    def __init__(
        self,
        endpoint: RosbridgeEndpoint | None = None,
        max_retries: int = 2,
        dry_run: bool = False,
    ):
        self.endpoint = endpoint or RosbridgeEndpoint()
        self._ws: Any | None = None
        self._lock = threading.RLock()
        self._max_retries = max(0, max_retries)
        self._dry_run = dry_run
        self._closed = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> RosTransportResult:
        if self._dry_run:
            return RosTransportResult(ok=True, data={"dry_run": True})

        with self._lock:
            if self._closed:
                return RosTransportResult(
                    ok=False,
                    error="transport is closed",
                )
            if self._ws is not None and getattr(self._ws, "connected", False):
                return RosTransportResult(ok=True)

            try:
                import websocket

                self._ws = websocket.create_connection(
                    self.endpoint.url,
                    timeout=self.endpoint.timeout_sec,
                )
                logger.info("Connected to rosbridge at %s", self.endpoint.url)
                return RosTransportResult(ok=True)
            except ImportError as exc:
                error = (
                    "websocket-client is required for rosbridge transport; "
                    "install with: pip install websocket-client"
                )
                logger.error(error)
                return RosTransportResult(ok=False, error=error, raw=str(exc))
            except Exception as exc:
                error = f"Failed to connect to {self.endpoint.url}: {exc}"
                logger.error(error)
                self._ws = None
                return RosTransportResult(ok=False, error=error, raw=str(exc))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._ws is not None:
                try:
                    if getattr(self._ws, "connected", False):
                        self._ws.close()
                        logger.info("Closed rosbridge connection")
                except Exception as exc:
                    logger.warning("Error closing rosbridge connection: %s", exc)
                finally:
                    self._ws = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Low-level send / receive
    # ------------------------------------------------------------------
    def send(self, message: dict[str, Any]) -> RosTransportResult:
        if self._dry_run:
            return RosTransportResult(ok=True, data={"dry_run": True})

        conn = self.connect()
        if not conn.ok:
            return conn

        with self._lock:
            try:
                payload = json.dumps(message)
            except (TypeError, ValueError) as exc:
                error = f"JSON serialization failed: {exc}"
                logger.error(error)
                return RosTransportResult(ok=False, error=error, raw=str(exc))

            try:
                self._ws.send(payload)
                return RosTransportResult(ok=True)
            except Exception as exc:
                error = f"Send failed: {exc}"
                logger.error(error)
                self._invalidate()
                return RosTransportResult(ok=False, error=error, raw=str(exc))

    def receive(self, timeout_sec: float | None = None) -> RosTransportResult:
        if self._dry_run:
            return RosTransportResult(ok=False, error="dry_run: no receive")

        conn = self.connect()
        if not conn.ok:
            return conn

        with self._lock:
            actual_timeout = (
                timeout_sec if timeout_sec is not None else self.endpoint.timeout_sec
            )
            try:
                self._ws.settimeout(actual_timeout)
                raw = self._ws.recv()
            except Exception as exc:
                error = f"Receive failed or timed out after {actual_timeout}s: {exc}"
                logger.warning(error)
                self._invalidate()
                return RosTransportResult(ok=False, error=error, raw=str(exc))

            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                data = json.loads(raw)
                return RosTransportResult(ok=True, data=data, raw=raw)
            except (json.JSONDecodeError, TypeError) as exc:
                error = f"Invalid JSON from rosbridge: {exc}"
                logger.warning(error)
                return RosTransportResult(ok=False, error=error, raw=raw)

    # ------------------------------------------------------------------
    # Request/response helper
    # ------------------------------------------------------------------
    def request(
        self,
        message: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> RosTransportResult:
        """Send a message and wait for a matching response by id.

        The message is augmented with a generated ``id`` if it does not
        already contain one.  Responses are matched on ``id`` to avoid
        interleaving with unsolicited rosbridge messages.
        """
        request_id = message.get("id") or f"rosclaw_{uuid.uuid4().hex[:12]}"
        message = {**message, "id": request_id}

        if self._dry_run:
            return RosTransportResult(
                ok=True,
                data={"dry_run": True, "op": message.get("op")},
                request_id=request_id,
            )

        last_error: str | None = None
        for _attempt in range(self._max_retries + 1):
            send_result = self.send(message)
            if not send_result.ok:
                last_error = send_result.error
                self._invalidate()
                continue

            deadline = timeout_sec if timeout_sec is not None else self.endpoint.timeout_sec
            for _ in range(10):  # bounded spin on unsolicited messages
                recv_result = self.receive(timeout_sec=deadline)
                if not recv_result.ok:
                    last_error = recv_result.error
                    self._invalidate()
                    break
                data = recv_result.data or {}
                if data.get("id") == request_id:
                    return RosTransportResult(
                        ok=True,
                        data=data,
                        raw=recv_result.raw,
                        request_id=request_id,
                    )
                # Unsolicited message (publish, service response for another id)
                logger.debug("Ignoring unsolicited rosbridge message: %s", data.get("op"))

        return RosTransportResult(
            ok=False,
            error=last_error or "request failed",
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers around rosbridge ops
    # ------------------------------------------------------------------
    def call_service(
        self,
        service: str,
        args: dict[str, Any],
        service_type: str | None = None,
        timeout_sec: float | None = None,
    ) -> RosTransportResult:
        message = {
            "op": "call_service",
            "service": service,
            "args": args,
        }
        if service_type:
            message["type"] = service_type
        return self.request(message, timeout_sec=timeout_sec)

    def advertise(
        self,
        topic: str,
        msg_type: str,
        latch: bool = False,
        queue_size: int = 1,
    ) -> RosTransportResult:
        return self.request(
            {
                "op": "advertise",
                "topic": topic,
                "type": msg_type,
                "latch": latch,
                "queue_size": queue_size,
            },
        )

    def unadvertise(self, topic: str) -> RosTransportResult:
        return self.request({"op": "unadvertise", "topic": topic})

    def publish(
        self,
        topic: str,
        msg: dict[str, Any],
    ) -> RosTransportResult:
        return self.send({"op": "publish", "topic": topic, "msg": msg})

    def subscribe_once(
        self,
        topic: str,
        msg_type: str | None = None,
        timeout_sec: float = 5.0,
        throttle_rate_ms: int = 0,
    ) -> RosTransportResult:
        request_id = f"rosclaw_sub_{uuid.uuid4().hex[:12]}"
        subscribe_msg: dict[str, Any] = {
            "op": "subscribe",
            "id": request_id,
            "topic": topic,
            "throttle_rate": throttle_rate_ms,
        }
        if msg_type:
            subscribe_msg["type"] = msg_type

        send_result = self.send(subscribe_msg)
        if not send_result.ok:
            return send_result

        deadline = time.time() + timeout_sec
        last_error: str | None = None
        while time.time() < deadline:
            remaining = deadline - time.time()
            result = self.receive(timeout_sec=max(remaining, 0.1))
            if not result.ok:
                last_error = result.error
                break
            data = result.data or {}
            if data.get("op") == "publish" and data.get("topic") == topic:
                # Best-effort unsubscribe; do not block result on it.
                import contextlib

                with contextlib.suppress(Exception):
                    self.send({"op": "unsubscribe", "id": request_id, "topic": topic})
                return RosTransportResult(
                    ok=True, data=data, raw=result.raw, request_id=request_id
                )

        # Best-effort unsubscribe; do not block result on it.
        import contextlib

        with contextlib.suppress(Exception):
            self.send({"op": "unsubscribe", "id": request_id, "topic": topic})
        return RosTransportResult(
            ok=False,
            error=last_error or f"No message received on {topic} within {timeout_sec}s",
        )

    def subscribe_for_duration(
        self,
        topic: str,
        msg_type: str | None = None,
        duration_sec: float = 1.0,
        throttle_rate_ms: int = 0,
    ) -> RosTransportResult:
        """Subscribe to a topic and collect messages for a fixed duration."""
        import time

        request_id = f"rosclaw_sub_{uuid.uuid4().hex[:12]}"
        subscribe_msg: dict[str, Any] = {
            "op": "subscribe",
            "id": request_id,
            "topic": topic,
            "throttle_rate": throttle_rate_ms,
        }
        if msg_type:
            subscribe_msg["type"] = msg_type

        send_result = self.send(subscribe_msg)
        if not send_result.ok:
            return send_result

        messages: list[dict[str, Any]] = []
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            result = self.receive(timeout_sec=0.2)
            if result.ok and result.data:
                data = result.data
                if data.get("op") == "publish" and data.get("topic") == topic:
                    messages.append(data.get("msg", {}))

        import contextlib

        with contextlib.suppress(Exception):
            self.send({"op": "unsubscribe", "id": request_id, "topic": topic})

        return RosTransportResult(
            ok=True,
            data={"topic": topic, "messages": messages, "count": len(messages)},
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        with self._lock:
            self._ws = None
