"""Tests for rosbridge transport."""

from __future__ import annotations

import pytest

from rosclaw.connectors.ros.transport import (
    MockTransport,
    RosbridgeEndpoint,
    RosbridgeTransport,
    RosTransportResult,
)


def test_endpoint_url():
    ep = RosbridgeEndpoint(host="localhost", port=9090)
    assert ep.url == "ws://localhost:9090"


def test_endpoint_url_custom_scheme():
    ep = RosbridgeEndpoint(host="example.com", port=443, scheme="wss")
    assert ep.url == "wss://example.com:443"


def test_transport_result_is_ok():
    ok = RosTransportResult(ok=True, data={})
    assert ok.is_ok
    fail = RosTransportResult(ok=False, error="boom")
    assert not fail.is_ok


def test_mock_transport_request_uses_queued_response():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"x": 1}}, request_id="r1")
    result = transport.request({"op": "call_service", "service": "/foo", "id": "r1"})
    assert result.ok
    assert result.data["values"]["x"] == 1
    assert result.request_id == "r1"
    assert len(transport.requests) == 1


def test_mock_transport_returns_error_without_response():
    transport = MockTransport()
    result = transport.request({"op": "call_service", "service": "/foo"})
    assert not result.ok
    assert "no mock response queued" in result.error


def test_mock_transport_dry_run():
    transport = MockTransport(dry_run=True)
    result = transport.request({"op": "publish", "topic": "/cmd_vel"})
    assert result.ok
    assert result.data.get("dry_run")


def test_rosbridge_transport_dry_run():
    transport = RosbridgeTransport(dry_run=True)
    result = transport.request({"op": "call_service", "service": "/foo"})
    assert result.ok
    assert result.data.get("dry_run")


def test_rosbridge_transport_request_injects_id():
    transport = RosbridgeTransport(dry_run=True)
    result = transport.request({"op": "call_service", "service": "/foo"})
    assert result.request_id is not None
    assert result.request_id.startswith("rosclaw_")


def test_rosbridge_transport_close_is_idempotent():
    transport = RosbridgeTransport(dry_run=True)
    transport.close()
    transport.close()  # should not raise


def test_rosbridge_loopback_connection_bypasses_environment_proxy(monkeypatch):
    import socket

    import websocket

    captured: dict[str, object] = {}

    class FakeWebSocket:
        connected = True

    class FakeSocket:
        pass

    def fake_create_connection(url, **options):
        captured["url"] = url
        captured.update(options)
        return FakeWebSocket()

    direct_socket = FakeSocket()
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: direct_socket)
    monkeypatch.setattr(websocket, "create_connection", fake_create_connection)
    transport = RosbridgeTransport(
        endpoint=RosbridgeEndpoint(host="127.0.0.1", port=9090),
        max_retries=0,
    )

    result = transport.connect()

    assert result.ok
    assert captured["url"] == "ws://127.0.0.1:9090"
    assert captured["socket"] is direct_socket


def test_rosbridge_closes_direct_socket_when_handshake_fails(monkeypatch):
    import socket

    import websocket

    class FakeSocket:
        closed = False

        def close(self):
            self.closed = True

    direct_socket = FakeSocket()

    def fail_handshake(*args, **kwargs):
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: direct_socket)
    monkeypatch.setattr(websocket, "create_connection", fail_handshake)
    transport = RosbridgeTransport(
        endpoint=RosbridgeEndpoint(host="127.0.0.1", port=9090),
        max_retries=0,
    )

    result = transport.connect()

    assert result.ok is False
    assert direct_socket.closed is True


def test_rosbridge_transport_send_invalid_json():
    transport = RosbridgeTransport(dry_run=False)
    # No websocket server; connect will fail, returning structured error.
    result = transport.send({"op": "publish", "topic": "/x", "msg": object()})
    assert not result.ok


def test_mock_transport_publishes_are_recorded():
    transport = MockTransport()
    result = transport.publish("/cmd_vel", {"linear": {"x": 0.1}})
    assert result.ok
    assert transport.requests[-1]["topic"] == "/cmd_vel"


def test_mock_transport_subscribe_for_duration():
    transport = MockTransport()
    result = transport.subscribe_for_duration("/joint_states", duration_sec=0.05)
    assert result.ok
    assert result.data["count"] == 0


@pytest.mark.slow
def test_rosbridge_real_connection_fails_without_server():
    transport = RosbridgeTransport(
        endpoint=RosbridgeEndpoint(host="127.0.0.1", port=9999, timeout_sec=0.5),
        max_retries=0,
    )
    result = transport.connect()
    assert not result.ok
    assert "Failed to connect" in result.error or "websocket-client" in result.error
