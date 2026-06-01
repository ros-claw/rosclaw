"""Tests for DashboardWebServer HTTP API and WebSocket endpoints."""

import pytest

from rosclaw.dashboard.web_server import DashboardWebServer, WebSocketClient
from rosclaw.core.event_bus import EventBus, Event


@pytest.fixture
def webserver():
    ws = DashboardWebServer(host="127.0.0.1", port=0)
    return ws


class TestWebSocketClient:
    def test_init(self):
        mock_ws = object()
        client = WebSocketClient(mock_ws)
        assert client._ws is mock_ws


class TestDashboardWebServerInit:
    def test_init(self, webserver):
        assert webserver.server is not None
        assert webserver.metrics is not None
        assert webserver.app is not None
        assert webserver.app.title == "ROSClaw Dashboard"

    def test_routes_exist(self, webserver):
        routes = [r.path for r in webserver.app.routes]
        assert "/health" in routes
        assert "/snapshot" in routes
        assert "/events/counts" in routes
        assert "/metrics/provider" in routes
        assert "/metrics/sandbox" in routes
        assert "/metrics/episode" in routes
        assert "/ws" in routes


class TestDashboardWebServerHttpApi:
    def test_health_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_snapshot_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        # Snapshot returns metrics data directly
        assert "provider" in data or "sandbox" in data or "episodes" in data or "module_health" in data

    def test_event_counts_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/events/counts")
        assert resp.status_code == 200

    def test_provider_metrics_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/metrics/provider")
        assert resp.status_code == 200

    def test_sandbox_metrics_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/metrics/sandbox")
        assert resp.status_code == 200

    def test_episode_metrics_endpoint(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.get("/metrics/episode")
        assert resp.status_code == 200

    def test_record_provider_post(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.post("/metrics/provider?provider=llm&capability=chat&latency_ms=150.0&status=ok")
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_record_sandbox_post(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.post("/metrics/sandbox?action_type=move&is_safe=true")
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_record_episode_post(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.post("/metrics/episode?episode_id=ep1&robot_id=ur5e&status=success&reward=0.95&duration_sec=1.5")
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_record_event_post(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.post("/event/rosclaw.test.topic")
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_set_module_health_post(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        resp = client.post("/health/runtime?status=HEALTHY")
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"


class TestDashboardWebServerEventBus:
    def test_attach_to_event_bus(self, webserver):
        bus = EventBus()
        webserver.attach_to_event_bus(bus)
        # Publish a praxis.completed event
        bus.publish(Event(
            topic="praxis.completed",
            payload={"episode_id": "ep_test", "robot_id": "ur5e", "success": True, "duration_sec": 2.0},
            source="test",
        ))
        stats = webserver.metrics.get_episode_stats()
        assert stats["total"] >= 1

    def test_attach_twice_idempotent(self, webserver):
        bus = EventBus()
        webserver.attach_to_event_bus(bus)
        # Should not raise
        webserver.attach_to_event_bus(bus)


class TestDashboardWebServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, webserver):
        await webserver.start()
        assert webserver.server._running is True
        await webserver.stop()
        assert webserver.server._running is False


class TestDashboardWebServerWebsocket:
    def test_websocket_ping_pong(self, webserver):
        from fastapi.testclient import TestClient
        client = TestClient(webserver.app)
        with client.websocket_connect("/ws") as ws:
            # Receive initial snapshot
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            # Send ping
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"
