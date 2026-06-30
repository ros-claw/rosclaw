"""Tests for Dashboard RealSense page and API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rosclaw.dashboard import web_server
from rosclaw.dashboard.metrics import DashboardMetrics


@pytest.fixture
def client() -> TestClient:
    with TestClient(web_server.app) as c:
        yield c


@pytest.fixture
def metrics() -> DashboardMetrics:
    fresh = DashboardMetrics()
    web_server._server.metrics = fresh
    web_server._server.server.metrics = fresh
    return fresh


def test_realsense_page_exists(client: TestClient) -> None:
    res = client.get("/realsense")
    assert res.status_code == 200
    text = res.text
    assert "ROSClaw RealSense Dashboard" in text
    assert "dual_lab_01" in text
    assert "D405" in text
    assert "D435i" in text


def test_realsense_streams_initial(client: TestClient, metrics: DashboardMetrics) -> None:
    res = client.get("/api/realsense/streams")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data
    assert data["cameras"]["d405"]["online"] is False
    assert data["cameras"]["d435i"]["online"] is False
    assert data["dual_online"] is False


def test_realsense_streams_updated(client: TestClient, metrics: DashboardMetrics) -> None:
    metrics.set_realsense_online("d405", True, {"serial": "12345"})
    metrics.record_realsense_frame("d405", "color", "/tmp/d405_001.jpg", latency_ms=33.0)
    metrics.set_realsense_online("d435i", True)

    res = client.get("/api/realsense/streams")
    assert res.status_code == 200
    data = res.json()
    assert data["cameras"]["d405"]["online"] is True
    assert data["cameras"]["d405"]["last_frame_path"] == "/tmp/d405_001.jpg"
    assert data["cameras"]["d405"]["last_latency_ms"] == 33.0
    assert data["cameras"]["d435i"]["online"] is True
    assert data["dual_online"] is True


def test_realsense_frames_endpoint(client: TestClient, metrics: DashboardMetrics) -> None:
    metrics.record_realsense_frame("d435i", "depth", "/tmp/d435i_depth_001.png")
    res = client.get("/api/realsense/frames?camera=d435i")
    assert res.status_code == 200
    data = res.json()
    assert data["camera"] == "d435i"
    assert data["frame_type"] == "depth"
    assert data["path"] == "/tmp/d435i_depth_001.png"


def test_realsense_frames_unknown_camera(client: TestClient) -> None:
    res = client.get("/api/realsense/frames?camera=d415")
    assert res.status_code == 404


def test_realsense_frames_no_frame(client: TestClient, metrics: DashboardMetrics) -> None:
    metrics.set_realsense_online("d405", True)
    res = client.get("/api/realsense/frames?camera=d405")
    assert res.status_code == 404


def test_snapshot_includes_realsense(client: TestClient, metrics: DashboardMetrics) -> None:
    metrics.set_realsense_online("d405", True)
    res = client.get("/snapshot")
    assert res.status_code == 200
    data = res.json()
    assert "realsense" in data
    assert data["realsense"]["cameras"]["d405"]["online"] is True


def test_websocket_snapshot_contains_realsense(client: TestClient, metrics: DashboardMetrics) -> None:
    metrics.set_realsense_online("d435i", True)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert "realsense" in msg["data"]
        assert msg["data"]["realsense"]["cameras"]["d435i"]["online"] is True
