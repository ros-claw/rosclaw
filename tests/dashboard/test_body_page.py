"""Tests for dashboard body page and API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rosclaw.body.registry import BodyRegistryManager
from rosclaw.dashboard.web_server import DashboardWebServer


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def client() -> TestClient:
    server = DashboardWebServer()
    return TestClient(server.app)


def test_api_body_returns_summary(client: TestClient) -> None:
    response = client.get("/api/body")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "bodies" in data
    assert "sense" in data


def test_body_page_returns_html(client: TestClient) -> None:
    response = client.get("/body")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ROSClaw Body" in response.text


def test_websocket_initial_snapshot_includes_body(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert "body" in msg["data"]


def test_api_body_with_registered_bodies(client: TestClient) -> None:
    ws = Path.home() / ".rosclaw"
    manager = BodyRegistryManager(ws)
    manager.create_body("g1-sim", "unitree-g1")
    manager.create_body("g1-real", "unitree-g1")

    response = client.get("/api/body")
    assert response.status_code == 200
    data = response.json()
    assert data["current"] in {"g1-sim", "g1-real"}
    assert len(data["bodies"]) == 2
    assert {b["body_id"] for b in data["bodies"]} == {"g1-sim", "g1-real"}
