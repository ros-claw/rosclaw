"""Tests for dashboard firstboot page and API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rosclaw.dashboard.web_server import DashboardWebServer


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def client() -> TestClient:
    server = DashboardWebServer()
    return TestClient(server.app)


def test_api_firstboot_returns_state(client: TestClient) -> None:
    response = client.get("/api/firstboot")
    assert response.status_code == 200
    data = response.json()
    assert "home" in data
    assert data["initialized"] is False
    assert data["install_state"] is None
    assert "steps" in data
    assert data["steps"]["bootstrap"] is False


def test_firstboot_page_returns_html(client: TestClient) -> None:
    response = client.get("/firstboot")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    text = response.text
    assert "ROSClaw First Boot" in text
    assert "rosclaw firstboot" in text


def test_firstboot_preview_default_command(client: TestClient) -> None:
    response = client.post("/api/firstboot/preview", json={})
    assert response.status_code == 200
    data = response.json()
    command = data["command"]
    assert command.startswith("rosclaw firstboot --yes")
    assert "--profile offline" in command
    assert "--robot sim_ur5e" in command
    assert "--safety strict" in command
    assert "--no-telemetry" in command
    assert "--enable-mcp" in command
    assert "--enable-sandbox" in command

    preview = data["preview"]
    assert preview["profile"] == "offline"
    assert preview["robot"] == "sim_ur5e"
    assert preview["telemetry"] is False


def test_firstboot_preview_custom_command(client: TestClient) -> None:
    choices = {
        "profile": "cloud",
        "robot": "unitree_g1",
        "safety": "moderate",
        "telemetry": True,
        "mcp": False,
        "sandbox": False,
        "ros2": True,
        "memory": True,
    }
    response = client.post("/api/firstboot/preview", json=choices)
    assert response.status_code == 200
    data = response.json()
    command = data["command"]
    assert "--profile cloud" in command
    assert "--robot unitree_g1" in command
    assert "--safety moderate" in command
    assert "--telemetry" in command
    assert "--disable-mcp" in command
    assert "--disable-sandbox" in command
    assert "--enable-ros2" in command
    assert "--enable-memory" in command
    assert "--no-telemetry" not in command

    preview = data["preview"]
    assert preview["telemetry"] is True
    assert preview["mcp"] is False
    assert preview["use_cases"]["ros2"] is True


def test_firstboot_preview_rejects_non_object(client: TestClient) -> None:
    response = client.post("/api/firstboot/preview", content=b"\"not-an-object\"")
    assert response.status_code == 400


def test_api_firstboot_with_initialized_workspace(client: TestClient) -> None:
    from rosclaw.firstboot.workspace import ensure_minimal_workspace

    home = Path.home() / ".rosclaw"
    ensure_minimal_workspace(home)

    response = client.get("/api/firstboot")
    assert response.status_code == 200
    data = response.json()
    assert data["initialized"] is True
    assert data["workspace_exists"] is True
    assert data["steps"]["bootstrap"] is True
