"""Contract tests: dashboard body API consumes EffectiveBody via BodyResolver."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import MaintenanceEvent
from rosclaw.body.service import BodyInstanceService
from rosclaw.dashboard.web_server import DashboardWebServer


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / ".rosclaw"
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService().create_or_init(
        robot="unitree-g1", name="g1-dashboard", mode="registry", update_registry=True, switch_active=True
    )
    return workspace


@pytest.fixture
def client(linked_workspace: Path) -> TestClient:
    return TestClient(DashboardWebServer().app)


def test_dashboard_api_body_hash_matches_resolver(linked_workspace: Path, client: TestClient):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    response = client.get("/api/body")
    assert response.status_code == 200
    data = response.json()

    assert data["current"] == body.body_instance_id
    assert data["effective_body_hash"] == body.effective_body_hash
    assert data["body_instance_id"] == body.body_instance_id


def test_dashboard_api_body_readiness_and_capabilities(linked_workspace: Path, client: TestClient):
    response = client.get("/api/body")
    assert response.status_code == 200
    data = response.json()

    assert "readiness" in data
    assert "capabilities" in data
    assert "forbidden_capabilities" in data


def test_dashboard_websocket_snapshot_body_hash_matches(linked_workspace: Path, client: TestClient):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        snapshot_body = msg["data"]["body"]
        assert snapshot_body["effective_body_hash"] == body.effective_body_hash
        assert snapshot_body["body_instance_id"] == body.body_instance_id


def test_dashboard_api_body_effective_matches_resolver(linked_workspace: Path, client: TestClient):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    response = client.get("/api/body/effective")
    assert response.status_code == 200
    data = response.json()

    assert data["body_instance_id"] == body.body_instance_id
    assert data["effective_body_hash"] == body.effective_body_hash
    assert data["eurdf_uri"] == body.eurdf_uri
    assert "frames" in data
    assert "joints" in data
    assert "sensors" in data
    assert "actuators" in data


def test_dashboard_api_body_skills_matches_resolver(linked_workspace: Path, client: TestClient):
    resolver = BodyResolver()
    report = resolver.get_skill_compatibility()

    response = client.get("/api/body/skills")
    assert response.status_code == 200
    data = response.json()

    assert data["body_instance_id"] == report.body_instance_id
    assert data["effective_body_hash"] == report.effective_body_hash
    assert "skills" in data
    assert "summary" in data


def test_dashboard_api_body_history_returns_events(linked_workspace: Path, client: TestClient):
    resolver = BodyResolver()
    log = MaintenanceLog(resolver.maintenance_log_path)
    log.append(
        MaintenanceEvent(
            ts="2026-01-01T00:00:00Z",
            type="maintenance",
            severity="info",
            author="test",
            body_instance_id=resolver.body_id,
            message="dashboard test event",
            summary="dashboard test event",
            component="test",
            affects=["test"],
            tags=["test"],
            requires_skill_recheck=False,
        )
    )

    response = client.get("/api/body/history")
    assert response.status_code == 200
    data = response.json()

    assert data["body_instance_id"] == resolver.body_id
    assert data["count"] >= 1
    assert any(e.get("summary") == "dashboard test event" for e in data["events"])


def test_dashboard_api_body_provider_health(linked_workspace: Path, client: TestClient):
    response = client.get("/api/body/provider-health")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "interfaces" in data
    assert "summary" in data
    assert data["body_instance_id"] == "g1-dashboard"
