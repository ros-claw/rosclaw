"""Tests for the dashboard practice-episode API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosclaw.cli import cmd_practice_run
from rosclaw.dashboard.web_server import DashboardWebServer


@pytest.fixture
def webserver():
    return DashboardWebServer(host="127.0.0.1", port=0)


@pytest.fixture
def sample_episode(linked_realsense_workspace, fake_realsense_skill, tmp_path):
    """Create a recorded practice episode via the CLI."""
    output_root = tmp_path / "episode"
    args = SimpleNamespace(
        robot="d405_lab_01",
        robot_type=None,
        task="realsense_inspection",
        skill="realsense_capture_rgbd",
        provider="cosmos-reason2-lan",
        capability="vlm.risk_assessment",
        output_root=str(output_root),
        data_root=None,
        workspace=str(linked_realsense_workspace),
        json=False,
    )
    assert cmd_practice_run(args) == 0
    sessions_dir = output_root / "sessions"
    session_dir = next(sessions_dir.iterdir())
    return str(output_root), session_dir.name, str(session_dir)


class TestDashboardPracticeApi:
    def test_list_episodes(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert any(ep["practice_id"] == practice_id for ep in data["episodes"])
        assert "command" in data

    def test_list_episodes_empty_state(self, webserver, tmp_path):
        from fastapi.testclient import TestClient

        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes?data_root={tmp_path}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert "command" in data

    def test_get_episode(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/{practice_id}?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode"]["practice_id"] == practice_id
        assert data["episode"]["outcome"] == "SUCCESS"

    def test_get_episode_timeline(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/{practice_id}/timeline?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        sources = {ev["source"] for ev in data["timeline"]}
        assert "camera" in sources
        assert "sandbox" in sources

    def test_get_episode_artifacts(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/{practice_id}/artifacts?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert any("color.png" in art for art in data["artifacts"])

    def test_get_episode_provider(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/{practice_id}/provider?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"]["provider_id"] == "cosmos-reason2-lan"
        assert "normalized" in data["provider"]

    def test_get_episode_sandbox(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/{practice_id}/sandbox?data_root={data_root}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data["decisions"][0]["payload"]["decision"] == "ALLOW"

    def test_get_episode_not_found(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, _, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/practice/episodes/does_not_exist?data_root={data_root}")
        assert resp.status_code == 404
