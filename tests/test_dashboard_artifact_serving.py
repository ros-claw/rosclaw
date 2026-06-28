"""Tests for safe artifact serving from the dashboard."""

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
        task=None,
        skill="realsense_capture_rgbd",
        provider=None,
        capability="vlm.risk_assessment",
        output_root=str(output_root),
        data_root=None,
        workspace=str(linked_realsense_workspace),
        json=False,
    )
    assert cmd_practice_run(args) == 0
    sessions_dir = output_root / "sessions"
    session_dir = next(sessions_dir.iterdir())
    return str(output_root), session_dir.name


class TestDashboardArtifactServing:
    def test_serve_color_frame(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(
            f"/api/artifacts/{practice_id}/artifacts/skill/color.png?data_root={data_root}"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] in ("image/png", "application/octet-stream")
        assert resp.content.startswith(b"\x89PNG")

    def test_artifact_missing_returns_404(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(
            f"/api/artifacts/{practice_id}/artifacts/skill/missing.png?data_root={data_root}"
        )
        assert resp.status_code == 404

    def test_artifact_path_traversal_blocked(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, practice_id = sample_episode
        client = TestClient(webserver.app)
        # Use percent-encoded slashes so the test client does not collapse "..".
        resp = client.get(
            f"/api/artifacts/{practice_id}/..%2f..%2fetc/passwd?data_root={data_root}"
        )
        assert resp.status_code == 403

    def test_artifact_absolute_path_blocked(self, webserver, sample_episode):
        from fastapi.testclient import TestClient

        data_root, _ = sample_episode
        client = TestClient(webserver.app)
        resp = client.get(f"/api/artifacts//etc/passwd?data_root={data_root}")
        assert resp.status_code == 403
