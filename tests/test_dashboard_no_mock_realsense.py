"""Tests that the RealSense dashboard shows real state, not mock data."""

from __future__ import annotations

import pytest

from rosclaw.dashboard.web_server import DashboardWebServer


@pytest.fixture
def webserver():
    return DashboardWebServer(host="127.0.0.1", port=0)


class TestDashboardNoMockRealsense:
    def test_realsense_status_reports_profile_presence(self, webserver, tmp_path):
        from fastapi.testclient import TestClient

        client = TestClient(webserver.app)
        resp = client.get(f"/api/realsense/status?data_root={tmp_path}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["profile_exists"] is True
        assert "body_linked" in data
        assert "latest_frame" in data

    def test_realsense_latest_frame_empty_state(self, webserver, tmp_path):
        from fastapi.testclient import TestClient

        client = TestClient(webserver.app)
        resp = client.get(f"/api/realsense/latest-frame?data_root={tmp_path}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert "command" in data
        assert "rosclaw practice run" in data["command"]

    def test_realsense_page_html(self, webserver):
        from fastapi.testclient import TestClient

        client = TestClient(webserver.app)
        resp = client.get("/realsense")
        assert resp.status_code == 200
        text = resp.text
        assert "ROSClaw RealSense" in text
        assert "/api/realsense/status" in text
        assert "/api/realsense/latest-frame" in text
