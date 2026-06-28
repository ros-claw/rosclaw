"""Verify Dashboard RealSense endpoints read from recorded practice episodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.dashboard.web_server import DashboardWebServer


try:
    from fastapi.testclient import TestClient
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"fastapi TestClient unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Dashboard client pointing at an empty practice data root."""
    server = DashboardWebServer()
    server.app.state.data_root = tmp_path
    return TestClient(server.app)


def _make_episode(data_root: Path, practice_id: str) -> Path:
    """Create a minimal practice session with one rgbd_frame event."""
    session_dir = data_root / "sessions" / practice_id
    session_dir.mkdir(parents=True)
    frames_dir = session_dir / "artifacts" / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "color_000001.png").write_bytes(b"fake png")
    (frames_dir / "depth_000001.png").write_bytes(b"fake depth png")

    frame_event = {
        "schema_version": "practice.event.v1",
        "practice_id": practice_id,
        "robot_id": "d405_lab_01",
        "source": "camera",
        "event_type": "rgbd_frame",
        "timestamp_ns": 1,
        "timestamp_utc": "2026-06-28T21:00:00Z",
        "sequence_id": 1,
        "event_id": "frame-1",
        "trace_id": practice_id,
        "payload": {
            "camera_id": "d405_lab_01",
            "width": 640,
            "height": 480,
            "rgb_encoding": "png",
            "depth_encoding": "png16",
            "rgb_ref": "artifacts/frames/color_000001.png",
            "depth_ref": "artifacts/frames/depth_000001.png",
        },
    }
    (session_dir / "timeline.jsonl").write_text(
        json.dumps(frame_event) + "\n", encoding="utf-8"
    )
    (session_dir / "episode.json").write_text(
        json.dumps(
            {
                "practice_id": practice_id,
                "robot_id": "d405_lab_01",
                "outcome": "SUCCESS",
                "event_count": 1,
                "start_time": "2026-06-28T21:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return session_dir


def test_realsense_status_empty(client, tmp_path):
    response = client.get("/api/realsense/status", params={"data_root": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["profile_exists"] is True
    assert data["latest_frame"]["found"] is False


def test_realsense_latest_frame(client, tmp_path):
    _make_episode(tmp_path, "prac_123")
    response = client.get("/api/realsense/latest-frame", params={"data_root": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["practice_id"] == "prac_123"
    assert "color_000001.png" in data["frame_url"]


def test_realsense_streams(client, tmp_path):
    _make_episode(tmp_path, "prac_123")
    response = client.get("/api/realsense/streams", params={"data_root": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    names = {s["name"] for s in data["streams"]}
    assert "color" in names
    assert "depth" in names
    assert all("url" in s for s in data["streams"])


def test_realsense_frames(client, tmp_path):
    _make_episode(tmp_path, "prac_123")
    response = client.get("/api/realsense/frames", params={"data_root": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["count"] == 1
    frame = data["frames"][0]
    assert frame["practice_id"] == "prac_123"
    assert frame["rgb_ref"].endswith("color_000001.png")
    assert frame["depth_ref"].endswith("depth_000001.png")


def test_realsense_frames_empty(client, tmp_path):
    response = client.get("/api/realsense/frames", params={"data_root": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False
    assert "command" in data


def test_artifact_serving(client, tmp_path):
    _make_episode(tmp_path, "prac_123")
    response = client.get(
        "/api/artifacts/prac_123/artifacts/frames/color_000001.png",
        params={"data_root": str(tmp_path)},
    )
    assert response.status_code == 200
    assert response.content == b"fake png"


def test_artifact_outside_session_not_found(client, tmp_path):
    _make_episode(tmp_path, "prac_123")
    (tmp_path / "secret.txt").write_text("secret")
    response = client.get(
        "/api/artifacts/secret.txt",
        params={"data_root": str(tmp_path)},
    )
    assert response.status_code == 404
