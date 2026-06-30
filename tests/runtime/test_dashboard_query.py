"""Tests for dashboard RealSense endpoints using RuntimeQueryAPI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rosclaw.core.event_bus import EventBus
from rosclaw.dashboard.web_server import DashboardWebServer
from rosclaw.runtime import RuntimeBus, RuntimeEvent


@pytest.fixture
def runtime_bus() -> RuntimeBus:
    return RuntimeBus(event_bus=EventBus(normalize_topics=False))


@pytest.fixture
def dashboard(runtime_bus: RuntimeBus) -> DashboardWebServer:
    return DashboardWebServer(runtime_bus=runtime_bus)


@pytest.fixture
def client(dashboard: DashboardWebServer) -> TestClient:
    return TestClient(dashboard.app)


def _publish_rgbd_frame(
    bus: RuntimeBus,
    episode_id: str = "ep_live_01",
    rgb_ref: str = "frames/rgb/frame_0001.png",
    depth_ref: str | None = "frames/depth/frame_0001.png",
) -> RuntimeEvent:
    event = RuntimeEvent(
        type="camera.rgbd_frame",
        source="realsense_camera",
        robot="realsense-d405",
        payload={
            "camera_id": "d405",
            "rgb_ref": rgb_ref,
            "depth_ref": depth_ref,
            "rgb_encoding": "png",
            "depth_encoding": "png16",
            "width": 1280,
            "height": 720,
        },
        metadata={"trace_id": episode_id, "episode_id": episode_id},
    )
    bus.publish(event)
    return event


def test_realsense_latest_frame_queries_runtime(
    client: TestClient, runtime_bus: RuntimeBus
) -> None:
    _publish_rgbd_frame(runtime_bus, episode_id="ep_latest")

    response = client.get("/api/realsense/latest-frame")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["practice_id"] == "ep_latest"
    assert data["rgb_ref"] == "frames/rgb/frame_0001.png"
    assert data["frame_url"] == "/api/artifacts/ep_latest/frames/rgb/frame_0001.png"


def test_realsense_streams_queries_runtime(
    client: TestClient, runtime_bus: RuntimeBus
) -> None:
    _publish_rgbd_frame(
        runtime_bus,
        episode_id="ep_streams",
        rgb_ref="color.png",
        depth_ref="depth.png",
    )

    response = client.get("/api/realsense/streams")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["practice_id"] == "ep_streams"
    streams = {s["name"]: s for s in data["streams"]}
    assert "color" in streams
    assert streams["color"]["type"] == "rgb"
    assert streams["color"]["ref"] == "color.png"
    assert "depth" in streams
    assert streams["depth"]["type"] == "depth"


def test_realsense_frames_queries_runtime(
    client: TestClient, runtime_bus: RuntimeBus
) -> None:
    for i in range(3):
        _publish_rgbd_frame(
            runtime_bus,
            episode_id="ep_frames",
            rgb_ref=f"frames/rgb/frame_{i:04d}.png",
            depth_ref=f"frames/depth/frame_{i:04d}.png",
        )

    response = client.get("/api/realsense/frames?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["count"] == 2
    assert len(data["frames"]) == 2
    assert data["frames"][0]["rgb_ref"] == "frames/rgb/frame_0002.png"
    assert data["frames"][1]["rgb_ref"] == "frames/rgb/frame_0001.png"


def test_realsense_status_includes_runtime_latest(
    client: TestClient, runtime_bus: RuntimeBus
) -> None:
    _publish_rgbd_frame(runtime_bus, episode_id="ep_status")

    response = client.get("/api/realsense/status")
    assert response.status_code == 200
    data = response.json()
    assert "profile_exists" in data
    assert "body_linked" in data
    assert data["latest_frame"]["found"] is True
    assert data["latest_frame"]["practice_id"] == "ep_status"


def test_realsense_latest_frame_falls_back_to_disk(tmp_path) -> None:
    """When no runtime bus is supplied, endpoints fall back to disk scans."""
    dashboard = DashboardWebServer()
    client = TestClient(dashboard.app)

    response = client.get(f"/api/realsense/latest-frame?data_root={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False
    assert "command" in data
