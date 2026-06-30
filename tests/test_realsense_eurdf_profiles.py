"""Tests for RealSense e-URDF profiles."""
from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.runtime import EURDFLoader, RobotRegistry


ZOO_PATH = Path(__file__).parent.parent / "e-urdf-zoo"


@pytest.fixture
def loader() -> EURDFLoader:
    return EURDFLoader(ZOO_PATH)


@pytest.mark.parametrize("robot_id", ["realsense-d405", "realsense-d435i", "realsense-dual"])
def test_realsense_profile_loads(loader: EURDFLoader, robot_id: str) -> None:
    profile = loader.load(robot_id)
    assert profile.robot_id == robot_id.replace("-", "_")
    assert profile.embodiment.dof == 0
    assert profile.embodiment.actuators == []
    assert profile.embodiment.metadata.get("no_actuation") is True
    assert "perception_only" in profile.semantic.semantic_tags
    assert profile.safety.safety_level == "STRICT"


def test_realsense_d405_sensors(loader: EURDFLoader) -> None:
    profile = loader.load("realsense-d405")
    sensor_names = {s["name"] for s in profile.embodiment.sensors}
    assert "color_camera" in sensor_names
    assert "depth_camera" in sensor_names
    assert "imu" not in sensor_names


def test_realsense_d435i_sensors(loader: EURDFLoader) -> None:
    profile = loader.load("realsense-d435i")
    sensor_names = {s["name"] for s in profile.embodiment.sensors}
    assert "color_camera" in sensor_names
    assert "depth_camera" in sensor_names
    assert "imu" in sensor_names


def test_realsense_dual_sensors(loader: EURDFLoader) -> None:
    profile = loader.load("realsense-dual")
    sensor_names = {s["name"] for s in profile.embodiment.sensors}
    assert "d405_color_camera" in sensor_names
    assert "d435i_imu" in sensor_names
    assert profile.embodiment.metadata.get("composition") == {
        "close_range_eye": "realsense_d405",
        "world_eye_imu": "realsense_d435i",
    }


def test_realsense_d405_depth_range(loader: EURDFLoader) -> None:
    profile = loader.load("realsense-d405")
    depth = next(s for s in profile.embodiment.sensors if s["name"] == "depth_camera")
    assert depth["min_range_m"] == pytest.approx(0.07)
    assert depth["max_range_m"] == pytest.approx(0.50)


def test_realsense_d435i_depth_range(loader: EURDFLoader) -> None:
    profile = loader.load("realsense-d435i")
    depth = next(s for s in profile.embodiment.sensors if s["name"] == "depth_camera")
    assert depth["min_range_m"] == pytest.approx(0.30)
    assert depth["max_range_m"] == pytest.approx(3.00)


def test_realsense_registry_list_available() -> None:
    registry = RobotRegistry(EURDFLoader(ZOO_PATH))
    available = registry.list_available()
    assert "realsense-d405" in available
    assert "realsense-d435i" in available
    assert "realsense-dual" in available


def test_realsense_validate_passes(loader: EURDFLoader) -> None:
    for robot_id in ["realsense-d405", "realsense-d435i", "realsense-dual"]:
        result = loader.validate(robot_id)
        assert result["valid"], f"{robot_id} validation failed: {result['errors']}"
