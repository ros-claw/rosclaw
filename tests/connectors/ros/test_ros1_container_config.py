"""Regression tests for the ROS1 rosbridge published-port configuration."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_ros1_external_port_is_forwarded_to_rosbridge() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.ros1-test.yml").read_text(encoding="utf-8"))
    service = compose["services"]["ros-bridge"]

    assert service["environment"]["ROSBRIDGE_EXTERNAL_PORT"] == "${ROSBRIDGE_HOST_PORT:-9091}"

    launch = (ROOT / "docker/ros1-noetic-launch.launch").read_text(encoding="utf-8")
    assert (
        '<param name="websocket_external_port" value="$(env ROSBRIDGE_EXTERNAL_PORT)" />' in launch
    )
