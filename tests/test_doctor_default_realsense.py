"""Tests for default ``rosclaw doctor`` RealSense reporting."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from rosclaw.cli import _run_doctor


class _FakeRobotRegistry:
    def get(self, robot_id: str):
        if robot_id == "realsense_d405":
            return object()
        return None


class _FakeInstalledRegistry:
    def list(self):
        return [
            SimpleNamespace(name="librealsense-mcp", status="healthy"),
            SimpleNamespace(name="realsense-ros-mcp", status="healthy"),
        ]


def _doctor_args(**kwargs):
    defaults = {
        "ros2": False,
        "ros": False,
        "realsense": False,
        "bootstrap": False,
        "full": False,
        "fix": False,
        "json": False,
        "gpu": False,
        "network": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def realsense_stack(monkeypatch):
    """Mock a healthy RealSense stack so default doctor reports it."""

    def _which(name: str):
        if name in ("rs-enumerate-devices", "ros2"):
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr(shutil, "which", _which)

    def _run(cmd, **kwargs):
        result = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0].endswith("ros2"):
            result.stdout = "realsense2_camera\nstd_msgs\n"
        elif cmd[0].endswith("rs-enumerate-devices"):
            result.stdout = "Device info\nUSB3\n"
        return result

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr("rosclaw.mcp.onboarding.installed.InstalledRegistry", _FakeInstalledRegistry)
    monkeypatch.setattr("rosclaw.runtime.RobotRegistry", _FakeRobotRegistry)


class TestDoctorDefaultRealSense:
    """Default doctor must include RealSense SDK, ROS2, and USB checks."""

    def test_default_doctor_includes_realsense_checks(self, realsense_stack, capsys):
        result = _run_doctor(_doctor_args())
        captured = capsys.readouterr().out

        assert result == 0
        for name in (
            "rs-enumerate-devices",
            "pyrealsense2",
            "ros2 CLI",
            "rclpy",
            "realsense2_camera package",
            "RealSense MCPs",
            "realsense_d405 profile",
            "Cosmos endpoint",
        ):
            assert name in captured, f"missing check: {name}"

    def test_default_doctor_reports_usb_speed(self, realsense_stack, capsys):
        result = _run_doctor(_doctor_args())
        captured = capsys.readouterr().out

        assert result == 0
        assert "USB3" in captured

    def test_default_doctor_surfaces_recommendations_when_stack_missing(self, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "which", lambda _name: None)

        result = _run_doctor(_doctor_args())
        captured = capsys.readouterr().out

        assert result == 0
        assert "RealSense recommendations" in captured
        assert "rs-enumerate-devices" in captured
