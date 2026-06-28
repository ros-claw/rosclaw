"""Tests for ``rosclaw doctor --realsense``."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from rosclaw.cli import _run_doctor_realsense


class _FakeRobotRegistry:
    def get(self, robot_id: str):
        if robot_id == "realsense_d405":
            return object()
        return None


class _FakeInstalledRegistry:
    def list(self):
        from types import SimpleNamespace

        return [
            SimpleNamespace(name="librealsense-mcp", status="healthy"),
            SimpleNamespace(name="realsense-ros-mcp", status="healthy"),
        ]


class TestDoctorRealSenseChecks:
    """Phase I RealSense doctor tests."""

    def _args(self, **kwargs):
        defaults = {"json": False}
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_doctor_reports_issues_when_stack_missing(self, monkeypatch, capsys):
        """Without RealSense installed, doctor reports actionable issues."""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: None)

        result = _run_doctor_realsense(self._args())
        captured = capsys.readouterr().out

        assert result == 1
        assert "rs-enumerate-devices" in captured
        assert "Issues found" in captured

    def test_doctor_reports_healthy_when_mocked(self, monkeypatch, capsys):
        """With all dependencies mocked, doctor returns healthy."""
        import importlib
        import shutil
        import urllib.request

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
        monkeypatch.setattr(importlib, "import_module", lambda name: object() if name == "rclpy" else __import__(name))

        monkeypatch.setattr(
            "rosclaw.mcp.onboarding.installed.InstalledRegistry", _FakeInstalledRegistry
        )
        monkeypatch.setattr("rosclaw.runtime.RobotRegistry", _FakeRobotRegistry)

        class _FakeResponse:
            status = 200

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda _req, **kwargs: _FakeResponse()
        )
        monkeypatch.setattr(urllib.request, "Request", lambda url, method="GET": object())

        result = _run_doctor_realsense(self._args())
        captured = capsys.readouterr().out

        assert result == 0
        assert "RealSense D405" in captured
        assert "✅" in captured

    def test_doctor_realsense_json_output(self, monkeypatch):
        """JSON output contains structured check results."""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: None)

        result = _run_doctor_realsense(self._args(json=True))
        # JSON path still returns 0/1 based on health; missing stack is unhealthy.
        assert result == 1

    def test_doctor_detects_usb2_degraded_mode(self, monkeypatch, capsys):
        """A USB2 RealSense connection surfaces a warning."""
        import shutil
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rs-enumerate-devices")

        def _run(cmd, **kwargs):
            result = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            result.stdout = "Device info\nUSB2\n"
            return result

        monkeypatch.setattr(subprocess, "run", _run)

        _run_doctor_realsense(self._args())
        captured = capsys.readouterr().out

        assert "USB2" in captured
        assert "degraded" in captured.lower()
