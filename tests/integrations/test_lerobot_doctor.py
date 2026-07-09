"""Test LeRobot doctor behavior."""

from __future__ import annotations

import importlib.util

import pytest

from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor

_LEROBOT_INSTALLED = importlib.util.find_spec("lerobot") is not None


@pytest.mark.skipif(_LEROBOT_INSTALLED, reason="LeRobot is installed; fake-info test not applicable")
def test_doctor_finds_fake_lerobot_info(fake_lerobot_info):
    """The doctor should detect a fake lerobot-info binary."""
    report = run_lerobot_doctor()
    assert report.lerobot_info_path is not None
    assert report.lerobot_info_ok is True
    assert "LeRobot info stub" in report.lerobot_info_output


def test_doctor_reports_python_and_hf_env():
    """The doctor should report Python version and HF endpoint."""
    report = run_lerobot_doctor()
    assert report.python_version
    assert report.python_executable
    assert report.hf_endpoint


def test_doctor_reports_rosclaw_and_lerobot_runtime():
    """The doctor should expose both ROSClaw and LeRobot runtime fields."""
    report = run_lerobot_doctor()
    assert report.rosclaw_python_executable
    assert report.rosclaw_python_version
    assert isinstance(report.worker_in_process_available, bool)
    assert isinstance(report.worker_subprocess_available, bool)
