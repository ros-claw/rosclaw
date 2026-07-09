"""Test LeRobot doctor behavior."""

from __future__ import annotations

from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor


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
