"""Test LeRobot doctor behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor
from rosclaw.integrations.lerobot.smoke_report import (
    SmokeReport,
    get_validation_status,
    write_smoke_report,
)

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


def test_doctor_includes_validation_status(tmp_path: Path, monkeypatch):
    """The doctor report should include the current validation state."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    report = run_lerobot_doctor()
    assert "state" in report.validation_status
    assert report.validation_status["state"] == "not_configured"
    assert report.validation_status["last_policy"] is None


def test_doctor_validation_status_reflects_successful_smoke(
    tmp_path: Path, monkeypatch
):
    """A successful smoke report makes the doctor validation state 'validated'."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    smoke = SmokeReport(
        status="ok",
        policy={"repo_id": "lerobot/act_test", "policy_type": "act"},
        action_proposal={
            "shape": [100, 14],
            "not_executed": True,
            "requires_sandbox": True,
            "body_mapping_required": True,
        },
        runtime={
            "lerobot_version": "0.6.1",
            "python_executable": "/fake/python",
            "device": "cpu",
        },
    )
    write_smoke_report(smoke)

    report = run_lerobot_doctor()
    assert report.validation_status["state"] == "validated"
    assert report.validation_status["policy_type"] == "act"
    assert report.validation_status["action_shape"] == [100, 14]
    assert "proposal_only" in report.validation_status["safety"]


def test_doctor_validation_status_detects_stale_report(
    tmp_path: Path, monkeypatch
):
    """A smoke report with mismatched runtime version is stale."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    smoke = SmokeReport(
        status="ok",
        policy={"repo_id": "lerobot/act_test", "policy_type": "act"},
        runtime={
            "lerobot_version": "0.5.0",
            "python_executable": "/fake/python",
            "device": "cpu",
        },
    )
    write_smoke_report(smoke)

    validation = get_validation_status(
        report=smoke,
        current_lerobot_version="0.6.1",
        current_python_executable="/fake/python",
    )
    assert validation["state"] == "stale"
    assert any("LeRobot 0.5.0" in reason for reason in validation["stale_reasons"])
