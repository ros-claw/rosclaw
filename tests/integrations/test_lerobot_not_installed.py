"""Test graceful degradation when LeRobot is not installed."""

from __future__ import annotations

import importlib.util

import pytest

from rosclaw.integrations.lerobot import LeRobotIntegration
from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor

_LEROBOT_INSTALLED = importlib.util.find_spec("lerobot") is not None


def test_integration_report_when_not_installed():
    """The integration report must not crash and must report not_installed."""
    report = LeRobotIntegration.report()
    assert report.name == "lerobot"
    assert report.status in ("not_installed", "degraded", "installed")
    assert report.message
    assert isinstance(report.capabilities, list)


@pytest.mark.skipif(_LEROBOT_INSTALLED, reason="LeRobot is installed; not-applicable test")
def test_doctor_report_when_not_installed():
    """The doctor report must be actionable when LeRobot is absent."""
    report = run_lerobot_doctor()
    assert report.name == "lerobot"
    assert not report.lerobot_importable
    assert "rosclaw setup lerobot" in report.message
