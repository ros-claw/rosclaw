"""Tests for rosclaw body diff."""

import sys
from unittest.mock import patch

import pytest

from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_diff_detects_sensor_change(linked_body, capsys):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test camera",
    ]):
        assert rosclaw_main() == 0

    with patch.object(sys, "argv", ["rosclaw", "body", "diff", "--format", "json"]):
        rc = rosclaw_main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "sensor_status" in out or "installed_components.sensors.head_camera.status" in out
    assert "requires_skill_recheck" in out


def test_diff_against_eurdf_default(linked_body, capsys):
    with patch.object(sys, "argv", ["rosclaw", "body", "diff"]):
        rc = rosclaw_main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "ROSClaw Body Diff" in out
