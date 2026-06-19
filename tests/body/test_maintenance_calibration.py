"""Tests for rosclaw body maintenance, calibration update, and retrofit commands."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rosclaw.body.resolver import BodyResolver
from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_maintenance_add_appends_event(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "maintenance", "add",
        "--type", "inspection",
        "--component", "right_arm",
        "--summary", "Routine joint inspection completed",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    events = resolver.get_maintenance_events()
    inspection = next(e for e in events if e.type == "inspection")
    assert inspection.component == "right_arm"
    assert "Routine joint inspection" in inspection.message


def test_calibration_update_replaces_file_and_logs_event(linked_body):
    resolver = BodyResolver()
    new_cal_path = Path(linked_body) / "new_cal.yaml"
    new_cal = {
        "schema": "rosclaw.calibration.v1",
        "body_instance_id": resolver.get_current_body_yaml().body_instance.get("id"),
        "validation": {"status": "validated", "last_validated_at": "2026-06-19T00:00:00+00:00"},
        "validity": {"overall_status": "validated"},
    }
    new_cal_path.write_text(yaml.safe_dump(new_cal), encoding="utf-8")

    with patch.object(sys, "argv", [
        "rosclaw", "body", "calibration", "update",
        "--file", str(new_cal_path),
    ]):
        assert rosclaw_main() == 0

    calibration = resolver.get_calibration()
    assert calibration.overall_status() == "validated"
    cal_events = [e for e in resolver.get_maintenance_events() if e.type == "calibration"]
    assert cal_events


def test_retrofit_add_records_event_and_updates_components(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "retrofit", "add",
        "--component", "wrist_camera",
        "--type", "sensor_install",
        "--summary", "Installed wrist camera for manipulation",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    events = [e for e in resolver.get_maintenance_events() if "retrofit" in e.tags]
    assert events
    assert events[0].component == "wrist_camera"

    body_yaml = resolver.get_current_body_yaml()
    assert body_yaml.installed_components.get("sensors", {}).get("wrist_camera", {}).get("installed")
