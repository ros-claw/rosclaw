"""Tests for rosclaw body capability disable/degrade/enable lifecycle."""

import sys
from unittest.mock import patch

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def _capability_events(resolver: BodyResolver):
    return [e for e in resolver.get_maintenance_events() if e.type == "capability_update"]


def test_disable_capability_updates_body_yaml_and_logs_event(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "capability", "disable",
        "walk",
        "--reason", "Knee actuator overheating",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    body_yaml = resolver.get_current_body_yaml()
    assert "walk" in body_yaml.capabilities.get("disabled", [])
    assert "walk" not in body_yaml.capabilities.get("enabled", [])

    events = _capability_events(resolver)
    assert any(e.component == "walk" and e.result.get("action") == "disable" for e in events)


def test_degrade_capability_updates_body_yaml_and_logs_event(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "capability", "degrade",
        "walk",
        "--mode", "slow",
        "--reason", "Reduced torque mode",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    body_yaml = resolver.get_current_body_yaml()
    assert "walk" in body_yaml.capabilities.get("degraded", [])

    events = _capability_events(resolver)
    assert any(e.component == "walk" and e.result.get("action") == "degrade" for e in events)


def test_enable_capability_restores_after_validation(linked_body):
    resolver = BodyResolver()

    with patch.object(sys, "argv", [
        "rosclaw", "body", "capability", "disable",
        "walk",
        "--reason", "Knee actuator overheating",
    ]):
        assert rosclaw_main() == 0

    with patch.object(sys, "argv", [
        "rosclaw", "body", "capability", "enable",
        "walk",
        "--after-validation", "run-42",
    ]):
        assert rosclaw_main() == 0

    body_yaml = resolver.get_current_body_yaml()
    assert "walk" in body_yaml.capabilities.get("enabled", [])
    assert "walk" not in body_yaml.capabilities.get("disabled", [])
    assert "walk" not in body_yaml.capabilities.get("degraded", [])

    events = _capability_events(resolver)
    enable_event = next(e for e in events if e.result.get("action") == "enable")
    assert "run-42" in enable_event.result.get("reason", "")
