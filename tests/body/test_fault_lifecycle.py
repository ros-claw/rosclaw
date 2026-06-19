"""Tests for rosclaw body fault add/resolve lifecycle."""

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


def test_fault_add_creates_open_fault_and_increments_generation(linked_body):
    resolver = BodyResolver()
    initial = resolver.get_effective_body()
    initial_gen = initial.generation

    with patch.object(sys, "argv", [
        "rosclaw", "body", "fault", "add",
        "--component", "left_knee_joint",
        "--severity", "high",
        "--summary", "temperature abnormal",
    ]):
        assert rosclaw_main() == 0

    events = resolver.get_maintenance_events()
    fault_events = [e for e in events if e.type == "fault"]
    assert len(fault_events) == 1
    assert fault_events[0].component == "left_knee_joint"
    assert fault_events[0].severity == "high"

    effective = resolver.get_effective_body()
    assert effective.generation > initial_gen
    open_faults = [f for f in effective.known_faults if f.get("status") == "open"]
    assert len(open_faults) == 1
    assert open_faults[0]["component"] == "left_knee_joint"


def test_fault_resolve_closes_fault(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "fault", "add",
        "--component", "left_knee_joint",
        "--severity", "high",
        "--summary", "temperature abnormal",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    fault_event = next(e for e in resolver.get_maintenance_events() if e.type == "fault")
    fault_id = fault_event.result["fault_id"]

    with patch.object(sys, "argv", [
        "rosclaw", "body", "fault", "resolve",
        fault_id,
        "--summary", "Replaced actuator",
    ]):
        assert rosclaw_main() == 0

    effective = resolver.get_effective_body()
    open_faults = [f for f in effective.known_faults if f.get("status") == "open"]
    assert not open_faults
    resolved = [f for f in effective.known_faults if f.get("status") == "resolved"]
    assert len(resolved) == 1
