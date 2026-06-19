"""Tests for rosclaw body note."""

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


def test_note_appends_jsonl(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "note",
        "Camera disconnected during navigation test.",
        "--type", "incident",
        "--severity", "warning",
        "--affects", "head_camera,visual_navigation",
    ]):
        assert rosclaw_main() == 0

    resolver = BodyResolver()
    events = resolver.get_maintenance_events()
    assert any("Camera disconnected" in e.message for e in events)
    incident = next(e for e in events if "Camera disconnected" in e.message)
    assert incident.requires_skill_recheck
    assert "head_camera" in incident.affects
