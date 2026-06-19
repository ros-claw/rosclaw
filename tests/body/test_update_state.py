"""Tests for rosclaw body update-state."""

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


def test_update_state_changes_hash(linked_body):
    resolver = BodyResolver()
    old_hash = resolver.get_effective_body_hash()

    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0

    new_hash = resolver.get_effective_body_hash()
    assert new_hash != old_hash
    effective = resolver.get_effective_body()
    assert effective.sensors["head_camera"]["status"] == "unavailable"


def test_update_state_maintenance_log(linked_body):
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "safety_overrides.max_base_speed_mps=0.2",
        "--reason", "indoor environment",
    ]):
        assert rosclaw_main() == 0
    resolver = BodyResolver()
    events = resolver.get_maintenance_events()
    assert any("safety_overrides.max_base_speed_mps" in e.message for e in events)


def test_update_state_forbidden_field():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"HOME": tmp}):
        with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
            assert rosclaw_main() == 0
        with patch.object(sys, "argv", [
            "rosclaw", "body", "update-state",
            "--set", "model_ref.profile_id=other",
        ]):
            assert rosclaw_main() == 1
