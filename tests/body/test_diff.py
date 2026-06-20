"""Tests for rosclaw body diff."""

import sys
from unittest.mock import patch

import pytest

from rosclaw.body.diff import BodyDiffer
from rosclaw.body.schema import EffectiveBody
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


def _make_effective(**overrides) -> EffectiveBody:
    defaults = {
        "body_instance_id": "test",
        "eurdf_uri": "rosclaw://eurdf/unitree-g1@1.0.0",
        "effective_body_hash": "h",
        "compiled_at": "t",
        "frames": {"root": "base_link", "world": "world"},
        "joints": {"waist": {"parent": "base", "child": "torso"}},
        "identity": {"robot_class": "humanoid"},
    }
    defaults.update(overrides)
    return EffectiveBody(**defaults)


def test_diff_detects_joint_added_removed_changed():
    old = _make_effective()
    new = _make_effective(
        joints={
            "waist": {"parent": "base", "child": "torso", "axis": "z"},
            "neck": {"parent": "torso", "child": "head"},
        },
    )
    diff = BodyDiffer().diff_effective_bodies(old, new)
    assert diff.requires_skill_recheck
    categories = {c.category for c in diff.changes}
    assert "structural" in categories
    severities = {c.severity for c in diff.changes}
    assert "critical" in severities
    assert any(c.path == "joints.neck" and c.new == "present" for c in diff.changes)
    assert any("topology changed" in (c.reason or "") for c in diff.changes)


def test_diff_detects_frame_removed_and_identity_change():
    old = _make_effective()
    new = _make_effective(
        frames={"root": "base_link"},
        identity={"robot_class": "humanoid", "site": "lab1"},
    )
    diff = BodyDiffer().diff_effective_bodies(old, new)
    assert any(c.path == "frames.world" and c.old == "present" for c in diff.changes)
    assert any(c.path == "identity.site" and c.new == "lab1" for c in diff.changes)
    assert "structural" in {c.category for c in diff.changes}
