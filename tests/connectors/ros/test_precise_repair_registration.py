"""Freeze candidate-only fixture settings before any container dispatch."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("paired_efficiency")


@pytest.mark.parametrize(
    "protocol,enabled", [({}, False), ({"precise_repair_waypoints": True}, True)]
)
def test_registration_matches_explicit_setting(module, protocol, enabled):
    module.validate_precise_repair_registration(protocol, enabled)


@pytest.mark.parametrize(
    "preset,strategy,enabled",
    [
        ("perimeter_stateless_overlap_continuous", "pose_aware_robust", True),
        ("perimeter_stateless_clearance_continuous", "pose_aware_robust", True),
        ("perimeter_stateless", "pose_aware_robust_sequence", True),
        ("perimeter_stateless", "pose_aware_robust", False),
    ],
)
def test_precise_boundary_is_independent_of_repair_without_relaxing_legacy_gate(
    module, preset, strategy, enabled
):
    module.validate_precise_waypoint_candidate(preset, strategy, enabled)


@pytest.mark.parametrize("strategy", ["greedy", "pose_aware_robust"])
def test_legacy_nonsequence_repair_cannot_enable_unregistered_precise_bt(module, strategy):
    with pytest.raises(ValueError, match="registered continuous boundary"):
        module.validate_precise_waypoint_candidate("perimeter_stateless", strategy, True)


@pytest.mark.parametrize(
    "protocol,enabled",
    [
        ({}, True),
        ({"precise_repair_waypoints": True}, False),
        ({"precise_repair_waypoints": 1}, True),
        ({"precise_repair_waypoints": None}, False),
    ],
)
def test_unregistered_or_non_boolean_setting_refused(module, protocol, enabled):
    with pytest.raises(ValueError):
        module.validate_precise_repair_registration(protocol, enabled)


@pytest.mark.parametrize(
    "arm,enabled,expected",
    [("baseline", True, False), ("candidate", True, True), ("candidate", False, False)],
)
def test_only_explicit_candidate_gets_precise_bt(
    module, monkeypatch, tmp_path, arm, enabled, expected
):
    (tmp_path / "protocol.json").write_text("{}")
    args = SimpleNamespace(
        profile="waffle",
        seed=123,
        candidate="perimeter_stateless",
        candidate_repair_strategy="pose_aware_robust_sequence",
        precise_repair_waypoints=enabled,
        port_base=20191,
        domain_base=81,
        mission_timeout=900,
        image="fixture",
    )
    calls = []

    def refuse_launch(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError("fixture refuses all actual process dispatch")

    monkeypatch.setattr(module, "command", refuse_launch)
    result = module.run_arm(tmp_path, arm, args, 0, "image", "source")
    assert result["status"] == "FAIL"
    assert len(calls) == 1
    assert ("--precise-repair-waypoints" in calls[0][-1]) is expected
    row = json.loads((tmp_path / arm / "run-result.json").read_text())
    assert row["precise_repair_waypoints"] is expected
    if arm == "baseline":
        assert row["preset"] == "baseline" and row["repair_strategy"] == "greedy"
