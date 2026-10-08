"""Paired SIM setup must retain Body-specific defaults and reject unsafe arms."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "coverage_experiments", ROOT / "integrations/ros_probe/acceptance/experiments.py"
)
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)


def test_profile_specific_baseline_and_body_are_preserved():
    waffle = SimpleNamespace(name="waffle", coverage_width_m=0.5, physical_radius_m=0.25)
    burger = SimpleNamespace(name="burger", coverage_width_m=0.3, physical_radius_m=0.15)
    before = dict(vars(burger))
    assert experiments.planning_parameters(waffle)["default_headland_width"] == 0.5
    assert experiments.planning_parameters(burger)["default_headland_width"] == 0.3
    experiments.planning_parameters(burger, "headland")
    assert vars(burger) == before
    with pytest.raises(ValueError, match="safe offline clearance"):
        experiments.planning_parameters(burger, "diagonal")
    with pytest.raises(ValueError, match="unknown"):
        experiments.planning_parameters(waffle, "arbitrary")


def test_seed_changes_only_sim_randomness_and_is_validated():
    old = ["gz", "sim", "-r", "-s", "--headless-rendering", "/evidence/world.sdf"]
    assert experiments.gazebo_arguments("/evidence/world.sdf") == old
    seeded = experiments.gazebo_arguments("/evidence/world.sdf", 100801)
    assert seeded == old[:-1] + ["--seed", "100801", old[-1]]
    for invalid in [True, -1, 2**31, 1.0, "1"]:
        with pytest.raises(ValueError, match="SIM seed"):
            experiments.gazebo_arguments("world.sdf", invalid)


def test_failed_journey_stops_owned_fixture_and_retains_failure(tmp_path, monkeypatch):
    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location("coverage_pairs", runner / "paired_efficiency.py")
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    (tmp_path / "protocol.json").write_text("{}")
    calls = []

    def fake_command(args, **kwargs):
        calls.append(args)
        return "owned-container"

    def fail_journey(*args):
        raise RuntimeError("injected incomplete journey")

    monkeypatch.setattr(pairs, "command", fake_command)
    monkeypatch.setattr(pairs, "wait_ready", lambda *args: None)
    monkeypatch.setattr(pairs, "journey", fail_journey)
    args = SimpleNamespace(
        profile="waffle",
        seed=100801,
        port_base=20191,
        domain_base=201,
        candidate="diagonal",
        image="test-image",
        mission_timeout=900,
    )
    result = pairs.run_arm(tmp_path, "candidate", args, 0, "test-digest", "test-commit")
    assert result["status"] == "FAIL"
    assert "incomplete journey" in result["failure"]
    assert any(call[:2] == ["docker", "stop"] for call in calls)
    assert not any(call[:2] == ["docker", "rm"] for call in calls)
    assert json.loads((tmp_path / "candidate/run-result.json").read_text()) == result
    assert (tmp_path / "candidate/protocol.json").read_text() == "{}"
