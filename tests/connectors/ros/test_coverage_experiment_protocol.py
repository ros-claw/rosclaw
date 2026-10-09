"""Paired SIM setup must retain Body-specific defaults and reject unsafe arms."""

import importlib.util
import json
from datetime import UTC, datetime
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


@pytest.mark.parametrize("live_state", ["active", "inactive", "stale", "missing"])
def test_startup_ready_requires_live_states_without_task_snapshot(
    tmp_path, monkeypatch, live_state
):
    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location(
        "coverage_pair_ready", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    (tmp_path / "measured_map.json").write_text("{}")
    for name in ["nav2.log", "coverage_lifecycle.log"]:
        (tmp_path / name).write_text("Managed nodes are active")
    sample = {
        "captured_at": datetime.now(UTC).isoformat(),
        "observation_complete": True,
        "collision_count": 0,
        "cleaning_enabled": False,
    }
    monkeypatch.setattr(pairs, "command", lambda *args: "true")
    monkeypatch.setattr(pairs, "latest_completed_observation", lambda *args: sample)
    from lifecycle_readiness import REQUIRED_NODES, SCHEMA

    lifecycle = {
        "schema_version": SCHEMA,
        "source": "actual_read_only_GetState_responses",
        "ready": True,
        "responses": {
            name: {
                "service": "/" + name + "/get_state",
                "state_id": 3,
                "state_label": "active",
                "received_monotonic_sec": pairs.time.monotonic(),
            }
            for name in REQUIRED_NODES
        },
    }
    if live_state == "inactive":
        lifecycle["responses"]["coverage_server"]["state_id"] = 2
    elif live_state == "stale":
        lifecycle["responses"]["coverage_server"]["received_monotonic_sec"] -= 3
    elif live_state == "missing":
        lifecycle["responses"]["coverage_server"] = None
    (tmp_path / "lifecycle-readiness.json").write_text(json.dumps(lifecycle))
    (tmp_path / "witness.jsonl").write_text(json.dumps(sample) + "\n")
    if live_state == "active":
        pairs.wait_ready(tmp_path, "mock-owned", timeout=1)
    else:
        with pytest.raises(TimeoutError, match="fresh_actual_ACTIVE_lifecycle_responses"):
            pairs.wait_ready(tmp_path, "mock-owned", timeout=0.05)
        retained = json.loads((tmp_path / "startup-gate-failure.json").read_text())
        assert retained["readiness"] is False
        assert retained["original_lifecycle_response_snapshot"] == lifecycle
    assert not (tmp_path / "snapshot.json").exists()


def test_rotation_latch_candidate_changes_no_speed_guard_or_planner_setting():
    for name, width in [("waffle", 0.5), ("burger", 0.3)]:
        profile = SimpleNamespace(name=name, coverage_width_m=width)
        before = dict(vars(profile))
        assert experiments.controller_parameters(profile, "baseline") == {}
        assert experiments.controller_parameters(profile, "perimeter_sequential") == {}
        assert experiments.controller_parameters(profile, "perimeter_stateless") == {
            "stateful": False
        }
        assert experiments.planning_parameters(
            profile, "perimeter_stateless"
        ) == experiments.planning_parameters(profile, "baseline")
        assert vars(profile) == before


def test_burger_combined_candidate_only_adds_screened_headland_to_stateless():
    profile = SimpleNamespace(name="burger", coverage_width_m=0.3)
    original = experiments.planning_parameters(profile, "perimeter_stateless")
    combined = experiments.planning_parameters(profile, "perimeter_stateless_headland")
    assert combined.pop("default_headland_width") == 0.35
    original.pop("default_headland_width")
    assert combined == original
    assert experiments.controller_parameters(profile, "perimeter_stateless_headland") == {
        "stateful": False
    }
    with pytest.raises(ValueError, match="only for Burger"):
        experiments.planning_parameters(
            SimpleNamespace(name="waffle", coverage_width_m=0.5), "perimeter_stateless_headland"
        )


@pytest.mark.parametrize(
    "name,width,spacing,headland", [("waffle", 0.5, 0.4, 0.5), ("burger", 0.3, 0.27, 0.35)]
)
def test_overlap_candidate_changes_only_spacing_of_existing_safe_main_configuration(
    name, width, spacing, headland
):
    profile = SimpleNamespace(name=name, coverage_width_m=width)
    original = experiments.planning_parameters(
        profile, "perimeter_stateless" if name == "waffle" else "perimeter_stateless_headland"
    )
    candidate = experiments.planning_parameters(profile, "perimeter_stateless_overlap")
    assert candidate.pop("operation_width") == spacing
    assert candidate == original and candidate["default_headland_width"] == headland
    assert experiments.controller_parameters(profile, "perimeter_stateless_overlap") == {
        "stateful": False
    }
    assert vars(profile) == {"name": name, "coverage_width_m": width}
    assert "operation_width" not in experiments.planning_parameters(profile, "baseline")


def test_repair_metrics_preserve_requested_waypoints_separately_from_goal_count(
    tmp_path, monkeypatch
):
    from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog

    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location(
        "coverage_pair_counts", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    audit = CoverageAuditLog(
        tmp_path / "actions/coverage-audit-source.jsonl", context={"run_id": "source-only"}
    )
    audit.emit("goal_started", {"stage": "REPAIR", "nav_goal_id": "single", "goal": {"pose": {}}})
    audit.emit(
        "goal_started", {"stage": "REPAIR", "nav_goal_id": "sequence", "goal": {"poses": [{}, {}]}}
    )
    audit.emit(
        "goal_started",
        {"stage": "BOUNDARY_PASS", "nav_goal_id": "boundary", "goal": {"poses": [{}, {}]}},
    )
    audit.close()
    assert pairs.repair_request_counts(tmp_path) == {
        "repair_requested_goal_count": 2,
        "repair_requested_waypoint_count": 3,
    }
