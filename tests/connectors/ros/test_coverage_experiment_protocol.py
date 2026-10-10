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


@pytest.mark.parametrize(
    "name,width,preset,original",
    [
        ("waffle", 0.5, "perimeter_stateless_overlap_continuous", "perimeter_stateless_overlap"),
        (
            "waffle",
            0.5,
            "perimeter_stateless_overlap_boundary_tracking",
            "perimeter_stateless_overlap",
        ),
        (
            "burger",
            0.3,
            "perimeter_stateless_clearance_boundary_tracking",
            "perimeter_stateless_clearance",
        ),
        (
            "burger",
            0.3,
            "perimeter_stateless_clearance_continuous",
            "perimeter_stateless_clearance",
        ),
    ],
)
def test_continuous_boundary_keeps_planner_controller_and_body_unchanged(
    name, width, preset, original
):
    profile = SimpleNamespace(name=name, coverage_width_m=width)
    assert experiments.planning_parameters(profile, preset) == experiments.planning_parameters(
        profile, original
    )
    assert experiments.controller_parameters(profile, preset) == experiments.controller_parameters(
        profile, original
    )
    assert vars(profile) == {"name": name, "coverage_width_m": width}
    wrong = SimpleNamespace(name="burger" if name == "waffle" else "waffle", coverage_width_m=width)
    with pytest.raises(ValueError, match="registered known profile"):
        experiments.planning_parameters(wrong, preset)


@pytest.mark.parametrize(
    "change",
    [
        {"preset": "baseline"},
        {"profile": "burger"},
        {"boundary_strategy": "through_poses"},
        {"boundary_pass": False},
        {"precise_through_poses": False},
        {"boundary_stage_budget_sec": 181},
        {"boundary_stage_budget_sec": 180.0},
        {"boundary_waypoint_count": 5},
        {"boundary_waypoint_count": 9.0},
    ],
)
def test_continuous_boundary_daemon_declaration_rejects_drift(change):
    valid = {
        "preset": "perimeter_stateless_overlap_continuous",
        "profile": "waffle",
        "boundary_strategy": "through_poses_midpoints",
        "boundary_pass": True,
        "precise_through_poses": True,
        "boundary_stage_budget_sec": 180,
        "boundary_waypoint_count": 9,
    }
    experiments.validate_continuous_boundary_experiment(valid)
    with pytest.raises(ValueError, match="nine precise bounded"):
        experiments.validate_continuous_boundary_experiment({**valid, **change})
    experiments.validate_continuous_boundary_experiment({"preset": "baseline"})


def test_continuous_boundary_pair_rejects_unregistered_or_imprecise_execution(monkeypatch):
    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location(
        "continuous_boundary_pairs", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    preset = "perimeter_stateless_overlap_continuous"
    valid = {
        "candidate_boundary_strategy": "through_poses_midpoints",
        "candidate_boundary_stage_budget_sec": 180,
        "candidate_boundary_waypoint_count": 9,
    }
    pairs.validate_continuous_boundary_registration(valid, preset, True)
    for protocol, precise in [
        (valid, False),
        ({}, True),
        ({**valid, "candidate_boundary_stage_budget_sec": 360}, True),
        ({**valid, "candidate_boundary_waypoint_count": 5}, True),
    ]:
        with pytest.raises(ValueError, match="nine precise bounded"):
            pairs.validate_continuous_boundary_registration(protocol, preset, precise)


@pytest.mark.parametrize(
    "preset,profile",
    [
        ("perimeter_stateless_overlap_boundary_tracking", "waffle"),
        ("perimeter_stateless_clearance_boundary_tracking", "burger"),
    ],
)
def test_tracking_boundary_protocol_and_daemon_require_registered_radius_and_digest(
    monkeypatch, preset, profile
):
    valid = {
        "preset": preset,
        "profile": profile,
        "boundary_pass": True,
        "precise_through_poses": True,
        "boundary_strategy": "through_poses_tracking_midpoints",
        "boundary_stage_budget_sec": 180,
        "boundary_waypoint_count": 9,
        "boundary_tracking_prune_radius_m": 0.1,
        "boundary_tracking_bt_sha256": "a" * 64,
    }
    experiments.validate_continuous_boundary_experiment(valid)
    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location(
        "tracking_boundary_pairs", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    protocol = {"candidate_" + k: v for k, v in valid.items() if k.startswith("boundary_")}
    pairs.validate_continuous_boundary_registration(protocol, preset, True)
    for changes in [
        {"boundary_tracking_prune_radius_m": 0.025},
        {"boundary_tracking_prune_radius_m": 0.7},
        {"boundary_tracking_prune_radius_m": True},
        {"boundary_tracking_bt_sha256": None},
        {"boundary_tracking_bt_sha256": "G" * 64},
        {"boundary_tracking_bt_sha256": "a" * 63},
    ]:
        with pytest.raises(ValueError, match="100mm checkpoints"):
            experiments.validate_continuous_boundary_experiment({**valid, **changes})
        with pytest.raises(ValueError, match="100mm checkpoints"):
            pairs.validate_continuous_boundary_registration(
                {**protocol, **{"candidate_" + k: v for k, v in changes.items()}}, preset, True
            )
    with pytest.raises(ValueError, match="nine precise bounded"):
        pairs.validate_continuous_boundary_registration(protocol, preset, False)
    with pytest.raises(ValueError, match="registered candidate"):
        experiments.validate_continuous_boundary_experiment({**valid, "preset": "baseline"})
    with pytest.raises(ValueError, match="registered candidate"):
        pairs.validate_continuous_boundary_registration(protocol, "baseline", True)


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


def test_clearance_diagnostic_changes_only_burger_headland_from_prior_candidate():
    burger = SimpleNamespace(name="burger", coverage_width_m=0.3, physical_radius_m=0.15)
    before = dict(vars(burger))
    old = experiments.planning_parameters(burger, "perimeter_stateless_headland")
    new = experiments.planning_parameters(burger, "perimeter_stateless_clearance")
    assert {key for key in old if old[key] != new[key]} == {"default_headland_width"}
    assert new["default_headland_width"] == 0.5
    assert experiments.controller_parameters(
        burger, "perimeter_stateless_clearance"
    ) == experiments.controller_parameters(burger, "perimeter_stateless_headland")
    assert vars(burger) == before
    assert experiments.planning_parameters(burger)["default_headland_width"] == 0.3


def test_clearance_diagnostic_cannot_silently_apply_to_another_body():
    for name in ["waffle", "unseen_robot"]:
        with pytest.raises(ValueError, match="only for known Burger"):
            experiments.planning_parameters(
                SimpleNamespace(name=name, coverage_width_m=0.5), "perimeter_stateless_clearance"
            )


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
    "name,width,spacing,headland", [("waffle", 0.5, 0.35, 0.5), ("burger", 0.3, 0.27, 0.35)]
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


def test_inner_ring_is_known_burger_opt_in_with_identical_planner_and_controller_settings():
    burger = SimpleNamespace(name="burger", coverage_width_m=0.3, physical_radius_m=0.15)
    old = "perimeter_stateless_clearance"
    new = "perimeter_stateless_clearance_inner_ring"
    assert experiments.planning_parameters(burger, old) == experiments.planning_parameters(
        burger, new
    )
    assert experiments.controller_parameters(burger, old) == experiments.controller_parameters(
        burger, new
    )
    with pytest.raises(ValueError, match="known Burger"):
        experiments.planning_parameters(SimpleNamespace(name="waffle", coverage_width_m=0.5), new)


def test_waffle_ring_retains_original_main_planner_and_controller_without_cross_body_use():
    waffle = SimpleNamespace(name="waffle", coverage_width_m=0.5, physical_radius_m=0.25)
    old, new = "perimeter_stateless", "perimeter_stateless_inner_ring"
    assert experiments.planning_parameters(waffle, new) == experiments.planning_parameters(
        waffle, old
    )
    assert experiments.controller_parameters(waffle, new) == experiments.controller_parameters(
        waffle, old
    )
    with pytest.raises(ValueError, match="known Waffle"):
        experiments.planning_parameters(SimpleNamespace(name="burger", coverage_width_m=0.3), new)


@pytest.mark.parametrize("candidate", list(experiments.INNER_RING_PROFILES))
@pytest.mark.parametrize(
    "budget,inset", [(None, None), (180, 1), (360, 0), (360, True), (True, 1), (360, 2)]
)
def test_extra_ring_cannot_launch_under_an_old_or_changed_stage_registration(
    monkeypatch, budget, inset, candidate
):
    runner = ROOT / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    spec = importlib.util.spec_from_file_location(
        "inner_ring_pairs", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    protocol = {
        "candidate_boundary_stage_budget_sec": budget,
        "candidate_inner_boundary_inset_cells": inset,
    }
    with pytest.raises(ValueError, match="preregistered"):
        pairs.validate_inner_ring_registration(protocol, candidate)
    pairs.validate_inner_ring_registration({}, "perimeter_stateless_clearance")
    pairs.validate_inner_ring_registration(
        {"candidate_boundary_stage_budget_sec": 360, "candidate_inner_boundary_inset_cells": 1},
        candidate,
    )


@pytest.mark.parametrize("preset,profile", list(experiments.INNER_RING_PROFILES.items()))
@pytest.mark.parametrize(
    "changes",
    [
        {"preset": "perimeter_stateless_clearance"},
        {"profile": "third_body"},
        {"preset": []},
        {"boundary_strategy": "sequential"},
        {"boundary_pass": 1},
        {"boundary_stage_budget_sec": 180},
        {"inner_boundary_inset_cells": True},
    ],
)
def test_daemon_rejects_changed_inner_ring_declarations_before_runtime(changes, preset, profile):
    valid = {
        "preset": preset,
        "profile": profile,
        "boundary_strategy": "sequential_inner_ring",
        "boundary_pass": True,
        "boundary_stage_budget_sec": 360,
        "inner_boundary_inset_cells": 1,
    }
    experiments.validate_inner_ring_experiment(valid)
    with pytest.raises(ValueError, match="registered known-fixture"):
        experiments.validate_inner_ring_experiment({**valid, **changes})
    experiments.validate_inner_ring_experiment({})


@pytest.mark.parametrize("preset,profile", list(experiments.INNER_RING_PROFILES.items()))
def test_actual_daemon_cli_rejects_ring_mismatch_before_runtime_or_endpoint(
    tmp_path, preset, profile
):
    import os
    import subprocess
    import sys

    (tmp_path / "execution_config.json").write_text(
        json.dumps(
            {
                "experiment": {
                    "preset": preset,
                    "profile": profile,
                    "boundary_strategy": "sequential_inner_ring",
                    "boundary_pass": True,
                    "boundary_stage_budget_sec": 180,
                    "inner_boundary_inset_cells": 1,
                }
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "integrations/ros_probe/acceptance/daemon.py"),
            "--directory",
            str(tmp_path),
            "--endpoint",
            "ws://127.0.0.1:1",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert (
        "inner boundary experiment must match the registered known-fixture stage" in result.stderr
    )
    assert "KeyError: 'body_id'" not in result.stderr
    assert not (tmp_path / "home").exists() and not (tmp_path / "memory.sqlite").exists()


@pytest.mark.parametrize("preset,profile", list(experiments.INNER_RING_PROFILES.items()))
def test_inner_ring_cannot_swap_a_known_body_or_silently_use_the_other_preset(preset, profile):
    with pytest.raises(ValueError, match="known-fixture"):
        experiments.validate_inner_ring_experiment(
            {
                "preset": preset,
                "profile": "burger" if profile == "waffle" else "waffle",
                "boundary_strategy": "sequential_inner_ring",
                "boundary_pass": True,
                "boundary_stage_budget_sec": 360,
                "inner_boundary_inset_cells": 1,
            }
        )


def test_boundary_tracking_actual_generated_digest_matches_frozen_protocol():
    experiment = {
        "preset": "perimeter_stateless_overlap_boundary_tracking",
        "profile": "waffle",
        "boundary_pass": True,
        "precise_through_poses": True,
        "boundary_strategy": "through_poses_tracking_midpoints",
        "boundary_stage_budget_sec": 180,
        "boundary_waypoint_count": 9,
        "boundary_tracking_prune_radius_m": 0.1,
        "boundary_tracking_bt_sha256": "a" * 64,
    }
    protocol = {"candidate_" + k: v for k, v in experiment.items() if k.startswith("boundary_")}
    protocol["precise_repair_waypoints"] = True
    validate = experiments.validate_boundary_tracking_runtime_registration
    validate(experiment, protocol)
    for field, wrong in [
        ("boundary_tracking_bt_sha256", "b" * 64),
        ("boundary_tracking_prune_radius_m", 0.025),
        ("boundary_stage_budget_sec", 180.0),
        ("boundary_waypoint_count", 8),
        ("boundary_strategy", "through_poses_midpoints"),
    ]:
        with pytest.raises(ValueError, match="preregistered"):
            validate(experiment, {**protocol, "candidate_" + field: wrong})
        missing = dict(protocol)
        missing.pop("candidate_" + field)
        with pytest.raises(ValueError, match="preregistered"):
            validate(experiment, missing)
    with pytest.raises(ValueError, match="runtime protocol"):
        validate(experiment, None)
    with pytest.raises(ValueError, match="precise repair"):
        validate(experiment, {**protocol, "precise_repair_waypoints": False})
    validate({"preset": "baseline"}, None)
