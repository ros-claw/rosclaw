"""Truthfulness and replay gates for the CPU MuJoCo trajectory backend."""

from __future__ import annotations

import argparse
import json

import pytest

from rosclaw.sandbox.backends import MujocoCpuBackend, RolloutRequest, ScenarioSpec
from rosclaw.sandbox.backends.fingerprints import file_hash
from rosclaw.sandbox.sandbox_api import Sandbox

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
MIDPATH_TABLE_COLLISION = [
    3.4426358094526863,
    -0.7680767522686045,
    2.253070730803216,
    2.480201653011009,
    -5.099721659051599,
    5.976851207161098,
]


def _request(tmp_path, trajectory, *, world="tabletop"):
    sandbox = Sandbox.create("ur5e", world, "mujoco")
    assert sandbox.has_physics, sandbox.load_error
    scenario = ScenarioSpec(
        scenario_id="trajectory-regression",
        robot_id="ur5e",
        world_id=world,
        body_snapshot_hash="sha256:test-body-snapshot",
        model_hash=file_hash(sandbox.model_path),
        seed=42,
    )
    request = RolloutRequest(
        scenario=scenario,
        trajectory=trajectory,
        artifact_dir=tmp_path,
    )
    return sandbox, MujocoCpuBackend(sandbox), request


def test_full_trajectory_detects_midpath_collision_with_safe_endpoint(tmp_path):
    sandbox, backend, request = _request(tmp_path, [MIDPATH_TABLE_COLLISION, HOME])
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert request.trajectory[-1] == HOME
    assert receipt.physics_executed is True
    assert receipt.is_safe is False
    assert "COLLISION_DETECTED" in receipt.violations
    assert any("tabletop_surface" in pair for pair in receipt.collision_pairs)
    assert receipt.valid_for_promotion is False


def test_safe_trajectory_is_deterministic_and_strictly_replayable(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    try:
        receipt = backend.rollout(request)
        report = backend.replay(receipt, strict=True)
    finally:
        sandbox.close()

    assert receipt.is_safe is True
    assert receipt.physics_executed is True
    assert report.verified is True
    assert report.hashes_verified is True
    assert report.final_qpos_max_abs_error == 0.0
    assert receipt.replay_report["verified"] is True
    assert receipt.valid_for_promotion is False


def test_strict_replay_rejects_tampered_artifact(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    try:
        receipt = backend.rollout(request)
        states = tmp_path / "trajectory_states.json"
        payload = json.loads(states.read_text(encoding="utf-8"))
        payload["states"].append({"tampered": True})
        states.write_text(json.dumps(payload), encoding="utf-8")
        report = backend.replay(receipt, strict=True)
    finally:
        sandbox.close()

    assert report.verified is False
    assert report.hashes_verified is False
    assert "artifact_hash:trajectory_states.json" in report.mismatches


def test_strict_replay_rejects_tampered_deterministic_metric(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    try:
        receipt = backend.rollout(request)
        assert backend.replay(receipt, strict=True).verified is True
        tampered = receipt.to_dict()
        tampered["metrics"]["steps"] += 1
        report = backend.replay(tampered, strict=True)
    finally:
        sandbox.close()

    assert report.verified is False
    assert "metrics:steps" in report.mismatches


def test_dimension_mismatch_fails_closed_without_physics(tmp_path):
    sandbox, backend, request = _request(tmp_path, [[0.0, 0.0]], world="empty")
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.is_safe is False
    assert receipt.physics_executed is False
    assert receipt.reason == "ACTION_DIMENSION_MISMATCH"
    assert receipt.valid_for_promotion is False


def test_seed_randomization_changes_initial_physics_state(tmp_path):
    hashes = []
    for seed in (42, 43):
        sandbox, backend, request = _request(tmp_path / str(seed), [HOME], world="empty")
        request = RolloutRequest(
            scenario=ScenarioSpec(
                **{
                    **request.scenario.__dict__,
                    "seed": seed,
                    "metadata": {"initial_qpos_jitter_rad": 0.002},
                }
            ),
            trajectory=request.trajectory,
            artifact_dir=request.artifact_dir,
        )
        try:
            receipt = backend.rollout(request)
        finally:
            sandbox.close()
        hashes.append(receipt.randomization["initial_state_hash"])

    assert len(set(hashes)) == 2


@pytest.mark.parametrize(("seed", "reason"), [(-1, "INVALID_SEED"), (True, "INVALID_SEED")])
def test_invalid_seed_fails_closed(tmp_path, seed, reason):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=ScenarioSpec(**{**request.scenario.__dict__, "seed": seed}),
        trajectory=request.trajectory,
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is False
    assert receipt.reason == reason


def test_control_period_must_match_mujoco_timestep(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=request.scenario,
        trajectory=request.trajectory,
        control_dt_sec=0.01,
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is False
    assert receipt.reason == "UNSUPPORTED_CONTROL_DT"


def test_rollout_step_budget_fails_closed(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=request.scenario,
        trajectory=request.trajectory,
        settle_steps=100,
        max_steps=1,
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is True
    assert receipt.is_safe is False
    assert "SIMULATION_STEP_BUDGET_EXCEEDED" in receipt.violations
    assert receipt.metrics["steps"] == 1


def test_rollout_rejects_excessive_step_budget(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=request.scenario,
        trajectory=request.trajectory,
        max_steps=250_001,
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is False
    assert receipt.reason == "INVALID_STEP_BUDGET"


def test_rollout_rejects_non_json_scenario_metadata(tmp_path):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=ScenarioSpec(
            **{**request.scenario.__dict__, "metadata": {"invalid": float("nan")}}
        ),
        trajectory=request.trajectory,
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is False
    assert receipt.reason == "INVALID_SCENARIO_METADATA"


@pytest.mark.parametrize(
    ("field", "value", "mismatch"),
    [
        ("request", "not-a-mapping", "request_contract"),
        ("backend", "not-a-mapping", "backend_contract"),
        ("final_qpos", {"invalid": True}, "final_qpos"),
        ("randomization", "not-a-mapping", "randomization"),
    ],
)
def test_replay_malformed_receipt_fails_closed(tmp_path, field, value, mismatch):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    try:
        receipt = backend.rollout(request).to_dict()
        receipt[field] = value
        report = backend.replay(receipt, strict=True)
    finally:
        sandbox.close()

    assert report.verified is False
    assert mismatch in report.mismatches


@pytest.mark.parametrize("waypoint", [[True] * 6, 1])
def test_invalid_waypoint_types_return_failed_receipt(tmp_path, waypoint):
    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    request = RolloutRequest(
        scenario=request.scenario,
        trajectory=[waypoint],
        artifact_dir=request.artifact_dir,
    )
    try:
        receipt = backend.rollout(request)
    finally:
        sandbox.close()

    assert receipt.physics_executed is False
    assert receipt.reason in {"NON_FINITE_WAYPOINT", "INVALID_WAYPOINT"}


def test_backend_rejects_free_joint_actuator_layout():
    sandbox = Sandbox.create("unitree_g1", "empty", "mujoco")
    try:
        assert sandbox.has_physics, sandbox.load_error
        with pytest.raises(RuntimeError, match="UNSUPPORTED_ACTUATOR_LAYOUT"):
            MujocoCpuBackend(sandbox).compile(
                ScenarioSpec(
                    scenario_id="unsupported-g1",
                    robot_id="unitree_g1",
                    world_id="empty",
                    body_snapshot_hash="sha256:test",
                    model_hash=file_hash(sandbox.model_path),
                )
            )
    finally:
        sandbox.close()


def test_backend_prioritizes_request_shape_before_unsupported_layout(tmp_path):
    sandbox = Sandbox.create("unitree_g1", "empty", "mujoco")
    try:
        assert sandbox.has_physics, sandbox.load_error
        backend = MujocoCpuBackend(sandbox)
        scenario = ScenarioSpec(
            scenario_id="unsupported-g1-rollout",
            robot_id="unitree_g1",
            world_id="empty",
            body_snapshot_hash="sha256:test-body",
            model_hash=file_hash(sandbox.model_path),
        )

        malformed = backend.rollout(
            RolloutRequest(
                scenario=scenario,
                trajectory=[[0.0, 0.0, 0.0]],
                artifact_dir=tmp_path / "malformed",
            )
        )
        unsupported = backend.rollout(
            RolloutRequest(
                scenario=scenario,
                trajectory=[[0.0] * int(sandbox.physics_model.nu)],
                artifact_dir=tmp_path / "unsupported",
            )
        )
    finally:
        sandbox.close()

    assert malformed.reason == "ACTION_DIMENSION_MISMATCH"
    assert malformed.physics_executed is False
    assert unsupported.reason == "UNSUPPORTED_ACTUATOR_LAYOUT"
    assert unsupported.physics_executed is False


def test_robot_identifier_cannot_escape_model_zoo():
    sandbox = Sandbox.create("../ur5e", "empty", "mujoco")
    try:
        assert sandbox.has_physics is False
        assert sandbox.load_error == "Invalid robot identifier."
    finally:
        sandbox.close()


def test_cli_receipt_replay_always_performs_strict_verification(tmp_path, capsys):
    from rosclaw.cli import cmd_sandbox_replay

    sandbox, backend, request = _request(tmp_path, [HOME], world="empty")
    try:
        receipt = backend.rollout(request)
        assert backend.replay(receipt, strict=True).verified is True
    finally:
        sandbox.close()

    result = cmd_sandbox_replay(
        argparse.Namespace(
            receipt=str(tmp_path / "simulation_receipt.json"),
            robot=None,
            world=None,
            json=True,
            episode_id=None,
        )
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["verified"] is True
    assert report["hashes_verified"] is True


def test_cli_receipt_replay_rejects_non_object_json(tmp_path, capsys):
    from rosclaw.cli import cmd_sandbox_replay

    receipt = tmp_path / "receipt.json"
    receipt.write_text("[]", encoding="utf-8")
    result = cmd_sandbox_replay(
        argparse.Namespace(
            receipt=str(receipt),
            robot=None,
            world=None,
            json=True,
            episode_id=None,
        )
    )
    assert result == 2
    assert "receipt root must be a JSON object" in capsys.readouterr().err


def test_cli_legacy_replay_rejects_path_escape(tmp_path, monkeypatch, capsys):
    from rosclaw.cli import cmd_sandbox_replay

    monkeypatch.chdir(tmp_path)
    result = cmd_sandbox_replay(
        argparse.Namespace(receipt=None, episode_id="../../etc", json=False)
    )
    assert result == 2
    assert "Invalid legacy episode identifier" in capsys.readouterr().err
