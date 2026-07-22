"""Truthfulness and replay gates for the CPU MuJoCo trajectory backend."""

from __future__ import annotations

import json

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
    assert receipt.valid_for_promotion is True


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
