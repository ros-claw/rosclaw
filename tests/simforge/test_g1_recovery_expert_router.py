from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.simforge.g1_contact_recovery_torque_policy import (
    G1RecoveryContextSnapshot,
)
from rosclaw.simforge.g1_recovery_expert_router import (
    build_g1_recovery_expert_router,
    load_g1_recovery_expert_router,
    write_g1_recovery_expert_router,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _context(speed: float, *, lateral: float = 0.0) -> G1RecoveryContextSnapshot:
    return G1RecoveryContextSnapshot(
        policy_phase=0.42,
        pelvis_height_m=0.74,
        projected_gravity_x=0.02,
        projected_gravity_y=-0.10,
        projected_gravity_z=-0.99,
        base_linear_velocity_x_mps=0.2,
        base_linear_velocity_y_mps=lateral,
        base_linear_velocity_z_mps=0.1,
        base_angular_velocity_x_rps=-0.2,
        base_angular_velocity_y_rps=0.4,
        base_angular_velocity_z_rps=0.1,
        ball_speed_mps=speed,
        ball_direction_x=1.0,
        ball_direction_y=0.0,
        ball_direction_z=0.0,
        left_contact=True,
        right_contact=False,
    )


def _artifact():  # type: ignore[no-untyped-def]
    return build_g1_recovery_expert_router(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        source_evidence_hash=_digest("evidence"),
        contexts=(_context(2.0), _context(3.0, lateral=0.2)),
        selected_expert_ids=("nominal", None),
        measured_gains=(0.20, 0.0),
        task_preserved=(True, True),
        expert_artifact_hashes={"nominal": _digest("nominal")},
        maximum_normalized_distance=0.25,
    )


def test_router_selects_qualified_prototype_and_fails_closed() -> None:
    artifact = _artifact()

    selected = artifact.route(_context(2.0))
    fallback = artifact.route(_context(3.0, lateral=0.2))
    outlier = artifact.route(replace(_context(2.0), ball_speed_mps=20.0))

    assert selected.eligible
    assert selected.expert_id == "nominal"
    assert selected.expected_naturalness_gain == pytest.approx(0.20)
    assert not fallback.eligible
    assert fallback.fallback_reason == "prototype_requires_parent_fallback"
    assert not outlier.eligible
    assert outlier.fallback_reason == "outside_sealed_context_envelope"
    assert not selected.hardware_authorized


def test_router_artifact_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    path = tmp_path / "router.json"
    artifact = _artifact()
    write_g1_recovery_expert_router(path, artifact)

    loaded = load_g1_recovery_expert_router(path)
    assert loaded.artifact_hash == artifact.artifact_hash
    value = json.loads(path.read_text())
    value["minimum_expected_gain"] = 0.01
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_recovery_expert_router(path)
