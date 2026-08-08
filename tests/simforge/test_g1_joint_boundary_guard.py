from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rosclaw.simforge.g1_joint_boundary_guard import (
    G1JointBoundaryGuardConfig,
    G1JointBoundaryGuardPolicy,
)
from rosclaw.simforge.g1_neural_torque import G1TorqueControlFrame


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _frame(**changes: object) -> G1TorqueControlFrame:
    values: dict[str, object] = {
        "joint_position": np.zeros(29),
        "joint_velocity": np.zeros(29),
        "joint_lower_limits": np.full(29, -1.0),
        "joint_upper_limits": np.full(29, 1.0),
        "torso_quaternion_wxyz": np.asarray((1.0, 0.0, 0.0, 0.0)),
        "pelvis_position": np.asarray((0.0, 0.0, 0.8)),
        "base_linear_velocity": np.zeros(3),
        "base_angular_velocity": np.zeros(3),
        "ball_position": np.asarray((1.0, 0.0, 0.115)),
        "ball_velocity": np.zeros(3),
        "target_y_m": 0.75,
        "target_z_m": 0.55,
        "policy_phase": 0.4,
        "left_contact": True,
        "right_contact": False,
    }
    values.update(changes)
    return G1TorqueControlFrame(**values)  # type: ignore[arg-type]


def _policy(
    config: G1JointBoundaryGuardConfig | None = None,
) -> G1JointBoundaryGuardPolicy:
    return G1JointBoundaryGuardPolicy(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        config=config,
    )


def test_guard_projects_only_audited_outward_joint_torque() -> None:
    position = np.zeros(29)
    velocity = np.zeros(29)
    position[3] = -0.98
    velocity[3] = -1.0
    position[4] = -0.98
    velocity[4] = -1.0
    parent = np.full(29, -10.0)
    guard = _policy()

    projected = guard.command(
        _frame(joint_position=position, joint_velocity=velocity),
        parent,
    )

    assert projected[3] > parent[3]
    assert projected[4] == parent[4]
    assert np.count_nonzero(projected != parent) == 1
    guard.note_applied(projected)
    receipt = guard.build_receipt()
    assert receipt.projection_count == 1
    assert receipt.projected_joint_count == 1
    assert receipt.safety_projection_only
    assert not receipt.direct_torque_actor
    assert not receipt.hardware_authorized


def test_guard_is_exact_parent_outside_boundary_and_before_phase() -> None:
    guard = _policy(G1JointBoundaryGuardConfig(minimum_policy_phase=0.5))
    parent = np.linspace(-5.0, 5.0, 29)

    projected = guard.command(_frame(policy_phase=0.49), parent)

    assert np.array_equal(projected, parent)
    guard.note_applied(projected)
    assert guard.build_receipt().projection_count == 0


def test_guard_caps_correction_and_requires_note_applied() -> None:
    guard = _policy(G1JointBoundaryGuardConfig(maximum_correction_nm=2.0))
    position = np.zeros(29)
    velocity = np.zeros(29)
    position[5] = -0.99
    velocity[5] = -10.0
    parent = np.full(29, -20.0)

    projected = guard.command(
        _frame(joint_position=position, joint_velocity=velocity),
        parent,
    )

    assert projected[5] - parent[5] == pytest.approx(2.0)
    with pytest.raises(RuntimeError, match="note_applied"):
        guard.command(_frame(), parent)
    guard.note_applied(projected)


def test_default_guard_config_is_sim_only() -> None:
    assert G1JointBoundaryGuardConfig().activation_ceiling == "SIM_ONLY"


def test_guard_rejects_unknown_joint_and_nonfinite_state() -> None:
    with pytest.raises(ValueError, match="unknown G1 joint"):
        G1JointBoundaryGuardConfig(protected_joint_names=("not-a-joint",))
    guard = _policy()
    position = np.zeros(29)
    position[3] = np.nan

    with pytest.raises(ValueError, match="finite 29-DoF"):
        guard.command(_frame(joint_position=position), np.zeros(29))
