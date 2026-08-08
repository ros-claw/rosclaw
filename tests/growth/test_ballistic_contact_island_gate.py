from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_island_gate import (
    derive_g1_ballistic_contact_island_gate,
    load_g1_ballistic_contact_island_gate,
)
from rosclaw.growth.cli import dispatch_growth_argv


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(
    root: Path,
    index: int,
    lead: float,
    trail: float,
    qualified: bool,
) -> Path:
    trajectory = root / f"trajectory-{index}.npz"
    count = 12
    peak = np.zeros(count, dtype=np.float64)
    peak[5] = 900.0
    np.savez_compressed(
        trajectory,
        right_foot_linear_velocity=np.zeros((count, 3), dtype=np.float64),
        ball_contact_force_peak_n=peak,
        ball_contact_normal=np.zeros((count, 3), dtype=np.float64),
        ball_contact_force_world=np.zeros((count, 3), dtype=np.float64),
    )
    crossing_height = 0.95 if qualified else 0.32
    error = 0.42 if qualified else 1.03
    path = root / f"evidence-{index}.json"
    path.write_text(
        json.dumps(
            {
                "body_hash": "sha256:" + "1" * 64,
                "implementation_hash": "sha256:" + "2" * 64,
                "strict_replay": True,
                "trajectory_path": str(trajectory.resolve()),
                "trajectory_hash": _hash(trajectory),
                "approach_strike_candidate_hash": "sha256:" + "3" * 64,
                "claims": {"contact_dynamics_observed_from_physics": True},
                "flow_config": {
                    "schema_version": "flow.v1",
                    "ballistic_contact_residual_rad": [
                        -0.02,
                        -0.01,
                        -0.06,
                        -0.06,
                        0.25,
                        0.03,
                    ],
                    "ballistic_contact_policy_frame": 256,
                    "ballistic_contact_lead_duration_sec": lead,
                    "ballistic_contact_trail_duration_sec": trail,
                    "post_contact_damping_scale": 2.5,
                },
                "sonic_runup_config": {
                    "schema_version": "sonic.v1",
                    "planner_seed": 0,
                },
                "runup_config": {"start_x_m": -3.4},
                "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
                "result": {
                    "finite_state": True,
                    "post_kick_fall": False,
                    "joint_limit_violation": False,
                    "torque_limit_violation": False,
                    "perceptual_continuity_passed": True,
                    "kick_contact_observed": True,
                    "kick_contact_peak_force_n": 900.0,
                    "goal_crossed": True,
                    "goal_plane_target_error_m": error,
                    "goal_crossing_xyz_m": [6.0, 0.9, crossing_height],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _dataset(root: Path) -> tuple[Path, ...]:
    cases = (
        (0.08, 0.065, True),
        (0.12, 0.065, True),
        (0.14, 0.065, True),
        (0.16, 0.065, True),
        (0.16, 0.070, True),
        (0.16, 0.075, True),
        (0.16, 0.080, True),
        (0.159, 0.065, False),
        (0.1605, 0.065, False),
        (0.162, 0.065, False),
        (0.18, 0.065, False),
        (0.20, 0.065, False),
    )
    return tuple(
        _evidence(root, index, lead, trail, qualified)
        for index, (lead, trail, qualified) in enumerate(cases)
    )


def test_contact_island_gate_learns_discontinuous_lead_boundary(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path)
    output = tmp_path / "gate.json"

    gate = derive_g1_ballistic_contact_island_gate(
        evidence_paths=paths,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )

    assert gate.event_feature_name == "lead_duration_sec"
    assert gate.event_tolerance == pytest.approx(0.0001)
    assert gate.qualified_event_values == (0.08, 0.12, 0.14, 0.16)
    assert gate.rejected_event_values == (0.159, 0.1605, 0.162, 0.18, 0.2)
    assert gate.training_balanced_accuracy == 1.0
    assert gate.leave_one_out_balanced_accuracy >= 0.75
    assert gate.leave_one_out_qualified_recall == pytest.approx(4 / 7)
    assert gate.leave_one_out_rejected_recall == 1.0
    assert gate.training_ready
    inside = (-0.02, -0.01, -0.06, -0.06, 0.25, 0.03, 256.0, 0.16, 0.065)
    edge = (*inside[:7], 0.1599, 0.065)
    unseen = (*inside[:7], 0.1598, 0.065)
    rejected = (*inside[:7], 0.159, 0.065)
    unsupported = (*inside[:7], 0.17, 0.065)
    outside = (0.01, *inside[1:])
    assert gate.predict(inside).island_admissible
    assert gate.predict(edge).island_admissible
    assert gate.predict(unseen).reason == "UNSEEN_CONTACT_EVENT"
    assert gate.predict(rejected).reason == "REJECTED_CONTACT_ISLAND"
    assert gate.predict(unsupported).reason == "OUTSIDE_QUALIFIED_REPLAY_SUPPORT"
    assert gate.predict(outside).reason == "OUTSIDE_QUALIFIED_REPLAY_SUPPORT"
    assert load_g1_ballistic_contact_island_gate(output) == gate
    assert (
        dispatch_growth_argv(
            [
                "growth",
                "ballistic-contact-island-gate",
                *[
                    item
                    for path in paths
                    for item in ("--evidence-json", str(path))
                ],
                "--output",
                str(tmp_path / "cli-gate.json"),
                "--source-checkout",
                str(tmp_path / "checkout"),
            ]
        )
        == 0
    )


def test_contact_island_gate_rejects_context_mixing(tmp_path: Path) -> None:
    paths = list(_dataset(tmp_path))
    mixed = json.loads(paths[-1].read_text(encoding="utf-8"))
    mixed["goal_spec"]["target_z_m"] = 0.65
    paths[-1].write_text(json.dumps(mixed), encoding="utf-8")

    with pytest.raises(ValueError, match="contexts disagree"):
        derive_g1_ballistic_contact_island_gate(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "gate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_contact_island_gate_rejects_duplicate_controls(tmp_path: Path) -> None:
    paths = list(_dataset(tmp_path))
    duplicate = json.loads(paths[-1].read_text(encoding="utf-8"))
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    duplicate["flow_config"] = first["flow_config"]
    paths[-1].write_text(json.dumps(duplicate), encoding="utf-8")

    with pytest.raises(ValueError, match="controls must be independent"):
        derive_g1_ballistic_contact_island_gate(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "gate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_contact_island_gate_recomputes_geometry_on_load(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    derive_g1_ballistic_contact_island_gate(
        evidence_paths=_dataset(tmp_path),
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value.pop("gate_hash")
    value["event_tolerance"] = 0.02
    value["gate_hash"] = canonical_hash(value)
    output.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="geometry is invalid"):
        load_g1_ballistic_contact_island_gate(output)
