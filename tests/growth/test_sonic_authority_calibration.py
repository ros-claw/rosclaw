from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.growth.approach_strike_contracts import STATE_FEATURES
from rosclaw.growth.approach_strike_residual import (
    G1ApproachStrikeResidualConfig,
    build_online_approach_strike_state,
)
from rosclaw.growth.sonic_authority_calibration import (
    derive_g1_sonic_authority_calibration,
    load_g1_sonic_authority_calibration,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_sonic_authority_calibration_is_replay_bound_and_joint_local(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.npz"
    commanded = np.zeros((60, 29), dtype=np.float64)
    commanded[:20, 4] = 65.0
    commanded[20:40, 2] = 120.0
    commanded[40:, 1] = 160.0
    np.savez_compressed(
        trajectory,
        commanded_torque=commanded,
        event_phase=np.asarray((0,) * 20 + (4,) * 20 + (6,) * 20, dtype=np.int64),
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "trajectory_path": str(trajectory.resolve()),
                "trajectory_hash": _hash(trajectory),
                "strict_replay": True,
                "body_hash": "sha256:" + "1" * 64,
                "implementation_hash": "sha256:" + "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "calibration.json"

    calibration = derive_g1_sonic_authority_calibration(
        trajectory_paths=(trajectory,),
        evidence_paths=(evidence,),
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )

    assert calibration.joint_gain_scales[4] < 1.0
    assert not calibration.approach_gain_frozen
    assert calibration.strike_gain_scales[2] < 1.0
    assert calibration.follow_through_gain_scales[1] < 1.0
    assert calibration.joint_gain_scales[0] == 1.0
    assert load_g1_sonic_authority_calibration(output) == calibration
    composed = derive_g1_sonic_authority_calibration(
        trajectory_paths=(trajectory,),
        evidence_paths=(evidence,),
        output_path=tmp_path / "composed.json",
        source_checkout=tmp_path / "checkout",
        base_calibration_path=output,
    )
    assert composed.base_calibration_hash == calibration.calibration_hash
    assert composed.joint_gain_scales[4] <= calibration.joint_gain_scales[4]
    assert composed.strike_gain_scales[2] <= calibration.strike_gain_scales[2]
    frozen = derive_g1_sonic_authority_calibration(
        trajectory_paths=(trajectory,),
        evidence_paths=(evidence,),
        output_path=tmp_path / "frozen.json",
        source_checkout=tmp_path / "checkout",
        freeze_approach_gain=True,
        calibration_step_fraction=0.10,
    )
    assert frozen.approach_gain_frozen
    assert frozen.joint_gain_scales == (1.0,) * 29
    assert frozen.strike_gain_scales[2] < 1.0
    assert frozen.strike_gain_scales[2] > calibration.strike_gain_scales[2]
    assert frozen.calibration_step_fraction == 0.10
    value = json.loads(output.read_text(encoding="utf-8"))
    value["joint_gain_scales"][4] = 1.0
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_sonic_authority_calibration(output)


def test_contact_demand_trains_the_runtime_follow_through_authority(tmp_path: Path) -> None:
    trajectory = tmp_path / "contact-trajectory.npz"
    commanded = np.zeros((60, 29), dtype=np.float64)
    commanded[20:40, 1] = 160.0
    peak_abs = np.abs(commanded)
    peak_abs[20:40, 1] = 190.0
    np.savez_compressed(
        trajectory,
        commanded_torque=commanded,
        commanded_torque_peak_abs=peak_abs,
        event_phase=np.asarray((0,) * 20 + (5,) * 20 + (6,) * 20, dtype=np.int64),
    )
    evidence = tmp_path / "contact-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "trajectory_path": str(trajectory.resolve()),
                "trajectory_hash": _hash(trajectory),
                "strict_replay": True,
                "body_hash": "sha256:" + "1" * 64,
                "implementation_hash": "sha256:" + "2" * 64,
            }
        ),
        encoding="utf-8",
    )

    calibration = derive_g1_sonic_authority_calibration(
        trajectory_paths=(trajectory,),
        evidence_paths=(evidence,),
        output_path=tmp_path / "contact-calibration.json",
        source_checkout=tmp_path / "checkout",
    )

    assert calibration.strike_gain_scales[1] < 1.0
    assert calibration.follow_through_gain_scales[1] < 1.0
    assert calibration.follow_through_gain_scales[1] < 0.70
    assert calibration.schema_version.endswith(".v7")


def test_online_approach_strike_state_matches_frozen_contract() -> None:
    qpos = np.zeros(43, dtype=np.float64)
    qpos[2] = 0.78
    qpos[3] = 1.0
    qpos[36:39] = (1.0, 0.0, 0.115)
    qvel = np.zeros(41, dtype=np.float64)
    data = SimpleNamespace(
        qpos=qpos,
        qvel=qvel,
        xquat=np.asarray(((1.0, 0.0, 0.0, 0.0),), dtype=np.float64),
    )
    ids = SimpleNamespace(torso=0, ball_qpos=36, ball_qvel=35)

    state = build_online_approach_strike_state(
        data=data,
        ids=ids,
        target=np.zeros(29, dtype=np.float64),
        event_phase=2,
    )

    assert state.shape == (len(STATE_FEATURES),)
    assert state[74] == 1.0
    with pytest.raises(ValueError, match="residual fraction"):
        G1ApproachStrikeResidualConfig(residual_fraction=0.0)
