from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.football_motion_prior import (
    G1FootballMotionEvent,
    G1FootballMotionPrior,
    blend_g1_football_motion_prior_target,
    load_g1_football_motion_prior,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_HASH = "sha256:" + "1" * 64


def _prior() -> G1FootballMotionPrior:
    event = G1FootballMotionEvent(
        relative_path="npz/soccer/train.npz",
        source_hash=_HASH,
        capture_id="train",
        contact_start_frame=100,
        contact_end_frame=105,
        reference_contact_frame=104,
        fps=90.0,
        score=4.0,
        outgoing_planar_speed_mps=4.0,
        outgoing_vertical_speed_mps=1.0,
        vertical_speed_delta_mps=0.9,
        right_foot_peak_speed_mps=3.0,
    )
    rows = (
        (0.20, -0.20, 0.0, 0.80, -0.20, 0.0),
        (0.10, -0.10, 0.0, 0.70, -0.10, 0.0),
        (0.00, 0.00, 0.0, 0.60, 0.00, 0.0),
    )
    return G1FootballMotionPrior(
        body_hash=_HASH,
        dataset_readme_hash=_HASH,
        split_manifest_hash=_HASH,
        joint_order_contract_hash=_HASH,
        train_partition_hash=_HASH,
        heldout_partition_commitment=_HASH,
        joint_names=G1_DDS_JOINT_NAMES[6:12],
        reference_times_sec=(-0.10, 0.0, 0.10),
        right_leg_reference_rad=rows,
        right_leg_iqr_rad=tuple((0.1,) * 6 for _ in rows),
        selected_events=(event,),
        train_files_considered=1,
        qualified_event_count=1,
    )


def test_motion_prior_round_trip_is_hash_bound(tmp_path: Path) -> None:
    prior = _prior()
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior.to_dict()), encoding="utf-8")

    loaded = load_g1_football_motion_prior(path)

    assert loaded.prior_hash == prior.prior_hash
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["right_leg_reference_rad"][1][0] = 0.4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_football_motion_prior(path)


def test_motion_prior_blend_is_windowed_and_support_bounded() -> None:
    target = np.full(29, 2.0, dtype=np.float64)
    prior = _prior()

    adapted, delta, active = blend_g1_football_motion_prior_target(
        target=target,
        prior=prior,
        policy_frame=265,
        contact_policy_frame=265,
        control_dt_sec=0.02,
        blend=0.50,
    )

    assert active
    assert np.allclose(delta[:6], 0.0)
    assert np.allclose(delta[12:], 0.0)
    assert np.max(np.abs(delta)) <= 0.225 + 1e-12
    assert np.allclose(adapted, target + delta)

    outside, outside_delta, outside_active = blend_g1_football_motion_prior_target(
        target=target,
        prior=prior,
        policy_frame=250,
        contact_policy_frame=265,
        control_dt_sec=0.02,
        blend=0.50,
    )
    assert not outside_active
    assert np.array_equal(outside, target)
    assert np.count_nonzero(outside_delta) == 0
