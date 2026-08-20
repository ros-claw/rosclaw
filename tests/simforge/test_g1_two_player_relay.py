from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.simforge.g1_two_player_relay import (
    _measure_handoff,
    run_g1_two_player_relay,
)


def test_relay_handoff_uses_measured_velocity_with_180_degree_frame_rotation() -> None:
    trajectory = {
        "time": np.asarray([5.60, 5.70, 6.90]),
        "ball_pose": np.asarray(
            [
                [1.20, -0.16, 0.115, 1.0, 0.0, 0.0, 0.0],
                [1.35, -0.16, 0.115, 1.0, 0.0, 0.0, 0.0],
                [2.49, -0.16, 0.115, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "ball_velocity": np.asarray(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.594, -0.0012, 0.0]]
        ),
    }
    episode = SimpleNamespace(
        result=SimpleNamespace(ball_contact_time_sec=5.60),
        trajectory=trajectory,
    )

    handoff = _measure_handoff(episode)

    assert handoff.source_sample_index == 2
    assert handoff.observed_speed_mps == pytest.approx(np.hypot(0.594, 0.0012))
    assert handoff.passer_local_velocity_mps == pytest.approx((0.594, -0.0012))
    assert handoff.shooter_local_velocity_mps == pytest.approx((-0.594, 0.0012))
    travel_time = 0.25 / 0.594
    assert handoff.shooter_launch_delay_sec == pytest.approx(5.40 - travel_time)
    assert handoff.shooter_ball_y_m == pytest.approx(-0.0012 * travel_time)
    assert handoff.handoff_hash.startswith("sha256:")


def test_relay_handoff_fails_closed_without_valid_receiver_speed() -> None:
    episode = SimpleNamespace(
        result=SimpleNamespace(ball_contact_time_sec=5.0),
        trajectory={
            "time": np.asarray([5.0, 5.2]),
            "ball_pose": np.asarray(
                [
                    [1.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0],
                    [1.1, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            "ball_velocity": np.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        },
    )

    with pytest.raises(RuntimeError, match="validated receiver-speed envelope"):
        _measure_handoff(episode)


def test_relay_evidence_cannot_be_written_inside_checkout(tmp_path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_two_player_relay(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_real_g1_two_player_relay_is_strict_and_high_targeted(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")

    report = run_g1_two_player_relay(
        asset_root=Path(asset_root),
        output_dir=tmp_path / "relay",
        source_checkout=Path(__file__).resolve().parents[2],
    )

    assert report.passed
    assert report.passer.strict_replay
    assert report.shooter.strict_replay
    assert report.shooter.scenario["target_z_m"] == pytest.approx(0.70)
    assert report.shooter.result["target_error_m"] <= 0.25
    assert report.handoff.observed_speed_mps >= 0.55
    assert report.recovery_evolution.decision == "SIM_CHAMPION"
    assert report.recovery_evolution.candidate_config_hash == (
        report.shooter.recovery_receipt["config_hash"]
    )
