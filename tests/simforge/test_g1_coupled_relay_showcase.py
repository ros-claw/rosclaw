from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.simforge.g1_coupled_relay_showcase import (
    run_g1_coupled_showcase,
    showcase_specs,
)
from rosclaw.simforge.g1_coupled_showcase_video import (
    render_g1_coupled_showcase_video,
)


def test_showcase_defines_five_distinct_physics_challenges() -> None:
    specs = showcase_specs()

    assert len(specs) == 5
    assert len({spec.case_id for spec in specs}) == 5
    assert {spec.shooter_start_sec for spec in specs} == {1.98, 2.02, 2.06}
    assert {spec.ball_ground_friction for spec in specs} == {0.05, 0.10, 0.15}
    assert len({spec.camera_azimuth_deg for spec in specs}) == 5


def test_showcase_evidence_cannot_be_written_inside_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_coupled_showcase(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


def test_showcase_video_cannot_be_written_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_coupled_showcase_video(
            evidence_path=tmp_path / "evidence.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "showcase.mp4",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_real_five_challenge_showcase_is_strict_and_stable(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")

    evidence = run_g1_coupled_showcase(
        asset_root=Path(asset_root),
        output_dir=tmp_path / "coupled-showcase",
        source_checkout=Path(__file__).resolve().parents[2],
    )

    assert evidence.passed
    assert len(evidence.cases) == 5
    assert all(case.strict_replay for case in evidence.cases)
    assert all(case.result.passed for case in evidence.cases)
    assert all(case.result.goal_crossing_z_m >= 0.95 for case in evidence.cases)
    assert all(case.result.target_error_m <= 0.25 for case in evidence.cases)
    assert evidence.cases[0].result.receiver_phase_hold_frames == 2
    assert evidence.cases[-1].result.receiver_phase_advance_frames == 2
    assert all(
        min(
            case.result.passer_min_pelvis_height_m,
            case.result.shooter_min_pelvis_height_m,
        )
        >= 0.65
        for case in evidence.cases
    )
