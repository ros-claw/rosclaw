from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.simforge.g1_self_aware_showcase import (
    run_g1_self_aware_showcase,
    self_aware_showcase_specs,
)
from rosclaw.simforge.g1_self_aware_showcase_video import (
    render_g1_self_aware_showcase_video,
)


def test_self_aware_showcase_defines_three_moving_ball_challenges() -> None:
    specs = self_aware_showcase_specs()

    assert len(specs) == 3
    assert len({spec.case_id for spec in specs}) == 3
    assert all(spec.scenario.generation == 9 for spec in specs)
    assert all(spec.scenario.ball_velocity_x_mps < 0.0 for spec in specs)
    assert {spec.scenario.disturbance_n for spec in specs} == {30.0, 32.0}
    assert {spec.scenario.support_ground_friction for spec in specs} == {0.78, 0.82}
    assert len({spec.scenario.scenario_commitment for spec in specs}) == 3


def test_self_aware_showcase_evidence_must_be_external(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_self_aware_showcase(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


def test_self_aware_showcase_video_must_be_external(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_self_aware_showcase_video(
            showcase_evidence_path=tmp_path / "showcase.json",
            rejected_v2_evidence_path=tmp_path / "v2.json",
            self_aware_v3_evidence_path=tmp_path / "v3.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "showcase.mp4",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_real_self_aware_showcase_is_strict_and_safe(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")

    evidence = run_g1_self_aware_showcase(
        asset_root=Path(asset_root),
        output_dir=tmp_path / "self-aware-showcase",
        source_checkout=Path(__file__).resolve().parents[2],
    )

    assert evidence.passed
    assert all(case.strict_replay for case in evidence.cases)
    assert all(case.result["success"] for case in evidence.cases)
    assert all(not case.result["post_kick_fall"] for case in evidence.cases)
    assert all(not case.result["joint_limit_violation"] for case in evidence.cases)
    assert all(case.result["physics_executed"] for case in evidence.cases)
