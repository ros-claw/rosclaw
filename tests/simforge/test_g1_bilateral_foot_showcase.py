from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_bilateral_foot_showcase import (
    bilateral_candidates,
    run_g1_bilateral_foot_showcase,
)


def test_bilateral_candidates_are_actual_opposite_feet_and_corners() -> None:
    candidates = bilateral_candidates()

    assert {case[0] for case in candidates} == {"left", "right"}
    assert {case[2] for case in candidates} == {-1.0, 1.0}
    assert all(abs(case[1]) <= 0.30 for case in candidates)
    assert all(0.115 <= case[3] <= 0.20 for case in candidates)


def test_bilateral_evidence_refuses_source_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_bilateral_foot_showcase(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )
