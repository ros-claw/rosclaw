from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_neural_torque_video import (
    _drawtext,
    render_g1_neural_torque_comparison_video,
)


def test_neural_torque_video_refuses_source_checkout_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the checkout"):
        render_g1_neural_torque_comparison_video(
            asset_root=tmp_path / "assets",
            stable_artifact_path=tmp_path / "stable.bin",
            parent_artifact_path=tmp_path / "parent.bin",
            candidate_artifact_path=tmp_path / "candidate.bin",
            output_path=tmp_path / "evidence" / "comparison.mp4",
            source_checkout=tmp_path,
        )


def test_neural_torque_video_drawtext_disables_expansion_and_escapes() -> None:
    value = _drawtext(
        "",
        "unsafe: label, 100%",
        x=1,
        y=2,
        size=3,
        color="white",
        enable="between(t,0,1)",
    )

    assert "expansion=none" in value
    assert "unsafe" in value
    assert "label" in value
    assert "100%" in value
    assert "unsafe: label, 100%" not in value
    assert "enable='between(t,0,1)'" in value
