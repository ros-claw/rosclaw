from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_bilateral_foot_showcase_video import (
    render_g1_bilateral_foot_showcase_video,
)


def test_bilateral_video_refuses_source_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_bilateral_foot_showcase_video(
            evidence_path=tmp_path / "evidence.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path,
        )
