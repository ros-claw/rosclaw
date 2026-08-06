from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.simforge.g1_structured_recovery_video import (
    render_g1_structured_recovery_video,
)


def test_structured_video_cannot_write_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_structured_recovery_video(
            evidence_path=tmp_path / "evaluation.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path,
        )


def test_structured_video_rejects_failed_campaign(tmp_path: Path) -> None:
    evidence = tmp_path / "evaluation.json"
    evidence.write_text(
        json.dumps({"passed": False, "status": "REJECTED_BY_SIM_GATE"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="passing SIM campaign"):
        render_g1_structured_recovery_video(
            evidence_path=evidence,
            asset_root=tmp_path / "assets",
            output_path=tmp_path.parent / "video.mp4",
            source_checkout=tmp_path,
        )
