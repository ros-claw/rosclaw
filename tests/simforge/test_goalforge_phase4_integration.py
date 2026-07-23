from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.robot_pack.g1.dds_adapter import run_unitree_dds_loopback
from rosclaw.simforge.g1_video import render_goalforge_video
from rosclaw.simforge.phase4_run import run_goalforge_demo

pytestmark = pytest.mark.integration


def test_real_g1_goalforge_failure_to_success(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    result = run_goalforge_demo(
        asset_root=Path(asset_root),
        output_dir=tmp_path / "demo",
        source_checkout=Path(__file__).resolve().parents[2],
    )
    assert result.passed
    video = render_goalforge_video(
        demo_path=result.report_path,
        asset_root=Path(asset_root),
        output_path=tmp_path / "goalforge.mp4",
        source_checkout=Path(__file__).resolve().parents[2],
        fps=20,
        width=640,
        height=360,
    )
    assert video.output_path.stat().st_size > 100_000
    assert video.frame_count > 0
    assert video.video_hash.startswith("sha256:")
    assert video.manifest_path.is_file()


def test_official_unitree_dds_loopback(tmp_path: Path) -> None:
    root = os.environ.get("ROSCLAW_UNITREE_MUJOCO_ROOT")
    if not root:
        pytest.skip("ROSCLAW_UNITREE_MUJOCO_ROOT is not configured")
    result = run_unitree_dds_loopback(
        unitree_mujoco_root=Path(root),
        output_dir=tmp_path / "dds",
        source_checkout=Path(__file__).resolve().parents[2],
        domain_id=79,
    )
    assert result.passed
