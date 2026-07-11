"""Tests for --dry-run feature preview."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.cli import main


@pytest.fixture
def rich_episode_dir() -> Path:
    return Path(__file__).parent.parent.parent / "examples" / "practice" / "rich_lerobot_episode"


def test_dry_run_safety_profile_no_files_written(capsys, rich_episode_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "dry_dataset"
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "lerobot",
            "export-dataset",
            "--episode",
            str(rich_episode_dir),
            "--output",
            str(out_dir),
            "--repo-id",
            "local/dry_run",
            "--profile",
            "safety",
            "--dry-run",
        ],
    ):
        rc = main()
    assert rc == 0
    assert not out_dir.exists() or not (out_dir / "meta").exists()
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "rosclaw.sandbox.decision" in out


def test_dry_run_rich_profile_json(capsys, rich_episode_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "dry_dataset"
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "practice",
            "export",
            "--format",
            "lerobot",
            "--writer",
            "real",
            "--episode",
            str(rich_episode_dir),
            "--output",
            str(out_dir),
            "--repo-id",
            "local/dry_run",
            "--profile",
            "safety-rich",
            "--dry-run",
            "--json",
        ],
    ):
        rc = main()
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "dry_run"
    assert "rosclaw.failure.code" in data["features"]
    assert "rosclaw.intervention.source" in data["features"]
