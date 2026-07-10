"""Tests for the LeRobot dataset export CLI paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.cli import main
from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DatasetInfo,
    DatasetValidationResult,
    DatasetWorkerResponse,
)


@pytest.fixture
def minimal_episode_dir() -> Path:
    return Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"


def _fake_runner_response(*args, **kwargs):  # noqa: ARG001
    return DatasetWorkerResponse(
        status="ok",
        op="export_dataset",
        output_dir="/tmp/out",
        repo_id="local/rosclaw_test",
        dataset=DatasetInfo(num_episodes=1, num_frames=3, fps=10.0),
        validation=DatasetValidationResult(load_ok=True, index_ok=True),
    )


def test_lerobot_export_dataset_cli(
    capsys, monkeypatch, minimal_episode_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.cli.LeRobotDatasetWorkerRunner.run",
        _fake_runner_response,
    )
    out_dir = tmp_path / "dataset"
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "lerobot",
            "export-dataset",
            "--episode",
            str(minimal_episode_dir),
            "--output",
            str(out_dir),
            "--repo-id",
            "local/rosclaw_test",
            "--fps",
            "10",
        ],
    ):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset export complete" in out
    assert out_dir.exists()
    report = out_dir / "rosclaw_export_report.json"
    assert report.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["schema_version"] == "rosclaw.lerobot.dataset_export.v1.1"
    assert data["status"] == "ok"


def test_practice_export_writer_real(
    capsys, monkeypatch, minimal_episode_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.cli.LeRobotDatasetWorkerRunner.run",
        _fake_runner_response,
    )
    out_dir = tmp_path / "dataset"
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
            str(minimal_episode_dir),
            "--output",
            str(out_dir),
            "--repo-id",
            "local/rosclaw_test",
        ],
    ):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset export complete" in out


def test_practice_export_writer_skeleton(
    capsys, monkeypatch, minimal_episode_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "skeleton"
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
            "skeleton",
            "--episode",
            str(minimal_episode_dir),
            "--output",
            str(out_dir),
        ],
    ):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "skeleton" in out.lower()
    assert (out_dir / "meta" / "info.json").exists()


def test_lerobot_validate_dataset_cli(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_validator.LeRobotDatasetWorkerRunner.run",
        _fake_runner_response,
    )
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "lerobot",
            "validate-dataset",
            "--dataset",
            str(tmp_path / "dataset"),
            "--repo-id",
            "local/rosclaw_test",
        ],
    ):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset validation" in out
