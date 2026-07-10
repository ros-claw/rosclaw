"""Tests that dataset export fails gracefully when no LeRobot runtime is configured."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.cli import main


@pytest.fixture
def minimal_episode_dir() -> Path:
    return Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"


def test_export_dataset_no_runtime(
    capsys, monkeypatch, minimal_episode_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_worker_runner.get_configured_lerobot_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_worker_runner.inspect_lerobot_runtime",
        lambda *_args, **_kwargs: _FakeRuntime(state="error"),
    )

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
            str(tmp_path / "out"),
            "--repo-id",
            "local/rosclaw_test",
        ],
    ):
        rc = main()

    assert rc == 1
    out = capsys.readouterr().out
    err = capsys.readouterr().err
    combined = (out + err).lower()
    assert "runtime" in combined or "lerobot" in combined


class _FakeRuntime:
    def __init__(self, state: str) -> None:
        self.state = state
        self.lerobot_version = None
