"""Tests for the dataset worker runner using a fake LeRobot runtime worker."""

from __future__ import annotations

import sys
from pathlib import Path

from rosclaw.integrations.lerobot.dataset_worker_runner import LeRobotDatasetWorkerRunner
from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DatasetValidationConfig,
    DatasetWorkerRequest,
    DatasetWriterConfig,
)


def _patch_runner(monkeypatch, runner: LeRobotDatasetWorkerRunner, script: Path) -> None:
    monkeypatch.setattr(runner, "worker_script", script)
    monkeypatch.setattr(runner, "_resolve_runtime", lambda: (sys.executable, None))


def test_dataset_worker_runner_export_ok(
    tmp_path: Path, monkeypatch, fake_dataset_worker_script: Path
) -> None:
    runner = LeRobotDatasetWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_dataset_worker_script)

    normalized = tmp_path / "episode.json"
    normalized.write_text('{"frames": []}', encoding="utf-8")
    request = DatasetWorkerRequest(
        op="export_dataset",
        normalized_episode_path=str(normalized),
        output_dir=str(tmp_path / "dataset"),
        repo_id="local/rosclaw_test",
        fps=10.0,
        writer=DatasetWriterConfig(use_videos=False),
        validation=DatasetValidationConfig(sample_indices=[0]),
    )
    response = runner.run(request)

    assert response.ok
    assert response.op == "export_dataset"
    assert response.repo_id == "local/rosclaw_test"
    assert response.dataset.num_frames == 3
    assert response.validation.load_ok is True


def test_dataset_worker_runner_inspect_api(
    tmp_path: Path, monkeypatch, fake_dataset_worker_script: Path
) -> None:
    runner = LeRobotDatasetWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_dataset_worker_script)

    request = DatasetWorkerRequest(op="inspect_api", timeout_sec=10)
    response = runner.run(request)

    assert response.ok
    assert response.op == "inspect_api"


def test_dataset_worker_runner_missing_runtime(monkeypatch, tmp_path: Path) -> None:
    runner = LeRobotDatasetWorkerRunner(timeout_sec=10)
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_worker_runner.get_configured_lerobot_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_worker_runner.inspect_lerobot_runtime",
        lambda *_args, **_kwargs: _FakeRuntime(state="error"),
    )

    request = DatasetWorkerRequest(
        op="export_dataset",
        normalized_episode_path=str(tmp_path / "episode.json"),
        output_dir=str(tmp_path / "out"),
        repo_id="local/rosclaw_test",
    )
    response = runner.run(request)

    assert not response.ok
    assert response.error_code() == "runtime_not_configured"


class _FakeRuntime:
    def __init__(self, state: str) -> None:
        self.state = state
        self.lerobot_version = None
