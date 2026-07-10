"""Tests for dataset validation and export report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.dataset_report import (
    DATASET_EXPORT_SCHEMA_VERSION,
    DatasetExportReport,
    get_dataset_export_validation_status,
    read_latest_dataset_export_report,
    report_from_worker_response,
    write_dataset_export_report,
)
from rosclaw.integrations.lerobot.dataset_validator import (
    format_validation_result,
    validate_dataset,
)
from rosclaw.integrations.lerobot.dataset_worker_schema import (
    DatasetValidationResult,
    DatasetWorkerResponse,
)


def test_validate_dataset_delegates_to_worker(monkeypatch, tmp_path: Path) -> None:
    response = DatasetWorkerResponse(
        status="ok",
        op="validate_dataset",
        output_dir=str(tmp_path),
        repo_id="local/rosclaw_test",
        validation=DatasetValidationResult(
            load_ok=True, index_ok=True, num_frames=3, num_episodes=1
        ),
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.dataset_validator.LeRobotDatasetWorkerRunner.run",
        lambda *args, **kwargs: response,
    )
    result = validate_dataset(tmp_path, "local/rosclaw_test")
    assert result.load_ok is True
    assert result.index_ok is True
    assert result.num_frames == 3


def test_format_validation_result() -> None:
    result = DatasetValidationResult(
        load_ok=True,
        index_ok=True,
        num_frames=3,
        num_episodes=1,
        sample_keys=["action", "observation.state"],
    )
    formatted = format_validation_result(result)
    assert formatted["load_ok"] is True
    assert formatted["sample_keys"] == ["action", "observation.state"]


def test_report_from_worker_response_ok() -> None:
    response = {
        "status": "ok",
        "validation": {"load_ok": True, "index_ok": True, "num_frames": 3},
        "dataset": {"num_frames": 3, "fps": 10.0},
        "timing": {"write_time_sec": 0.5},
        "runtime": {"python_version": "0.6.1"},
    }
    report = report_from_worker_response(
        episode_path="/tmp/ep",
        episode_id="ep_001",
        output_dir="/tmp/out",
        repo_id="local/rosclaw_test",
        response_dict=response,
    )
    assert report.status == "ok"
    assert report.schema_version == DATASET_EXPORT_SCHEMA_VERSION
    assert report.validation["load_ok"] is True
    assert report.target["repo_id"] == "local/rosclaw_test"
    assert report.runtime["python_version"] == "0.6.1"


def test_write_and_read_report(tmp_path: Path) -> None:
    report = DatasetExportReport(status="ok")
    report_path = write_dataset_export_report(report, tmp_path)
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == DATASET_EXPORT_SCHEMA_VERSION
    latest = read_latest_dataset_export_report()
    assert latest is not None
    assert latest.status == "ok"


def test_dataset_export_validation_status(tmp_path: Path, monkeypatch) -> None:
    response = {
        "status": "ok",
        "validation": {"load_ok": True, "index_ok": True, "num_frames": 3},
        "dataset": {"num_frames": 3, "features": {"observation.state": {"shape": [2]}}},
        "runtime": {"lerobot_version": "0.6.1", "python": "/usr/bin/python3.12"},
    }
    report = report_from_worker_response(
        episode_path="/tmp/ep",
        episode_id="ep_001",
        output_dir=str(tmp_path),
        repo_id="local/rosclaw_test",
        response_dict=response,
    )
    write_dataset_export_report(report, tmp_path)
    (tmp_path / "meta" / "info.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    status = get_dataset_export_validation_status(
        current_lerobot_version="0.6.1",
        current_python_executable="/usr/bin/python3.12",
    )
    assert status["state"] == "validated"
    assert status["num_frames"] == 3
