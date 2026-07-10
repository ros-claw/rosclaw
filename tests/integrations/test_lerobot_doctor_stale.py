"""Tests for dataset export doctor stale states."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.dataset_report import (
    DatasetExportReport,
    get_dataset_export_validation_status,
    write_dataset_export_report,
)


def test_missing_output_dir_is_stale(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing_dataset"
    report = DatasetExportReport(
        status="ok",
        target={"repo_id": "local/test", "output_dir": str(output_dir)},
        dataset={"num_frames": 3, "num_episodes": 1},
        validation={"load_ok": True, "index_ok": True},
    )
    write_dataset_export_report(report, tmp_path)
    status = get_dataset_export_validation_status()
    assert status["state"] == "stale"
    assert any("missing" in reason.lower() for reason in status.get("stale_reasons", []))


def test_old_schema_report_is_stale(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    (output_dir / "meta" / "info.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
    report = DatasetExportReport(
        schema_version="rosclaw.lerobot.dataset_export.v1",
        status="ok",
        target={"repo_id": "local/test", "output_dir": str(output_dir)},
        dataset={"num_frames": 3, "num_episodes": 1},
        validation={"load_ok": True, "index_ok": True},
    )
    write_dataset_export_report(report, tmp_path)
    status = get_dataset_export_validation_status()
    assert status["state"] == "stale"
    assert any("schema" in reason.lower() for reason in status.get("stale_reasons", []))
