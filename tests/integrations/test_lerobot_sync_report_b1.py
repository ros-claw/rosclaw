"""Tests for Gate B.1 synchronization report blocks and sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.clock_mapping import build_clock_mappings
from rosclaw.integrations.lerobot.dataset_report import (
    DATASET_EXPORT_SCHEMA_VERSION,
    DatasetExportReport,
)
from rosclaw.integrations.lerobot.source_stream_reader import read_source_bundle
from rosclaw.integrations.lerobot.sync_config import default_sync_config
from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality
from rosclaw.integrations.lerobot.sync_report import (
    build_missingness_report_block,
    build_synchronization_report_block,
    build_sync_stats_rows,
    write_clock_mappings_sidecar,
    write_sync_config_sidecar,
)
from rosclaw.integrations.lerobot.sync_stats import compute_sync_stats, write_sync_stats_parquet
from rosclaw.integrations.lerobot.synchronize import synchronize_bundle


ASYNC_FIXTURE = Path(__file__).parent.parent.parent / "examples" / "practice" / "async_physical_episode"


def test_dataset_export_report_v1_2_schema() -> None:
    report = DatasetExportReport(status="ok")
    assert report.schema_version == DATASET_EXPORT_SCHEMA_VERSION
    assert "v1.2" in report.schema_version
    data = report.to_dict()
    assert "synchronization" in data
    assert "missingness" in data


def test_dataset_export_report_v1_1_migration() -> None:
    legacy = {
        "schema_version": "rosclaw.lerobot.dataset_export.v1.1",
        "status": "ok",
        "source": {},
        "target": {},
        "dataset": {},
        "validation": {},
        "safety": {},
        "limitations": [],
        "timing": {},
    }
    report = DatasetExportReport.from_dict(legacy)
    assert report.schema_version == "rosclaw.lerobot.dataset_export.v1.1"
    assert report.synchronization == {}
    assert report.missingness == {}


def test_synchronization_report_block() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features, bundle.streams, config, mapping, result.timeline
    )
    quality = evaluate_quality(stats, mapping, default_quality_config("balanced"))

    block = build_synchronization_report_block(
        result, mapping, quality, input_timing_mode="source_streams", quality_profile="balanced"
    )
    assert block["input_mode"] == "source_streams"
    assert block["level"] == "validated"
    assert block["resampled"] is True
    assert block["target_fps"] == 10.0
    assert block["canonical_frames"] == 11
    assert block["clock_mappings_valid"] is True
    assert block["quality_profile"] == "balanced"
    assert block["quality_passed"] is True


def test_missingness_report_block() -> None:
    block = build_missingness_report_block("nan")
    assert block["policy"] == "nan"
    assert block["unknown_float_encoding"] == "NaN"
    assert block["unknown_bool_encoding"] == -1
    assert block["unknown_category_encoding"] == 0


def test_clock_mappings_sidecar(tmp_path: Path) -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    path = tmp_path / "clock_mappings.json"
    write_clock_mappings_sidecar(mapping, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "rosclaw.clock_mappings.v1"
    assert data["target_clock"] == "episode_time"
    assert "mappings" in data


def test_sync_config_sidecar(tmp_path: Path) -> None:
    config = default_sync_config(fps=10.0)
    path = tmp_path / "sync_config.json"
    write_sync_config_sidecar(config, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "rosclaw.sync_config.v1"
    assert data["target_fps"] == 10.0
    assert "observation.state" in data["policies"]


def test_sync_stats_rows_and_parquet(tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("pandas")
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features, bundle.streams, config, mapping, result.timeline
    )
    rows = build_sync_stats_rows(stats)
    assert len(rows) == len(stats)
    assert rows[0]["feature_key"] == stats[0].feature_key

    parquet_path = tmp_path / "sync_stats.parquet"
    written = write_sync_stats_parquet(stats, parquet_path)
    assert written.exists()
