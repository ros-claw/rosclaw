"""Tests for dataset export report v1.1 fields."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.dataset_report import (
    DATASET_EXPORT_SCHEMA_VERSION,
    DatasetExportReport,
    build_limitations_block,
    build_safety_block,
    report_from_worker_response,
)


def test_report_from_worker_response_populates_v1_1_blocks() -> None:
    response = {
        "status": "ok",
        "validation": {
            "load_ok": True,
            "index_ok": True,
            "dataloader_ok": True,
            "num_frames": 4,
            "num_episodes": 1,
        },
        "dataset": {
            "num_frames": 4,
            "num_episodes": 1,
            "fps": 10.0,
            "features": {"observation.state": {"shape": [6]}, "rosclaw.sandbox.decision": {"shape": [1]}},
            "visual": {"storage_mode": "images", "camera_keys": ["front"], "use_videos": False},
        },
        "runtime": {"python": "/usr/bin/python3.12", "lerobot_version": "0.6.1"},
        "api_info": {
            "create_signature": "(repo_id, fps, features, root, robot_type, use_videos)",
            "has_add_frame": True,
            "has_save_episode": True,
            "has_finalize": True,
            "has_consolidate": False,
            "lerobot_version": "0.6.1",
        },
        "feature_groups_written": ["safety"],
        "timing": {"write_time_sec": 0.5},
    }
    report = report_from_worker_response(
        episode_path="/tmp/ep",
        episode_id="ep_001",
        output_dir="/tmp/out",
        repo_id="local/rosclaw_test",
        response_dict=response,
    )
    assert report.schema_version == DATASET_EXPORT_SCHEMA_VERSION
    assert report.visual["storage_mode"] == "images"
    assert report.lerobot_dataset_api["has_add_frame"] is True
    assert report.quality_gates["dataloader_ok"] is True
    assert "rosclaw.sandbox.decision" in report.dataset["features"]
    assert report.feature_groups == ["safety"]


def test_legacy_v1_report_migration(tmp_path: Path) -> None:
    legacy = {
        "schema_version": "rosclaw.lerobot.dataset_export.v1",
        "status": "ok",
        "created_at": "2024-01-01T00:00:00Z",
        "source": {},
        "target": {},
        "dataset": {},
        "validation": {},
        "safety": {},
        "limitations": [],
        "timing": {},
    }
    report_path = tmp_path / "legacy_report.json"
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    report = DatasetExportReport.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
    assert report.schema_version == "rosclaw.lerobot.dataset_export.v1"
    assert report.visual == {}
    assert report.quality_gates == {}


def test_limitations_derived_from_features() -> None:
    limitations = build_limitations_block(["observation.state", "action", "rosclaw.sandbox.decision"])
    assert any("No sandbox" not in lim for lim in limitations)
    assert any("No intervention" in lim for lim in limitations)


def test_safety_block() -> None:
    block = build_safety_block()
    assert block["contains_executed_actions"] is True
    assert block["not_a_policy_execution"] is True
