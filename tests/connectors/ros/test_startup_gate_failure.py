"""Startup log evidence is diagnostic and cannot grant readiness or stop."""

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path


def test_observations_are_preserved_when_only_coverage_bringup_marker_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("startup_gate_failure")
    sample = {
        "captured_at": datetime.now(UTC).isoformat(),
        "observation_complete": True,
        "collision_count": 0,
        "cleaning_enabled": False,
    }
    (tmp_path / "witness.jsonl").write_text(json.dumps(sample) + "\n")
    (tmp_path / "measured_map.json").write_text("{}")
    (tmp_path / "nav2.log").write_text("Managed nodes are active")
    (tmp_path / "coverage_lifecycle.log").write_text("Configuring coverage_server")
    report = module.retain_startup_failure(tmp_path)
    assert report["missing_requirements"] == ["startup_completion_marker:coverage_lifecycle.log"]
    assert report["original_last_completed_witness"] == sample
    assert report["root_cause"] == "UNKNOWN" and report["readiness"] is False
    assert report["physical_stop_proof"] == "NOT_MEASURED"
    assert (
        report["original_log_snapshots"]["coverage_lifecycle.log"]["actual_live_lifecycle_state"]
        == "NOT_MEASURED"
    )
