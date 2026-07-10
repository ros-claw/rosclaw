"""End-to-end test exporting a rich episode with ROSClaw safety metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.dataset_worker_runner import run_dataset_export


RICH_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "rich_lerobot_episode"


def _runtime_available() -> bool:
    runtime = get_configured_lerobot_runtime()
    return bool(runtime and runtime.get("subprocess_available"))


@pytest.mark.skipif(not _runtime_available(), reason="LeRobot runtime not available")
@pytest.mark.usefixtures("real_lerobot_runtime_config")
def test_rich_export_creates_rosclaw_features_and_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "lerobot_rich"
    repo_id = "local/rich_safety_test"
    response = run_dataset_export(
        normalized_episode_path=str(RICH_EPISODE / "episode.json"),
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=10.0,
        profile="safety-rich",
        include_body_snapshot=True,
        body_snapshot_mode="sanitized",
        dataloader=True,
    )
    assert response.ok, response.error_message
    assert output_dir.exists()
    assert (output_dir / "meta" / "info.json").exists()

    # Sidecars
    assert (output_dir / "meta" / "rosclaw" / "schema.json").exists()
    assert (output_dir / "meta" / "rosclaw" / "vocab.json").exists()
    sidecar = output_dir / "meta" / "rosclaw" / "episodes.parquet"
    jsonl = output_dir / "meta" / "rosclaw" / "episodes.jsonl"
    assert sidecar.exists() or jsonl.exists()
    events_parquet = output_dir / "meta" / "rosclaw" / "events.parquet"
    events_jsonl = output_dir / "meta" / "rosclaw" / "events.jsonl"
    assert events_parquet.exists() or events_jsonl.exists()

    # Body snapshot
    assert (output_dir / "meta" / "rosclaw" / "body_snapshots" / "manifest.json").exists()
    body_yaml = output_dir / "meta" / "rosclaw" / "body_snapshots" / "body.yaml"
    assert body_yaml.exists()
    text = body_yaml.read_text(encoding="utf-8")
    assert "SECRET_SERIAL_12345" not in text
    manifest = json.loads((output_dir / "meta" / "rosclaw" / "body_snapshots" / "manifest.json").read_text(encoding="utf-8"))
    assert "source_sha256" in manifest["files"]["body.yaml"]
    assert "sanitized_sha256" in manifest["files"]["body.yaml"]
    assert manifest["files"]["body.yaml"]["redaction_count"] >= 0
