"""End-to-end test exporting a physical-telemetry episode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.dataset_worker_runner import run_dataset_export

PHYSICAL_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "physical_lerobot_episode"


def _runtime_available() -> bool:
    runtime = get_configured_lerobot_runtime()
    return bool(runtime and runtime.get("subprocess_available"))


@pytest.mark.skipif(not _runtime_available(), reason="LeRobot runtime not available")
@pytest.mark.usefixtures("real_lerobot_runtime_config")
def test_physical_export_creates_telemetry_features_and_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "lerobot_physical"
    repo_id = "local/physical_test"
    response = run_dataset_export(
        normalized_episode_path=str(PHYSICAL_EPISODE / "episode.json"),
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=10.0,
        profile="physical",
        dataloader=True,
    )
    assert response.ok, response.error_message
    assert output_dir.exists()
    assert (output_dir / "meta" / "info.json").exists()

    feature_keys = set(response.dataset.features.keys())
    assert "observation.state" in feature_keys
    assert "action" in feature_keys
    assert "observation.images.front" in feature_keys
    assert "observation.motor_current" in feature_keys
    assert "observation.joint_temperature" in feature_keys
    assert "observation.force_torque" in feature_keys
    assert "observation.contact" in feature_keys
    assert "observation.joint_velocity" in feature_keys
    assert "observation.joint_effort" in feature_keys
    assert "rosclaw.sandbox.decision" in feature_keys
    assert "rosclaw.action.source" in feature_keys

    assert response.dataset.features["observation.motor_current"].shape == [6]
    assert response.dataset.features["observation.force_torque"].shape == [6]
    assert response.dataset.features["observation.contact"].shape == [6]

    rosclaw_meta = output_dir / "meta" / "rosclaw"
    assert (rosclaw_meta / "schema.json").exists()
    assert (rosclaw_meta / "vocab.json").exists()
    assert (rosclaw_meta / "episodes.parquet").exists() or (rosclaw_meta / "episodes.jsonl").exists()
    assert (rosclaw_meta / "events.parquet").exists() or (rosclaw_meta / "events.jsonl").exists()
    assert (rosclaw_meta / "sync_stats.parquet").exists() or (rosclaw_meta / "sync_stats.jsonl").exists()
    assert (rosclaw_meta / "units.json").exists()
    assert (rosclaw_meta / "feature_names.json").exists()

    units = json.loads((rosclaw_meta / "units.json").read_text(encoding="utf-8"))
    assert "observation.motor_current" in units["units"]
    assert units["units"]["observation.motor_current"]["unit"] == "A"

    feature_names = json.loads((rosclaw_meta / "feature_names.json").read_text(encoding="utf-8"))
    assert "observation.force_torque" in feature_names["features"]
    assert feature_names["features"]["observation.force_torque"]["axis_names"] == ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    assert response.validation.load_ok
    assert response.validation.index_ok
    assert response.validation.dataloader_ok


@pytest.mark.skipif(not _runtime_available(), reason="LeRobot runtime not available")
@pytest.mark.usefixtures("real_lerobot_runtime_config")
def test_physical_export_missing_policy_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "lerobot_physical_error"
    repo_id = "local/physical_error_test"

    # Create a minimal episode missing telemetry by using the physical fixture and
    # stripping one telemetry field from a frame.
    import shutil

    episode_dir = tmp_path / "physical_missing"
    shutil.copytree(PHYSICAL_EPISODE, episode_dir)
    data = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    del data["frames"][1]["observation"]["motor_current"]
    (episode_dir / "episode.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    response = run_dataset_export(
        normalized_episode_path=str(episode_dir / "episode.json"),
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=10.0,
        profile="physical",
        missing_policy="error",
    )
    assert not response.ok
    msg = response.error_message()
    assert "telemetry_missing" in msg or "missing" in msg.lower()
