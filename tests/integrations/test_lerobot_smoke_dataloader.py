"""DataLoader smoke test against a real exported dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.dataset_validator import run_dataloader_smoke
from rosclaw.integrations.lerobot.dataset_worker_runner import run_dataset_export

MINIMAL_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"


def _runtime_available() -> bool:
    runtime = get_configured_lerobot_runtime()
    return bool(runtime and runtime.get("subprocess_available"))


@pytest.mark.skipif(not _runtime_available(), reason="LeRobot runtime not available")
@pytest.mark.usefixtures("real_lerobot_runtime_config")
def test_dataloader_smoke_on_minimal_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "lerobot_minimal"
    repo_id = "local/dataloader_test"
    response = run_dataset_export(
        normalized_episode_path=str(MINIMAL_EPISODE / "episode.json"),
        output_dir=str(output_dir),
        repo_id=repo_id,
        fps=10.0,
        profile="minimal",
    )
    assert response.ok, response.error_message

    result = run_dataloader_smoke(output_dir, repo_id, batch_size=2, num_workers=0)
    assert result.dataloader_ok is True
    assert "action" in result.batch_keys
    assert "observation.state" in result.batch_keys
