"""Tests for the ROSClaw Practice -> LeRobot normalizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.practice_normalizer import (
    NORMALIZED_SCHEMA_VERSION,
    NormalizationError,
    normalize_practice_episode,
    write_normalized_episode,
)


@pytest.fixture
def minimal_episode_dir() -> Path:
    return Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"


def test_normalize_valid_episode(minimal_episode_dir: Path) -> None:
    episode = normalize_practice_episode(minimal_episode_dir)
    assert episode.schema_version == NORMALIZED_SCHEMA_VERSION
    assert len(episode.frames) == 3
    assert episode.frames[0].frame_index == 0
    assert episode.task.text == "transfer the cube"
    assert episode.fps == 10.0


def test_normalize_with_overrides(minimal_episode_dir: Path) -> None:
    episode = normalize_practice_episode(
        minimal_episode_dir,
        task="pick the cube",
        robot_id="test_bot",
        body_profile="test_body",
        fps=30.0,
    )
    assert episode.task.text == "pick the cube"
    assert episode.robot.robot_id == "test_bot"
    assert episode.robot.body_profile == "test_body"
    assert episode.fps == 30.0


def test_normalize_missing_episode(tmp_path: Path) -> None:
    with pytest.raises(NormalizationError) as exc_info:
        normalize_practice_episode(tmp_path / "missing")
    assert exc_info.value.code == "practice_episode_not_found"


def test_normalize_state_dim_mismatch(tmp_path: Path) -> None:
    episode_file = tmp_path / "episode.json"
    episode_file.write_text(
        '{"frames": [ '
        '{"frame_index": 0, "timestamp": 0.0, "observation": {"state": [0.0]}, "action": [0.0]},'
        '{"frame_index": 1, "timestamp": 0.1, "observation": {"state": [0.0, 0.0]}, "action": [0.0]}'
        ']}',
        encoding="utf-8",
    )
    with pytest.raises(NormalizationError) as exc_info:
        normalize_practice_episode(episode_file)
    assert exc_info.value.code == "state_dim_mismatch"


def test_normalize_action_dim_mismatch(tmp_path: Path) -> None:
    episode_file = tmp_path / "episode.json"
    episode_file.write_text(
        '{"frames": [ '
        '{"frame_index": 0, "timestamp": 0.0, "observation": {"state": [0.0]}, "action": [0.0]},'
        '{"frame_index": 1, "timestamp": 0.1, "observation": {"state": [0.0]}, "action": [0.0, 0.0]}'
        ']}',
        encoding="utf-8",
    )
    with pytest.raises(NormalizationError) as exc_info:
        normalize_practice_episode(episode_file)
    assert exc_info.value.code == "action_dim_mismatch"


def test_normalize_image_missing(tmp_path: Path) -> None:
    episode_file = tmp_path / "episode.json"
    episode_file.write_text(
        '{"frames": [{"frame_index": 0, "timestamp": 0.0, '
        '"observation": {"state": [0.0], "images": {"front": "frames/missing.png"}}, "action": [0.0]}]}',
        encoding="utf-8",
    )
    with pytest.raises(NormalizationError) as exc_info:
        normalize_practice_episode(episode_file)
    assert exc_info.value.code == "image_file_not_found"


def test_write_normalized_episode(tmp_path: Path, minimal_episode_dir: Path) -> None:
    episode = normalize_practice_episode(minimal_episode_dir)
    # Copy referenced frames next to the output JSON so it can be re-loaded.
    import shutil
    shutil.copytree(minimal_episode_dir / "frames", tmp_path / "frames")
    out = tmp_path / "normalized.json"
    path = write_normalized_episode(episode, out)
    assert path == out
    assert out.exists()
    loaded = normalize_practice_episode(out)
    assert len(loaded.frames) == len(episode.frames)
