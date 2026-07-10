"""Tests for NormalizedPracticeEpisode v2 schema and v1 migration."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.practice_normalizer import (
    NORMALIZED_SCHEMA_VERSION,
    NormalizationError,
    NormalizedActionContext,
    NormalizedFailure,
    NormalizedFrame,
    NormalizedIntervention,
    NormalizedPracticeEpisode,
    NormalizedRobot,
    NormalizedSafety,
    normalize_practice_episode,
    write_normalized_episode,
)


MINIMAL_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"
RICH_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "rich_lerobot_episode"


def test_v2_round_trip() -> None:
    episode = normalize_practice_episode(RICH_EPISODE)
    assert episode.schema_version == NORMALIZED_SCHEMA_VERSION
    assert len(episode.frames) >= 3
    frame = episode.frames[1]
    assert frame.safety is not None
    assert frame.safety.decision == "BLOCK"
    assert frame.intervention is not None
    assert frame.intervention.active is True
    assert frame.failure is not None
    assert frame.action_context is not None

    data = episode.to_dict()
    loaded = NormalizedPracticeEpisode.from_dict(data)
    assert loaded.frames[1].safety.decision == "BLOCK"
    assert loaded.frames[1].intervention.source == "HUMAN_JOYSTICK"


def test_v1_migration_defaults(tmp_path: Path) -> None:
    episode = normalize_practice_episode(MINIMAL_EPISODE)
    assert episode.schema_version == NORMALIZED_SCHEMA_VERSION
    frame = episode.frames[0]
    assert frame.safety is not None
    assert frame.safety.decision == "UNKNOWN"
    assert frame.safety.modified is None
    assert frame.failure is not None
    assert frame.failure.active is None
    assert frame.intervention is not None
    assert frame.intervention.active is None
    assert frame.action_context is not None
    assert frame.action_context.source == "UNKNOWN"
    assert frame.action_context.was_clamped is None
    # The minimal v1 fixture explicitly sets done/success; they are preserved.
    assert frame.done is False
    assert frame.success is False


def test_unknown_done_success_when_absent() -> None:
    frame = NormalizedFrame.from_dict({
        "frame_index": 0,
        "timestamp": 0.0,
        "observation": {"state": [0.0], "images": {}},
        "action": [0.0],
    })
    assert frame.done is None
    assert frame.success is None
    data = frame.to_dict()
    assert "done" not in data
    assert "success" not in data


def test_default_safety_values() -> None:
    safety = NormalizedSafety()
    assert safety.decision == "UNKNOWN"
    assert safety.modified is None
    assert safety.risk_score is None


def test_default_failure_values() -> None:
    failure = NormalizedFailure()
    assert failure.active is None
    assert failure.code is None
    assert failure.severity == 0


def test_default_intervention_values() -> None:
    intervention = NormalizedIntervention()
    assert intervention.active is None
    assert intervention.source is None
    assert intervention.confidence is None


def test_default_action_context_values() -> None:
    ctx = NormalizedActionContext()
    assert ctx.source == "UNKNOWN"
    assert ctx.was_clamped is None


def test_robot_fields_round_trip() -> None:
    robot = NormalizedRobot(
        robot_id="r1",
        body_profile="bp",
        body_yaml_path="/tmp/body.yaml",
        body_hash="abc",
        eurdf_repo="repo",
        eurdf_revision="rev",
    )
    data = robot.to_dict()
    loaded = NormalizedRobot.from_dict(data)
    assert loaded.body_hash == "abc"
    assert loaded.eurdf_repo == "repo"
