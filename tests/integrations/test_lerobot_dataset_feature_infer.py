"""Tests for feature inference from normalized practice episodes."""

from __future__ import annotations

from pathlib import Path

from rosclaw.integrations.lerobot.dataset_feature_infer import (
    FeatureInferenceError,
    feature_summary,
    infer_features,
)
from rosclaw.integrations.lerobot.practice_normalizer import (
    NormalizedPracticeEpisode,
    normalize_practice_episode,
)


MINIMAL_EPISODE = Path(__file__).parent.parent.parent / "examples" / "practice" / "minimal_lerobot_episode"


def test_infer_features_state_action_image() -> None:
    episode = normalize_practice_episode(MINIMAL_EPISODE)
    features = infer_features(episode)
    assert "observation.state" in features
    assert features["observation.state"]["dtype"] == "float32"
    assert features["observation.state"]["shape"] == [3]
    assert "action" in features
    assert features["action"]["shape"] == [3]
    assert "observation.images.front" in features
    assert features["observation.images.front"]["dtype"] == "image"
    assert features["observation.images.front"]["shape"] == [48, 64, 3]


def test_feature_summary() -> None:
    episode = normalize_practice_episode(MINIMAL_EPISODE)
    features = infer_features(episode)
    summary = feature_summary(features)
    assert summary["observation.state"] == [3]
    assert summary["action"] == [3]
    assert summary["observation.images.front"] == [48, 64, 3]


def test_infer_features_empty_episode() -> None:
    episode = NormalizedPracticeEpisode()
    try:
        infer_features(episode)
        raise AssertionError("Expected FeatureInferenceError")
    except FeatureInferenceError as exc:
        assert exc.code == "normalized_episode_invalid"
