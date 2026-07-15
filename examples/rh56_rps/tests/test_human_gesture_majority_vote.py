"""Tests for human gesture majority vote."""
from __future__ import annotations

from rosclaw_rps.gesture_schema import GesturePrediction
from rosclaw_rps.vision.majority_vote import MajorityVoteBuffer


def test_majority_vote_stable():
    vote = MajorityVoteBuffer(window_size=10, min_confidence=0.6, majority_ratio=0.6)
    for _ in range(7):
        vote.update(GesturePrediction(label="scissors", confidence=0.9))
    for _ in range(3):
        vote.update(GesturePrediction(label="paper", confidence=0.9))
    result = vote.final()
    assert result.label == "scissors"
    assert result.confidence > 0.8


def test_majority_vote_rejects_low_ratio():
    vote = MajorityVoteBuffer(window_size=10, min_confidence=0.6, majority_ratio=0.7)
    for _ in range(5):
        vote.update(GesturePrediction(label="rock", confidence=0.9))
    for _ in range(5):
        vote.update(GesturePrediction(label="paper", confidence=0.9))
    result = vote.final()
    assert result.label == "unknown"


def test_majority_vote_rejects_low_confidence():
    vote = MajorityVoteBuffer(window_size=10, min_confidence=0.8)
    for _ in range(10):
        vote.update(GesturePrediction(label="rock", confidence=0.5))
    result = vote.final()
    assert result.label == "unknown"


def test_majority_vote_ignores_unknown():
    vote = MajorityVoteBuffer(window_size=10, min_confidence=0.6, majority_ratio=0.6)
    for _ in range(3):
        vote.update(GesturePrediction(label="unknown", confidence=0.0))
    for _ in range(7):
        vote.update(GesturePrediction(label="paper", confidence=0.9))
    result = vote.final()
    assert result.label == "paper"
