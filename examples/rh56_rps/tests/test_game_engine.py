"""Tests for RPS game rules and fairness."""
from __future__ import annotations

import pytest

from rosclaw_rps.game_engine import judge, CommitReveal, GameEngine, RPS_BEATS
from rosclaw_rps.gesture_schema import GesturePrediction, GestureExecutionResult


@pytest.mark.parametrize(
    "robot,human,expected",
    [
        ("rock", "rock", "draw"),
        ("rock", "scissors", "robot_win"),
        ("rock", "paper", "robot_lose"),
        ("paper", "rock", "robot_win"),
        ("paper", "paper", "draw"),
        ("paper", "scissors", "robot_lose"),
        ("scissors", "paper", "robot_win"),
        ("scissors", "scissors", "draw"),
        ("scissors", "rock", "robot_lose"),
    ],
)
def test_rps_rules_all_9_cases(robot, human, expected):
    assert judge(robot, human) == expected


def test_commit_verify_roundtrip():
    commit = CommitReveal.create("paper", "round_0001")
    assert commit.verify("round_0001")
    assert commit.robot_choice == "paper"
    assert len(commit.commit_hash) == 64


def test_commit_tamper_detection():
    commit = CommitReveal.create("rock", "round_0002")
    commit.robot_choice = "paper"
    assert not commit.verify("round_0002")


def test_robot_commit_before_human_capture():
    engine = GameEngine()
    round_id = engine.new_round_id()
    commit = engine.commit_robot_choice(round_id, "scissors")
    # Commit time must be before resolution
    pred = GesturePrediction(label="scissors", confidence=0.9)
    robot_result = GestureExecutionResult("scissors", True, True)
    round_obj = engine.resolve_round(round_id, commit, pred, robot_result)
    assert round_obj.robot_choice == "scissors"
    assert round_obj.result == "draw"
    assert commit.commit_time < round_obj.started_at


def test_invalid_when_human_unknown():
    engine = GameEngine()
    round_id = engine.new_round_id()
    commit = engine.commit_robot_choice(round_id, "rock")
    pred = GesturePrediction(label="unknown", confidence=0.0)
    robot_result = GestureExecutionResult("rock", True, True)
    round_obj = engine.resolve_round(round_id, commit, pred, robot_result)
    assert round_obj.result == "invalid"


def test_invalid_when_robot_gesture_unverified():
    engine = GameEngine()
    round_id = engine.new_round_id()
    commit = engine.commit_robot_choice(round_id, "rock")
    pred = GesturePrediction(label="scissors", confidence=0.9)
    robot_result = GestureExecutionResult("rock", True, False, failure_reason="joint_not_reached")
    round_obj = engine.resolve_round(round_id, commit, pred, robot_result)
    assert round_obj.result == "invalid"


def test_game_summary():
    engine = GameEngine()
    rounds = []
    for robot, human, result in [
        ("rock", "scissors", "robot_win"),
        ("rock", "paper", "robot_lose"),
        ("rock", "rock", "draw"),
    ]:
        rid = engine.new_round_id()
        commit = CommitReveal.create(robot, rid)
        pred = GesturePrediction(label=human, confidence=0.9)
        robot_result = GestureExecutionResult(robot, True, True)
        rounds.append(engine.resolve_round(rid, commit, pred, robot_result))
    summary = engine.summary(rounds)
    assert summary["rounds_total"] == 3
    assert summary["valid_rounds"] == 3
    assert summary["robot_win"] == 1
    assert summary["robot_lose"] == 1
    assert summary["draw"] == 1
    assert summary["robot_gesture_verified_rate"] == 1.0
