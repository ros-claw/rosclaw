"""Integration test for a full mock round."""
from __future__ import annotations

import uuid

from rosclaw_rps.game_engine import GameEngine
from rosclaw_rps.gesture_schema import GestureConfig, GesturePrediction
from rosclaw_rps.hand.gesture_executor import GestureExecutor, GestureVerifier
from rosclaw_rps.hand.rh56_controller import MockHandController
from rosclaw_rps.logging.round_logger import RoundLogger


def test_mock_round_logging(tmp_path):
    gestures = {
        "ready": GestureConfig(name="ready", description="ready", angles=[1000] * 6),
        "rock": GestureConfig(name="rock", description="rock", angles=[0, 0, 0, 0, 0, 1000]),
        "paper": GestureConfig(name="paper", description="paper", angles=[1000] * 6),
        "scissors": GestureConfig(name="scissors", description="scissors", angles=[0, 0, 1000, 1000, 0, 1000]),
        "win": GestureConfig(name="win", description="win", angles=[1000, 1000, 1000, 410, 420, 300]),
        "lose": GestureConfig(name="lose", description="lose", angles=[0, 0, 0, 1000, 0, 1000]),
        "draw": GestureConfig(name="draw", description="draw", angles=[0, 1000, 1000, 1000, 0, 1000]),
        "error": GestureConfig(name="error", description="error", angles=[1000] * 6),
    }
    hand = MockHandController()
    executor = GestureExecutor(hand, gestures, GestureVerifier())
    engine = GameEngine()
    logger = RoundLogger(tmp_path / "run", save_frames=False, save_telemetry=True)

    executor.execute("ready")
    round_id = engine.new_round_id()
    commit = engine.commit_robot_choice(round_id, "rock")
    human_pred = GesturePrediction(label="scissors", confidence=0.9)
    robot_result = executor.execute(commit.robot_choice)
    round_obj = engine.resolve_round(round_id, commit, human_pred, robot_result)
    result_gesture = engine.result_gesture_map.get(round_obj.result, "error")
    executor.execute(result_gesture)
    logger.log_round(round_obj)

    assert round_obj.result == "robot_win"
    assert (tmp_path / "run" / "rounds.jsonl").exists()
    summary = engine.summary(logger.rounds)
    assert summary["valid_rounds"] == 1
    assert summary["robot_win"] == 1
