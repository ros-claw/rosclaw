"""Tests for gesture mapping, executor, and verifier."""
from __future__ import annotations

import pytest

from rosclaw_rps.gesture_schema import GestureConfig, HandTelemetry
from rosclaw_rps.hand.gesture_executor import GestureVerifier, GestureExecutor
from rosclaw_rps.hand.rh56_controller import MockHandController


def _gesture(name: str, angles: list) -> GestureConfig:
    return GestureConfig(name=name, description=name, angles=angles)


def test_gesture_verifier_success():
    verifier = GestureVerifier()
    config = _gesture("paper", [1000, 1000, 1000, 1000, 1000, 1000])
    telemetry = HandTelemetry(
        timestamp=0.0,
        angle_actual={n: 1000 for n in ["little", "ring", "middle", "index", "thumb", "thumb_rot"]},
    )
    result = verifier.verify(config, telemetry)
    assert result.ok
    assert result.confidence > 0.9
    assert result.failure_type is None


def test_gesture_verifier_joint_not_reached():
    verifier = GestureVerifier()
    config = _gesture("rock", [0, 0, 0, 0, 0, 1000])
    telemetry = HandTelemetry(
        timestamp=0.0,
        angle_actual={n: 1000 for n in ["little", "ring", "middle", "index", "thumb", "thumb_rot"]},
    )
    result = verifier.verify(config, telemetry)
    assert not result.ok
    assert result.failure_type == "joint_not_reached"


def test_gesture_verifier_over_current():
    verifier = GestureVerifier()
    config = _gesture("rock", [0, 0, 0, 0, 0, 1000])
    telemetry = HandTelemetry(
        timestamp=0.0,
        angle_actual={n: 0 for n in ["little", "ring", "middle", "index", "thumb"]},
        angle_set={"thumb_rot": 1000},
        current_ma={n: 1200 for n in ["little", "ring", "middle", "index", "thumb", "thumb_rot"]},
    )
    result = verifier.verify(config, telemetry)
    assert not result.ok
    assert result.failure_type == "over_current"


def test_mock_hand_all_gestures():
    gestures = {
        "ready": _gesture("ready", [1000] * 6),
        "rock": _gesture("rock", [0, 0, 0, 0, 0, 1000]),
        "paper": _gesture("paper", [1000] * 6),
        "scissors": _gesture("scissors", [0, 0, 1000, 1000, 0, 1000]),
        "win": _gesture("win", [1000, 1000, 1000, 410, 420, 300]),
        "lose": _gesture("lose", [0, 0, 0, 1000, 0, 1000]),
        "draw": _gesture("draw", [0, 1000, 1000, 1000, 0, 1000]),
        "error": _gesture("error", [1000] * 6),
    }
    hand = MockHandController()
    executor = GestureExecutor(hand, gestures, GestureVerifier())
    for name in ["ready", "rock", "paper", "scissors", "win", "lose", "draw", "error"]:
        result = executor.execute(name)
        assert result.command_success
        assert result.verified, f"{name} should verify on mock hand"


def test_executor_safe_open_on_failure():
    class StuckHandController(MockHandController):
        """Always reports open palm regardless of commanded gesture."""

        def read_telemetry(self):
            tel = super().read_telemetry()
            tel.angle_actual = {n: 1000 for n in tel.angle_actual}
            return tel

    gestures = {
        "rock": _gesture("rock", [0, 0, 0, 0, 0, 1000]),
        "error": _gesture("error", [1000] * 6),
    }
    hand = StuckHandController()
    executor = GestureExecutor(hand, gestures, GestureVerifier())
    result = executor.execute("rock")
    assert not result.verified
    assert result.failure_reason == "joint_not_reached"
    # safe_open should have been called
    tel = hand.read_telemetry()
    assert all(v == 1000 for v in tel.angle_actual.values())
