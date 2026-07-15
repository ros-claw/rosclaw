"""Rock-Paper-Scissors game engine with commit-reveal fairness."""
from __future__ import annotations

import hashlib
import random
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from .gesture_schema import GesturePrediction, GestureExecutionResult, RPSRound


RPS_BEATS = {
    "rock": "scissors",
    "scissors": "paper",
    "paper": "rock",
}
RPS_CHOICES = list(RPS_BEATS.keys())
RESULT_GESTURES = {
    "robot_win": "win",
    "robot_lose": "lose",
    "draw": "draw",
    "invalid": "error",
}
RESULT_REFEREE_GESTURES = {
    "robot_win": "left_pinkie",
    "robot_lose": "left_thumb_up",
    "draw": "left_orchid",
    "invalid": "left_ready",
}


def judge(robot_choice: str, human_choice: str) -> str:
    """Return robot_win | robot_lose | draw."""
    if robot_choice not in RPS_CHOICES or human_choice not in RPS_CHOICES:
        return "invalid"
    if robot_choice == human_choice:
        return "draw"
    if RPS_BEATS[robot_choice] == human_choice:
        return "robot_win"
    return "robot_lose"


@dataclass
class CommitReveal:
    robot_choice: str
    nonce: str
    commit_time: float
    commit_hash: str

    @classmethod
    def create(cls, robot_choice: str, round_id: str) -> "CommitReveal":
        nonce = secrets.token_hex(8)
        commit_time = time.time()
        payload = f"{round_id}:{robot_choice}:{nonce}".encode()
        commit_hash = hashlib.sha256(payload).hexdigest()
        return cls(
            robot_choice=robot_choice,
            nonce=nonce,
            commit_time=commit_time,
            commit_hash=commit_hash,
        )

    def verify(self, round_id: str) -> bool:
        payload = f"{round_id}:{self.robot_choice}:{self.nonce}".encode()
        return hashlib.sha256(payload).hexdigest() == self.commit_hash


class GameEngine:
    """State machine for a single RPS round."""

    def __init__(
        self,
        result_gesture_map: Optional[Dict[str, str]] = None,
        referee_gesture_map: Optional[Dict[str, str]] = None,
    ):
        self.result_gesture_map = result_gesture_map or RESULT_GESTURES.copy()
        self.referee_gesture_map = referee_gesture_map or RESULT_REFEREE_GESTURES.copy()
        self._round_counter = 0

    def new_round_id(self) -> str:
        self._round_counter += 1
        return f"round_{self._round_counter:04d}"

    def commit_robot_choice(self, round_id: str, choice: Optional[str] = None) -> CommitReveal:
        if choice is None:
            choice = random.choice(RPS_CHOICES)
        if choice not in RPS_CHOICES:
            raise ValueError(f"Invalid robot choice: {choice}")
        return CommitReveal.create(choice, round_id)

    def resolve_round(
        self,
        round_id: str,
        commit: CommitReveal,
        human_prediction: GesturePrediction,
        robot_gesture_result: GestureExecutionResult,
    ) -> RPSRound:
        started_at = time.time()
        result = "invalid"
        result_gesture = self.result_gesture_map.get("invalid", "error")

        # Fairness: commit must verify and predate capture.
        commit_ok = commit.verify(round_id) and commit.commit_time < started_at

        if not commit_ok:
            human_choice = "unknown"
        else:
            human_choice = human_prediction.label if human_prediction.label in RPS_CHOICES else "unknown"

        if human_choice == "unknown":
            result = "invalid"
        elif not robot_gesture_result.verified:
            result = "invalid"
        else:
            result = judge(commit.robot_choice, human_choice)

        result_gesture = self.result_gesture_map.get(result, "error")
        referee_gesture = self.referee_gesture_map.get(result, "left_ready")

        return RPSRound(
            round_id=round_id,
            started_at=started_at,
            robot_choice=commit.robot_choice,
            robot_commit_hash=commit.commit_hash,
            robot_commit_nonce=commit.nonce,
            human_prediction=human_prediction,
            robot_gesture_result=robot_gesture_result,
            result=result,
            result_gesture=result_gesture,
            referee_gesture=referee_gesture,
            trace_id=str(uuid.uuid4()),
            ended_at=time.time(),
        )

    def summary(self, rounds: List[RPSRound]) -> dict:
        total = len(rounds)
        valid = [r for r in rounds if r.result != "invalid"]
        return {
            "rounds_total": total,
            "valid_rounds": len(valid),
            "robot_win": sum(1 for r in valid if r.result == "robot_win"),
            "robot_lose": sum(1 for r in valid if r.result == "robot_lose"),
            "draw": sum(1 for r in valid if r.result == "draw"),
            "invalid": total - len(valid),
            "robot_gesture_verified_rate": (
                sum(1 for r in rounds if r.robot_gesture_result.verified) / total if total else 0.0
            ),
        }
