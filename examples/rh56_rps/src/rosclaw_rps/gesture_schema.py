"""Core dataclasses / schemas for the RH56 RPS demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GestureConfig:
    """Configuration for a single RH56 gesture."""

    name: str
    description: str
    angles: List[int]
    hold_s: float = 0.8
    speed: int = 600
    force: int = 300
    verify_tolerance: int = 80
    verify_current_max: int = 800
    verify_force_max: int = 800
    verify_driver_status: bool = True

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "GestureConfig":
        verify = data.get("verify", {})
        return cls(
            name=name,
            description=data.get("description", ""),
            angles=list(data["angles"]),
            hold_s=float(data.get("hold_s", 0.8)),
            speed=int(data.get("speed", 600)),
            force=int(data.get("force", 300)),
            verify_tolerance=int(verify.get("tolerance", 80)),
            verify_current_max=int(verify.get("current_max", 800)),
            verify_force_max=int(verify.get("force_max", 800)),
            verify_driver_status=bool(verify.get("check_driver_status", True)),
        )


@dataclass
class HandTelemetry:
    """Snapshot of RH56 telemetry after a gesture move."""

    timestamp: float
    angle_actual: Dict[str, int] = field(default_factory=dict)
    angle_set: Dict[str, int] = field(default_factory=dict)
    force_act: Dict[str, Optional[int]] = field(default_factory=dict)
    current_ma: Dict[str, Optional[int]] = field(default_factory=dict)
    temperature_c: Dict[str, Optional[int]] = field(default_factory=dict)
    error: Dict[str, Optional[int]] = field(default_factory=dict)
    status: Dict[str, Optional[int]] = field(default_factory=dict)


@dataclass
class VerificationResult:
    ok: bool
    confidence: float
    failure_type: Optional[str]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestureExecutionResult:
    gesture_name: str
    command_success: bool
    verified: bool
    telemetry: Optional[HandTelemetry] = None
    failure_reason: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


@dataclass
class GesturePrediction:
    label: str  # "rock" | "paper" | "scissors" | "unknown"
    confidence: float
    landmarks: Optional[List[Any]] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraFrame:
    color: np.ndarray
    timestamp: float
    depth: Optional[np.ndarray] = None


@dataclass
class RPSRound:
    round_id: str
    started_at: float
    robot_choice: str
    robot_commit_hash: str
    robot_commit_nonce: str
    human_prediction: GesturePrediction = field(default_factory=lambda: GesturePrediction("unknown", 0.0))
    robot_gesture_result: GestureExecutionResult = field(
        default_factory=lambda: GestureExecutionResult("error", False, False)
    )
    result: str = "invalid"  # robot_win | robot_lose | draw | invalid
    result_gesture: str = "error"
    referee_gesture: str = "left_ready"
    trace_id: str = ""
    ended_at: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "started_at": self.started_at,
            "robot_choice": self.robot_choice,
            "robot_commit_hash": self.robot_commit_hash,
            "human_prediction": {
                "label": self.human_prediction.label,
                "confidence": self.human_prediction.confidence,
            },
            "robot_gesture_verified": self.robot_gesture_result.verified,
            "robot_gesture_failure_reason": self.robot_gesture_result.failure_reason,
            "result": self.result,
            "result_gesture": self.result_gesture,
            "referee_gesture": self.referee_gesture,
            "trace_id": self.trace_id,
            "ended_at": self.ended_at,
            "latency_ms": self.latency_ms,
        }
