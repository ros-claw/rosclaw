"""Schema for the ``get_robot_state`` MCP tool."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class BodyState(TypedDict, total=False):
    """Kinematic state of the robot body."""

    joint_positions: list[float]
    joint_velocities: list[float]
    timestamp: float


class Readiness(TypedDict, total=False):
    """Readiness summary produced by SenseRuntime."""

    overall: str
    factors: dict[str, Any]


class RiskSummary(TypedDict, total=False):
    """Risk summary produced by SenseRuntime."""

    risk_level: str
    violations: list[str]


class RobotStateResponse(TypedDict):
    """Envelope payload returned by ``get_robot_state``."""

    robot_id: str
    mode: str
    body_state: Any
    body_sense: Any
    readiness: Any
    is_stale: bool
    age_ms: int | float
    note: NotRequired[str]
