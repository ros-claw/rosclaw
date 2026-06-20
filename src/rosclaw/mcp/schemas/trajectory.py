"""Schema for the ``validate_trajectory`` MCP tool."""

from __future__ import annotations

from typing import TypedDict


class ValidateTrajectoryResponse(TypedDict):
    """Envelope payload returned by ``validate_trajectory``."""

    is_safe: bool
    risk_score: float
    reason: str
    violations: list[str]
    replay_id: str | None
