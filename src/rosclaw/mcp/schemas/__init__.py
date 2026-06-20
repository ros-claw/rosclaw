"""Shared MCP P0 schema definitions.

Each submodule defines TypedDict-style structures for one MCP tool response.
Import them here for convenient access by the server and adapters.
"""

from __future__ import annotations

from rosclaw.mcp.schemas.common import MCPError, make_error, make_response
from rosclaw.mcp.schemas.memory import QueryMemoryResponse
from rosclaw.mcp.schemas.practice import PracticeQueryResponse
from rosclaw.mcp.schemas.robot_state import (
    BodyState,
    Readiness,
    RiskSummary,
    RobotStateResponse,
)
from rosclaw.mcp.schemas.safety import EmergencyStopResponse
from rosclaw.mcp.schemas.sandbox import SandboxRunResponse
from rosclaw.mcp.schemas.skill import ListSkillsResponse
from rosclaw.mcp.schemas.trajectory import ValidateTrajectoryResponse

__all__ = [
    "MCPError",
    "QueryMemoryResponse",
    "PracticeQueryResponse",
    "BodyState",
    "Readiness",
    "RiskSummary",
    "RobotStateResponse",
    "EmergencyStopResponse",
    "SandboxRunResponse",
    "ListSkillsResponse",
    "ValidateTrajectoryResponse",
    "make_error",
    "make_response",
]
