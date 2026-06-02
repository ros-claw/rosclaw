"""ROSClaw Core Types - Shared type definitions for all modules.

This module provides canonical data structures used across ROSClaw:
- RobotState: Unified robot state representation
- PraxisEvent: Unified practice event binding LLM CoT with physical data
- PraxisEventType: Type-safe enumeration for praxis event types

RFC-0001 Core Types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class PraxisEventType(Enum):
    """Type-safe enumeration for praxis event types."""
    SUCCESS = "success"
    FAILURE = "failure"
    EMERGENCY = "emergency"
    MOVE = "move"
    GRASP = "grasp"
    VALIDATE = "validate"


@dataclass
class RobotState:
    """
    Canonical robot state representation.

    Used by ALL modules — replaces duplicate definitions in
    data/flywheel.py and mcp/ur5_server.py.
    """
    timestamp: float
    joint_positions: np.ndarray         # Shape: (dof,)
    joint_velocities: np.ndarray        # Shape: (dof,)
    joint_torques: np.ndarray           # Shape: (dof,)
    joint_names: list[str] = field(default_factory=list)
    end_effector_pose: Optional[np.ndarray] = None  # Shape: (4, 4)
    gripper_state: Optional[float] = None            # 0.0=open, 1.0=closed
    is_connected: bool = True

    def validate(self, expected_dof: int) -> bool:
        return (
            self.joint_positions.shape == (expected_dof,) and  # noqa: W504
            self.joint_velocities.shape == (expected_dof,) and  # noqa: W504
            self.joint_torques.shape == (expected_dof,)
        )


@dataclass(frozen=True)
class PraxisEvent:
    """
    Unified practice event — the core data structure of ROSClaw.

    Binds LLM reasoning (CoT) with physical execution data on a single timeline.
    Published on EventBus topics: praxis.completed, praxis.failed, praxis.emergency

    RFC-0001 core type.
    """
    event_id: str
    event_type: str              # Use PraxisEventType.SUCCESS.value etc
    timestamp: float
    robot_id: str
    agent_instruction: str      # LLM's original natural language instruction
    cot_trace: list[str]         # Chain-of-Thought reasoning steps
    initial_state: RobotState
    final_state: Optional[RobotState]
    trajectory: list[list[float]]
    mcap_path: Optional[str]     # Path to MCAP recording
    error_details: Optional[str]
    duration_sec: float
    metadata: dict = field(default_factory=dict)
