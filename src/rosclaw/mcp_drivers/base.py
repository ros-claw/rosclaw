"""
ROSClaw MCP Drivers - Base Hardware Abstraction Interface

All hardware drivers (real robots, simulators, serial devices)
implement the BaseDriver interface for uniform control.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from rosclaw.core.lifecycle import LifecycleMixin


@dataclass
class DriverState:
    """Standardized state report from any driver."""
    connected: bool = False
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    joint_torques: list[float] = field(default_factory=list)
    end_effector_pose: Optional[list[float]] = None
    gripper_state: Optional[float] = None
    error_code: int = 0
    error_message: str = ""

    def is_ready(self) -> bool:
        return self.connected and self.error_code == 0


@dataclass
class TrajectoryCommand:
    """Standard trajectory command."""
    waypoints: list[list[float]]
    times: list[float]
    velocity_limits: Optional[list[float]] = None
    acceleration_limits: Optional[list[float]] = None


class BaseDriver(LifecycleMixin, ABC):
    """
    Abstract base for all ROSClaw hardware drivers.

    Provides uniform interface for:
    - Joint position/velocity/torque control
    - Trajectory execution
    - State feedback
    - Emergency stop
    """

    def __init__(self, robot_id: str, joint_dof: int = 6):
        super().__init__()
        self.robot_id = robot_id
        self.joint_dof = joint_dof
        self._driver_state = DriverState()

    @property
    def state(self) -> DriverState:
        return self._driver_state

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """Read current joint positions in radians."""

    @abstractmethod
    def get_joint_velocities(self) -> list[float]:
        """Read current joint velocities."""

    @abstractmethod
    def get_joint_torques(self) -> list[float]:
        """Read current joint torques."""

    def _ensure_ready(self, operation: str = "operation") -> None:
        """Guard: require READY or RUNNING state."""
        if not self.is_ready:
            lifecycle = super().state
            raise RuntimeError(
                f"Cannot perform {operation}: driver lifecycle state is {lifecycle.name}. "
                f"Call initialize() and start() first."
            )

    def _validate_joint_positions(self, positions: list[float]) -> None:
        """Guard: validate joint positions are finite and within reasonable bounds."""
        import math
        if len(positions) != self.joint_dof:
            raise ValueError(
                f"Expected {self.joint_dof} joint positions, got {len(positions)}"
            )
        for i, p in enumerate(positions):
            if not isinstance(p, (int, float)):
                raise TypeError(f"Joint position {i} must be numeric, got {type(p).__name__}")
            if not math.isfinite(p):
                raise ValueError(f"Joint position {i} is not finite: {p}")
            if abs(p) > 1e6:
                raise ValueError(
                    f"Joint position {i} exceeds safe bounds: {p}. "
                    f"Max allowed absolute value is 1e6."
                )

    @abstractmethod
    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        """Command joints to target positions."""

    @abstractmethod
    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        """Execute a multi-point trajectory."""

    @abstractmethod
    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        """Set gripper position (0=open, 1=closed)."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediate halt of all motion."""

    @abstractmethod
    def get_state(self) -> DriverState:
        """Get full driver state."""

    def is_connected(self) -> bool:
        return self._driver_state.connected
