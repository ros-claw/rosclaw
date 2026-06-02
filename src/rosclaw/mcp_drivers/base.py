"""
ROSClaw MCP Drivers - Base Hardware Abstraction Interface

All hardware drivers (real robots, simulators, serial devices)
implement the BaseDriver interface for uniform control.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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

    def to_dict(self) -> dict:
        """Return state as a plain dict for easy inspection."""
        return {
            "connected": self.connected,
            "joint_positions": list(self.joint_positions),
            "joint_velocities": list(self.joint_velocities),
            "joint_torques": list(self.joint_torques),
            "end_effector_pose": list(self.end_effector_pose) if self.end_effector_pose else None,
            "gripper_state": self.gripper_state,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


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

    @property
    def state_dict(self) -> dict:
        """Get driver state as a plain dict for easy inspection."""
        return self._driver_state.to_dict()

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
            if abs(p) > 1e5:
                raise ValueError(
                    f"Joint position {i} exceeds safe bounds: {p}. "
                    f"Max allowed absolute value is 1e5."
                )

    def _validate_duration(self, duration: float) -> None:
        """Guard: validate duration is positive and finite."""
        import math
        if not isinstance(duration, (int, float)):
            raise TypeError(f"Duration must be numeric, got {type(duration).__name__}")
        if not math.isfinite(duration):
            raise ValueError(f"Duration must be finite, got {duration}")
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")

    def _validate_trajectory(self, trajectory: TrajectoryCommand) -> None:
        """Guard: validate trajectory has matching waypoint lengths and positive times."""
        import math
        if not trajectory.waypoints:
            raise ValueError("Trajectory must have at least one waypoint")
        if len(trajectory.waypoints) != len(trajectory.times):
            raise ValueError(
                f"Waypoint count ({len(trajectory.waypoints)}) != time count ({len(trajectory.times)})"
            )
        for i, wp in enumerate(trajectory.waypoints):
            if len(wp) != self.joint_dof:
                raise ValueError(
                    f"Waypoint {i} has {len(wp)} joints, expected {self.joint_dof}"
                )
            for j, p in enumerate(wp):
                if not math.isfinite(p):
                    raise ValueError(f"Waypoint {i} joint {j} is not finite: {p}")
        for i, t in enumerate(trajectory.times):
            if not math.isfinite(t):
                raise ValueError(f"Time {i} is not finite: {t}")
            if t < 0:
                raise ValueError(f"Time {i} is negative: {t}")

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
