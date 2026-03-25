"""Base adapter protocol for ROS/ROS2 hardware integration.

Provides the abstraction layer between robot manifests and physical
(or simulated) hardware via ROS/ROS2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class AdapterState(Enum):
    """States of an adapter connection."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    CALIBRATING = auto()
    READY = auto()
    ERROR = auto()


class ControlMode(Enum):
    """Robot control modes."""
    IDLE = auto()
    TELEOP = auto()      # Manual control via leader
    POLICY = auto()      # AI policy control
    SERVO = auto()       # Direct servo control
    EMERGENCY_STOP = auto()


@dataclass
class JointState:
    """Current state of a joint."""
    name: str
    position: float
    velocity: float = 0.0
    effort: float = 0.0
    timestamp: float = 0.0


@dataclass
class RobotState:
    """Complete state of a robot."""
    joint_states: dict[str, JointState]
    gripper_position: float | None = None
    timestamp: float = 0.0
    is_ready: bool = False


@runtime_checkable
class AdapterProtocol(Protocol):
    """Protocol for robot adapters.

    All hardware adapters must implement this protocol to integrate
    with the ROSClaw runtime system.
    """

    @property
    def state(self) -> AdapterState:
        """Current adapter state."""
        ...

    @property
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        ...

    async def connect(self) -> bool:
        """Connect to hardware. Returns success."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from hardware."""
        ...

    async def calibrate(self) -> bool:
        """Run calibration routine. Returns success."""
        ...

    async def read_state(self) -> RobotState:
        """Read current robot state from hardware."""
        ...

    async def write_command(self, positions: dict[str, float]) -> bool:
        """Write joint position commands to hardware."""
        ...

    async def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        ...

    async def reset(self) -> bool:
        """Reset from emergency stop."""
        ...


class RobotAdapter(ABC):
    """Abstract base class for robot adapters.

    Provides common functionality and state management for hardware adapters.
    Subclasses implement transport-specific logic (ROS1, ROS2, WebSocket, etc.)
    """

    def __init__(self, robot_id: str, namespace: str = "/robot"):
        self._robot_id = robot_id
        self._namespace = namespace
        self._state = AdapterState.DISCONNECTED
        self._control_mode = ControlMode.IDLE
        self._last_state: RobotState | None = None
        self._error_message: str | None = None

    @property
    def robot_id(self) -> str:
        """Unique robot identifier."""
        return self._robot_id

    @property
    def namespace(self) -> str:
        """ROS namespace."""
        return self._namespace

    @property
    def state(self) -> AdapterState:
        """Current adapter state."""
        return self._state

    @property
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        return self._control_mode

    @property
    def is_ready(self) -> bool:
        """Check if adapter is ready for commands."""
        return self._state == AdapterState.READY

    @property
    def last_error(self) -> str | None:
        """Last error message."""
        return self._error_message

    def _set_state(self, state: AdapterState, error: str | None = None) -> None:
        """Update adapter state."""
        self._state = state
        if error:
            self._error_message = error

    def _set_control_mode(self, mode: ControlMode) -> None:
        """Update control mode."""
        self._control_mode = mode

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to hardware.

        Implementations should:
        1. Set state to CONNECTING
        2. Establish transport connection
        3. Set state to CONNECTED or ERROR
        """
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from hardware.

        Implementations should:
        1. Set state to DISCONNECTED
        2. Close transport connection
        """
        raise NotImplementedError

    @abstractmethod
    async def calibrate(self) -> bool:
        """Run calibration routine.

        Implementations should:
        1. Set state to CALIBRATING
        2. Execute calibration sequence
        3. Set state to READY or ERROR
        """
        raise NotImplementedError

    @abstractmethod
    async def read_state(self) -> RobotState:
        """Read current robot state.

        Returns:
            RobotState with current joint positions, velocities, efforts
        """
        raise NotImplementedError

    @abstractmethod
    async def write_command(self, positions: dict[str, float]) -> bool:
        """Write joint position commands.

        Args:
            positions: Dict of joint_name -> target_position

        Returns:
            True if command was accepted
        """
        raise NotImplementedError

    async def emergency_stop(self) -> None:
        """Trigger emergency stop.

        Default implementation sets control mode and state.
        Subclasses should override to send hardware commands.
        """
        self._set_control_mode(ControlMode.EMERGENCY_STOP)
        self._set_state(AdapterState.ERROR, "Emergency stop triggered")

    async def reset(self) -> bool:
        """Reset from emergency stop.

        Returns:
            True if reset successful
        """
        if self._control_mode == ControlMode.EMERGENCY_STOP:
            self._set_control_mode(ControlMode.IDLE)
            self._set_state(AdapterState.CONNECTED)
            return True
        return False

    async def enter_teleop(self) -> bool:
        """Enter teleoperation mode."""
        if not self.is_ready:
            return False
        self._set_control_mode(ControlMode.TELEOP)
        return True

    async def exit_teleop(self) -> bool:
        """Exit teleoperation mode."""
        if self._control_mode == ControlMode.TELEOP:
            self._set_control_mode(ControlMode.IDLE)
            return True
        return False

    async def enter_policy(self) -> bool:
        """Enter AI policy control mode."""
        if not self.is_ready:
            return False
        self._set_control_mode(ControlMode.POLICY)
        return True

    async def exit_policy(self) -> bool:
        """Exit AI policy control mode."""
        if self._control_mode == ControlMode.POLICY:
            self._set_control_mode(ControlMode.IDLE)
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert adapter state to dictionary."""
        return {
            "robot_id": self._robot_id,
            "namespace": self._namespace,
            "state": self._state.name,
            "control_mode": self._control_mode.name,
            "is_ready": self.is_ready,
            "error": self._error_message,
        }
