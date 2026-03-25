"""Runtime session management for robot execution.

A RuntimeSession represents a single execution context for robot operations,
managing state, configuration, and adapter lifecycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from rosclaw_core.definitions.robot import RobotManifest
from rosclaw_core.definitions.assembly import AssemblyManifest
from rosclaw_core.adapters.base import RobotAdapter, AdapterState


class SessionState(Enum):
    """States of a runtime session."""
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class SessionConfig:
    """Configuration for a runtime session.

    Attributes:
        name: Session identifier
        data_root: Root directory for datasets and logs
        debug_mode: Enable debug logging and diagnostics
        auto_save: Auto-save session state
        max_history: Maximum procedure history to retain
    """

    name: str = "default"
    data_root: Path = field(default_factory=lambda: Path("~/.rosclaw/data"))
    debug_mode: bool = False
    auto_save: bool = True
    max_history: int = 100

    def __post_init__(self):
        self.data_root = Path(self.data_root).expanduser()


@dataclass
class ProcedureRecord:
    """Record of a procedure execution."""
    procedure_type: str
    started_at: float
    completed_at: float | None = None
    success: bool | None = None
    result: Any = None
    error: str | None = None


class RuntimeSession:
    """Manages a single robot execution session.

    A session encapsulates:
    - Robot/Assembly configuration
    - Hardware adapter lifecycle
    - Procedure execution history
    - State management

    Example:
        session = RuntimeSession(
            manifest=robot_manifest,
            config=SessionConfig(name="demo")
        )
        await session.initialize()
        await session.connect_hardware()
        # ... run procedures ...
        await session.shutdown()
    """

    def __init__(
        self,
        manifest: RobotManifest | AssemblyManifest,
        config: SessionConfig | None = None,
    ):
        self._manifest = manifest
        self._config = config or SessionConfig()
        self._state = SessionState.INITIALIZING
        self._adapters: dict[str, RobotAdapter] = {}
        self._history: list[ProcedureRecord] = []
        self._error_message: str | None = None
        self._initialized_at: float | None = None

    @property
    def name(self) -> str:
        """Session name."""
        return self._config.name

    @property
    def manifest(self) -> RobotManifest | AssemblyManifest:
        """Session manifest."""
        return self._manifest

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def is_single_robot(self) -> bool:
        """Check if session is for a single robot."""
        return isinstance(self._manifest, RobotManifest)

    @property
    def is_assembly(self) -> bool:
        """Check if session is for a multi-agent assembly."""
        return isinstance(self._manifest, AssemblyManifest)

    @property
    def adapters(self) -> dict[str, RobotAdapter]:
        """Active hardware adapters."""
        return self._adapters.copy()

    @property
    def history(self) -> list[ProcedureRecord]:
        """Procedure execution history."""
        return self._history.copy()

    @property
    def last_error(self) -> str | None:
        """Last error message."""
        return self._error_message

    def _set_state(self, state: SessionState, error: str | None = None) -> None:
        """Update session state."""
        self._state = state
        if error:
            self._error_message = error

    async def initialize(self) -> bool:
        """Initialize the session.

        Creates data directories and validates configuration.

        Returns:
            True if initialization successful
        """
        try:
            # Create data directories
            self._config.data_root.mkdir(parents=True, exist_ok=True)
            (self._config.data_root / "datasets").mkdir(exist_ok=True)
            (self._config.data_root / "policies").mkdir(exist_ok=True)
            (self._config.data_root / "logs").mkdir(exist_ok=True)

            # Validate manifest
            if self.is_assembly:
                errors = self._manifest.validate_assembly()
                if errors:
                    self._set_state(SessionState.ERROR, f"Validation failed: {errors}")
                    return False

            self._initialized_at = asyncio.get_event_loop().time()
            self._set_state(SessionState.READY)
            return True

        except Exception as e:
            self._set_state(SessionState.ERROR, str(e))
            return False

    async def register_adapter(self, robot_id: str, adapter: RobotAdapter) -> bool:
        """Register a hardware adapter.

        Args:
            robot_id: Robot identifier
            adapter: Hardware adapter instance

        Returns:
            True if registration successful
        """
        if robot_id in self._adapters:
            self._error_message = f"Adapter for '{robot_id}' already registered"
            return False

        self._adapters[robot_id] = adapter
        return True

    async def connect_hardware(self) -> dict[str, bool]:
        """Connect all registered hardware adapters.

        Returns:
            Dict of robot_id -> connection success
        """
        results = {}
        for robot_id, adapter in self._adapters.items():
            try:
                results[robot_id] = await adapter.connect()
            except Exception as e:
                results[robot_id] = False
                self._error_message = f"Failed to connect {robot_id}: {e}"

        # Update session state based on results
        if any(results.values()):
            self._set_state(SessionState.READY)

        return results

    async def disconnect_hardware(self) -> None:
        """Disconnect all hardware adapters."""
        for adapter in self._adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                pass
        self._adapters.clear()

    async def calibrate_all(self) -> dict[str, bool]:
        """Calibrate all connected robots.

        Returns:
            Dict of robot_id -> calibration success
        """
        results = {}
        for robot_id, adapter in self._adapters.items():
            if adapter.state == AdapterState.CONNECTED:
                try:
                    results[robot_id] = await adapter.calibrate()
                except Exception as e:
                    results[robot_id] = False
                    self._error_message = f"Calibration failed for {robot_id}: {e}"
        return results

    def record_procedure(
        self,
        procedure_type: str,
        started_at: float,
        completed_at: float | None = None,
        success: bool | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Record a procedure execution."""
        record = ProcedureRecord(
            procedure_type=procedure_type,
            started_at=started_at,
            completed_at=completed_at,
            success=success,
            result=result,
            error=error,
        )
        self._history.append(record)

        # Trim history if needed
        if len(self._history) > self._config.max_history:
            self._history = self._history[-self._config.max_history :]

    async def emergency_stop_all(self) -> None:
        """Trigger emergency stop on all robots."""
        self._set_state(SessionState.ERROR, "Emergency stop triggered")
        for adapter in self._adapters.values():
            try:
                await adapter.emergency_stop()
            except Exception:
                pass

    async def reset_all(self) -> dict[str, bool]:
        """Reset all robots from emergency stop.

        Returns:
            Dict of robot_id -> reset success
        """
        results = {}
        for robot_id, adapter in self._adapters.items():
            try:
                results[robot_id] = await adapter.reset()
            except Exception:
                results[robot_id] = False

        if all(results.values()):
            self._set_state(SessionState.READY)

        return results

    async def shutdown(self) -> None:
        """Shutdown the session cleanly."""
        await self.disconnect_hardware()
        self._set_state(SessionState.SHUTDOWN)

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "name": self.name,
            "state": self._state.name,
            "is_single_robot": self.is_single_robot,
            "is_assembly": self.is_assembly,
            "manifest": self._manifest.to_dict(),
            "adapters": {k: v.to_dict() for k, v in self._adapters.items()},
            "history_count": len(self._history),
            "error": self._error_message,
        }
