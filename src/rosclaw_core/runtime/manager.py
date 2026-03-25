"""Runtime manager for procedure execution.

Manages the lifecycle of runtime sessions and coordinates procedure
execution across single robots and multi-agent assemblies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw_core.definitions.robot import RobotManifest
from rosclaw_core.definitions.assembly import AssemblyManifest
from rosclaw_core.runtime.session import RuntimeSession, SessionConfig, SessionState
from rosclaw_core.runtime.executor import ProcedureExecutor


@dataclass
class RuntimeManagerConfig:
    """Configuration for the runtime manager.

    Attributes:
        max_sessions: Maximum concurrent sessions
        default_session_config: Default configuration for new sessions
        auto_discover: Auto-discover connected hardware
    """

    max_sessions: int = 10
    default_session_config: SessionConfig | None = None
    auto_discover: bool = True


class RuntimeManager:
    """Central manager for robot runtime sessions.

    The RuntimeManager:
    - Manages multiple concurrent sessions
    - Handles session lifecycle (create, start, stop, destroy)
    - Provides procedure execution coordination
    - Maintains session registry

    Example:
        manager = RuntimeManager()

        # Create a single robot session
        session = await manager.create_robot_session(manifest)

        # Execute a procedure
        result = await manager.execute_procedure(
            session.name,
            ConnectProcedure(),
            {"port": "/dev/ttyACM0"}
        )

        # Cleanup
        await manager.destroy_session(session.name)
    """

    _instance: RuntimeManager | None = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one RuntimeManager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: RuntimeManagerConfig | None = None):
        if hasattr(self, "_initialized"):
            return

        self._config = config or RuntimeManagerConfig()
        self._sessions: dict[str, RuntimeSession] = {}
        self._executors: dict[str, ProcedureExecutor] = {}
        self._lock = asyncio.Lock()
        self._initialized = True

    @property
    def session_names(self) -> list[str]:
        """List of active session names."""
        return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    async def create_robot_session(
        self,
        manifest: RobotManifest,
        config: SessionConfig | None = None,
    ) -> RuntimeSession:
        """Create a new single-robot session.

        Args:
            manifest: Robot hardware specification
            config: Optional session configuration

        Returns:
            Initialized RuntimeSession

        Raises:
            RuntimeError: If max sessions reached or session exists
        """
        async with self._lock:
            if len(self._sessions) >= self._config.max_sessions:
                raise RuntimeError(f"Max sessions ({self._config.max_sessions}) reached")

            session_config = config or self._config.default_session_config or SessionConfig()
            if manifest.name in self._sessions:
                raise RuntimeError(f"Session '{manifest.name}' already exists")

            session = RuntimeSession(manifest=manifest, config=session_config)
            executor = ProcedureExecutor(session)

            if not await session.initialize():
                raise RuntimeError(f"Failed to initialize session: {session.last_error}")

            self._sessions[manifest.name] = session
            self._executors[manifest.name] = executor

            return session

    async def create_assembly_session(
        self,
        manifest: AssemblyManifest,
        config: SessionConfig | None = None,
    ) -> RuntimeSession:
        """Create a new multi-agent assembly session.

        Args:
            manifest: Assembly specification
            config: Optional session configuration

        Returns:
            Initialized RuntimeSession
        """
        async with self._lock:
            if len(self._sessions) >= self._config.max_sessions:
                raise RuntimeError(f"Max sessions ({self._config.max_sessions}) reached")

            session_config = config or self._config.default_session_config or SessionConfig()
            if manifest.name in self._sessions:
                raise RuntimeError(f"Session '{manifest.name}' already exists")

            # Validate assembly before creating session
            errors = manifest.validate_assembly()
            if errors:
                raise RuntimeError(f"Assembly validation failed: {errors}")

            session = RuntimeSession(manifest=manifest, config=session_config)
            executor = ProcedureExecutor(session)

            if not await session.initialize():
                raise RuntimeError(f"Failed to initialize session: {session.last_error}")

            self._sessions[manifest.name] = session
            self._executors[manifest.name] = executor

            return session

    async def get_session(self, name: str) -> RuntimeSession | None:
        """Get a session by name.

        Args:
            name: Session name

        Returns:
            RuntimeSession or None if not found
        """
        return self._sessions.get(name)

    async def destroy_session(self, name: str) -> bool:
        """Destroy a session and cleanup resources.

        Args:
            name: Session name

        Returns:
            True if session was destroyed
        """
        async with self._lock:
            session = self._sessions.get(name)
            if not session:
                return False

            await session.shutdown()
            del self._sessions[name]
            del self._executors[name]
            return True

    async def execute_procedure(
        self,
        session_name: str,
        procedure: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a procedure in a session.

        Args:
            session_name: Target session name
            procedure: Procedure to execute
            params: Procedure parameters

        Returns:
            Procedure result

        Raises:
            RuntimeError: If session not found or execution fails
        """
        executor = self._executors.get(session_name)
        if not executor:
            raise RuntimeError(f"Session '{session_name}' not found")

        return await executor.execute(procedure, params or {})

    async def execute_all(
        self,
        procedure: Any,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a procedure in all sessions.

        Args:
            procedure: Procedure to execute
            params: Procedure parameters

        Returns:
            Dict of session_name -> result
        """
        results = {}
        for name, executor in self._executors.items():
            try:
                results[name] = await executor.execute(procedure, params or {})
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    async def emergency_stop_all(self) -> None:
        """Trigger emergency stop on all sessions."""
        for session in self._sessions.values():
            await session.emergency_stop_all()

    async def get_session_status(self, name: str) -> dict[str, Any] | None:
        """Get status of a session.

        Args:
            name: Session name

        Returns:
            Session status dict or None
        """
        session = self._sessions.get(name)
        return session.to_dict() if session else None

    async def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all sessions.

        Returns:
            Dict of session_name -> status
        """
        return {name: session.to_dict() for name, session in self._sessions.items()}

    async def shutdown(self) -> None:
        """Shutdown all sessions and cleanup."""
        for session in list(self._sessions.values()):
            await session.shutdown()
        self._sessions.clear()
        self._executors.clear()
        RuntimeManager._instance = None
