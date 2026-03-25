"""Procedure executor for robot operations.

Provides the execution engine for robot procedures, handling
lifecycle, error recovery, and result tracking.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from rosclaw_core.runtime.session import RuntimeSession, SessionState


class ProcedureStatus(Enum):
    """Status of procedure execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ProcedureResult:
    """Result of procedure execution.

    Attributes:
        status: Final execution status
        data: Result data (procedure-specific)
        error: Error message if failed
        duration_ms: Execution duration in milliseconds
    """

    status: ProcedureStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Check if procedure succeeded."""
        return self.status == ProcedureStatus.COMPLETED


class Procedure(ABC):
    """Abstract base class for robot procedures.

    Procedures represent discrete operations that can be executed
    against a robot or assembly. They follow a lifecycle of:
    validate -> pre_execute -> execute -> post_execute

    Example:
        class MoveProcedure(Procedure):
            @property
            def name(self) -> str:
                return "MOVE"

            def validate(self, session, params):
                if "joint_positions" not in params:
                    return "joint_positions required"
                return None

            async def execute(self, session, params):
                # ... perform move ...
                return {"positions_reached": True}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Procedure name/identifier."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"{self.name} procedure"

    @property
    def timeout_seconds(self) -> float:
        """Default timeout for this procedure."""
        return 60.0

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        """Validate parameters before execution.

        Args:
            session: Target runtime session
            params: Procedure parameters

        Returns:
            Error message if invalid, None if valid
        """
        if session.state == SessionState.ERROR:
            return "Session is in error state"
        if session.state == SessionState.SHUTDOWN:
            return "Session is shutdown"
        return None

    async def pre_execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> bool:
        """Prepare for execution.

        Args:
            session: Target runtime session
            params: Procedure parameters

        Returns:
            True if preparation successful
        """
        return True

    @abstractmethod
    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> Any:
        """Execute the procedure.

        Args:
            session: Target runtime session
            params: Procedure parameters

        Returns:
            Procedure result data
        """
        raise NotImplementedError

    async def post_execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
        result: Any,
    ) -> None:
        """Cleanup after execution.

        Args:
            session: Target runtime session
            params: Procedure parameters
            result: Execution result
        """
        pass

    async def on_error(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
        error: Exception,
    ) -> None:
        """Handle execution error.

        Args:
            session: Target runtime session
            params: Procedure parameters
            error: Exception that occurred
        """
        pass


class ProcedureExecutor:
    """Executes procedures within a runtime session.

    Manages procedure lifecycle, timeouts, and result tracking.
    """

    def __init__(self, session: RuntimeSession):
        self._session = session
        self._current_procedure: Procedure | None = None
        self._cancel_event = asyncio.Event()

    @property
    def session(self) -> RuntimeSession:
        """Associated runtime session."""
        return self._session

    @property
    def is_executing(self) -> bool:
        """Check if a procedure is currently executing."""
        return self._current_procedure is not None

    async def execute(
        self,
        procedure: Procedure,
        params: dict[str, Any],
    ) -> ProcedureResult:
        """Execute a procedure.

        Args:
            procedure: Procedure to execute
            params: Procedure parameters

        Returns:
            ProcedureResult with status and data
        """
        start_time = asyncio.get_event_loop().time()
        self._current_procedure = procedure
        self._cancel_event.clear()

        # Validate
        validation_error = procedure.validate(self._session, params)
        if validation_error:
            return self._create_result(
                ProcedureStatus.FAILED,
                error=validation_error,
                start_time=start_time,
            )

        # Pre-execute
        try:
            if not await procedure.pre_execute(self._session, params):
                return self._create_result(
                    ProcedureStatus.FAILED,
                    error="Pre-execution failed",
                    start_time=start_time,
                )
        except Exception as e:
            return self._create_result(
                ProcedureStatus.FAILED,
                error=f"Pre-execution error: {e}",
                start_time=start_time,
            )

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                procedure.execute(self._session, params),
                timeout=procedure.timeout_seconds,
            )

            # Check if cancelled
            if self._cancel_event.is_set():
                return self._create_result(
                    ProcedureStatus.CANCELLED,
                    start_time=start_time,
                )

            # Post-execute
            await procedure.post_execute(self._session, params, result)

            # Record success
            self._session.record_procedure(
                procedure_type=procedure.name,
                started_at=start_time,
                completed_at=asyncio.get_event_loop().time(),
                success=True,
                result=result,
            )

            return self._create_result(
                ProcedureStatus.COMPLETED,
                data=result,
                start_time=start_time,
            )

        except asyncio.TimeoutError:
            await procedure.on_error(
                self._session,
                params,
                TimeoutError(f"Procedure timed out after {procedure.timeout_seconds}s"),
            )
            return self._create_result(
                ProcedureStatus.FAILED,
                error=f"Timeout after {procedure.timeout_seconds}s",
                start_time=start_time,
            )

        except Exception as e:
            await procedure.on_error(self._session, params, e)
            return self._create_result(
                ProcedureStatus.FAILED,
                error=str(e),
                start_time=start_time,
            )

        finally:
            self._current_procedure = None

    async def cancel(self) -> bool:
        """Cancel the current procedure.

        Returns:
            True if cancellation was requested
        """
        if self._current_procedure:
            self._cancel_event.set()
            return True
        return False

    def _create_result(
        self,
        status: ProcedureStatus,
        data: Any = None,
        error: str | None = None,
        start_time: float = 0.0,
    ) -> ProcedureResult:
        """Create a ProcedureResult."""
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return ProcedureResult(
            status=status,
            data=data,
            error=error,
            duration_ms=duration_ms,
        )
