"""
ROSClaw Lifecycle Management

Provides standardized lifecycle states for all ROSClaw modules:
- UNINITIALIZED -> INITIALIZING -> READY -> RUNNING -> SHUTTING_DOWN -> STOPPED

Every grounding engine (firewall, memory, practice, swarm, etc.)
implementes the LifecycleMixin for consistent startup/shutdown.
"""

from enum import Enum, auto


class LifecycleState(Enum):
    """Standard lifecycle states for ROSClaw modules."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()
    ERROR = auto()


class LifecycleMixin:
    """
    Mixin providing standardized lifecycle management.

    All ROSClaw modules should inherit from this to ensure
    consistent initialization and shutdown behavior.
    """

    def __init__(self) -> None:
        self._lifecycle_state = LifecycleState.UNINITIALIZED
        self._error_message: str | None = None

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state."""
        return self._lifecycle_state

    @property
    def is_ready(self) -> bool:
        """Check if module is ready for operation."""
        return self._lifecycle_state in (LifecycleState.READY, LifecycleState.RUNNING)

    @property
    def is_running(self) -> bool:
        """Check if module is actively running."""
        return self._lifecycle_state == LifecycleState.RUNNING

    @property
    def error_message(self) -> str | None:
        """Error message if in ERROR state."""
        return self._error_message

    def initialize(self) -> None:
        """Initialize the module. Override in subclass."""
        if self._lifecycle_state not in (
            LifecycleState.UNINITIALIZED,
            LifecycleState.ERROR,
            LifecycleState.STOPPED,
        ):
            raise RuntimeError(
                f"Cannot initialize: already in state {self._lifecycle_state.name}. "
                f"Call stop() first if re-initialization is needed."
            )
        self._lifecycle_state = LifecycleState.INITIALIZING
        self._error_message = None
        try:
            self._do_initialize()
            self._lifecycle_state = LifecycleState.READY
        except Exception as e:
            self._lifecycle_state = LifecycleState.ERROR
            self._error_message = str(e)
            raise

    def start(self) -> None:
        """Start the module. Must be initialized first."""
        if self._lifecycle_state == LifecycleState.RUNNING:
            return
        if self._lifecycle_state not in (LifecycleState.READY, LifecycleState.PAUSED):
            raise RuntimeError(f"Cannot start from state {self._lifecycle_state.name}")
        self._lifecycle_state = LifecycleState.RUNNING
        self._do_start()

    def pause(self) -> None:
        """Pause the module."""
        if self._lifecycle_state == LifecycleState.RUNNING:
            self._lifecycle_state = LifecycleState.PAUSED
            self._do_pause()

    def resume(self) -> None:
        """Resume from paused state."""
        if self._lifecycle_state == LifecycleState.PAUSED:
            self._lifecycle_state = LifecycleState.RUNNING
            self._do_resume()

    def stop(self) -> None:
        """Stop the module gracefully."""
        if self._lifecycle_state in (
            LifecycleState.RUNNING,
            LifecycleState.PAUSED,
            LifecycleState.READY,
        ):
            self._lifecycle_state = LifecycleState.SHUTTING_DOWN
            try:
                self._do_stop()
            finally:
                self._lifecycle_state = LifecycleState.STOPPED

    # --- Override these in subclasses ---

    def _do_initialize(self) -> None:
        """Module-specific initialization logic."""

    def _do_start(self) -> None:
        """Module-specific start logic."""

    def _do_pause(self) -> None:
        """Module-specific pause logic."""

    def _do_resume(self) -> None:
        """Module-specific resume logic."""

    def _do_stop(self) -> None:
        """Module-specific stop logic."""
