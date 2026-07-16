"""Runtime state machine for the persistent LeRobot policy runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RuntimeStateValue = Literal[
    "idle",
    "starting",
    "ready",
    "busy",
    "degraded",
    "error",
    "stopped",
]


@dataclass
class RuntimeState:
    """Mutable state snapshot for a persistent policy runtime process."""

    state: RuntimeStateValue = "idle"
    pid: int | None = None
    policy_path: str | None = None
    policy_loaded: bool = False
    policy_metadata: dict[str, Any] = field(default_factory=dict)
    active_session_id: str | None = None
    worker_generation: int = 0
    error: str | None = None
    last_health_at: float | None = None

    def transition(self, new_state: RuntimeStateValue, error: str | None = None) -> None:
        """Move the runtime to a new state, optionally recording an error."""
        self.state = new_state
        if error is not None:
            self.error = error

    def is_operational(self) -> bool:
        """Return True if the runtime can accept requests."""
        return self.state in {"ready", "busy", "degraded"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "pid": self.pid,
            "policy_path": self.policy_path,
            "policy_loaded": self.policy_loaded,
            "active_session_id": self.active_session_id,
            "worker_generation": self.worker_generation,
            "error": self.error,
            "last_health_at": self.last_health_at,
        }
