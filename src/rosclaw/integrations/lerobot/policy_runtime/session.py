"""Episode-scoped session state for the persistent LeRobot policy runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicySession:
    """A single policy episode session."""

    session_id: str
    policy_path: str
    body_id: str | None = None
    created_at: float = 0.0
    last_step_index: int = -1
    context: dict[str, Any] = field(default_factory=dict)

    def bump_step(self) -> int:
        """Advance the step counter and return the new step index."""
        self.last_step_index += 1
        return self.last_step_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "policy_path": self.policy_path,
            "body_id": self.body_id,
            "created_at": self.created_at,
            "last_step_index": self.last_step_index,
            "context": self.context,
        }
