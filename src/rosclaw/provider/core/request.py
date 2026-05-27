"""ROSClaw Provider - Canonical request envelope.

All provider invocations flow through ProviderRequest.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderRequest:
    """Standard request envelope for all ROSClaw provider invocations.

    Attributes:
        request_id: Unique identifier for tracing.
        capability: Capability name (e.g., "vlm.object_grounding").
        inputs: Input data dict (modality-specific).
        context: Execution context (robot, scene, task_id, etc.).
        constraints: Latency, safety_level, offline, cost, etc.
        output_schema: Optional JSON schema for structured output validation.
    """

    request_id: str
    capability: str
    inputs: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.capability:
            raise ValueError("capability is required")

    @property
    def robot_id(self) -> str:
        return self.context.get("robot", "")

    @property
    def safety_level(self) -> str:
        return self.constraints.get("safety_level", "MODERATE")

    @property
    def latency_budget_ms(self) -> int | None:
        return self.constraints.get("latency_ms")

    @property
    def requires_offline(self) -> bool:
        return self.constraints.get("offline", False)
