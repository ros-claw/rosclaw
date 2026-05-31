"""ROSClaw Provider - Observability / Trace.

Lightweight trace collector for provider invocation chains.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rosclaw.provider.core.trace")


@dataclass
class TraceStep:
    """Single step in a provider trace."""

    name: str
    provider: str
    capability: str
    latency_ms: int
    status: str  # "success" | "failed" | "blocked" | "fallback"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderTrace:
    """Full trace for a provider invocation chain.

    Example:
        trace = ProviderTrace(task_id="task_pick_red_cup")
        trace.add_step("locate_object", "grounding_sam", "vlm.object_grounding", 188, "success")
        trace.add_guard_result(blocked=False, checks=["schema", "collision"])
        logger.info("%s", trace.to_dict())
    """

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def add_step(
        self,
        name: str,
        provider: str,
        capability: str,
        latency_ms: int,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            TraceStep(
                name=name,
                provider=provider,
                capability=capability,
                latency_ms=latency_ms,
                status=status,
                metadata=metadata or {},
            )
        )

    def add_guard_result(
        self,
        blocked: bool,
        checks: list[dict[str, Any]] | None = None,
        reason: str = "",
    ) -> None:
        self.guard = {
            "blocked": blocked,
            "checks": checks or [],
            "reason": reason,
        }

    @property
    def total_latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "total_latency_ms": self.total_latency_ms,
            "guard": self.guard,
            "steps": [
                {
                    "name": s.name,
                    "provider": s.provider,
                    "capability": s.capability,
                    "latency_ms": s.latency_ms,
                    "status": s.status,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
        }
