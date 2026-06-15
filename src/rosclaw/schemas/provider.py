"""Provider request/response schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderRequest:
    """Capability request routed through rosclaw-provider."""

    request_id: str = ""
    capability: str = ""  # vlm.object_grounding | skill.grasp | critic.success_detection
    task_id: str = ""
    robot_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    priority: str = "NORMAL"
    timeout_ms: int = 30000

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability": self.capability,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "payload": self.payload,
            "priority": self.priority,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProviderRequest:
        return cls(
            request_id=d.get("request_id", ""),
            capability=d.get("capability", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            payload=dict(d.get("payload", {})),
            priority=d.get("priority", "NORMAL"),
            timeout_ms=d.get("timeout_ms", 30000),
        )


@dataclass
class ProviderResponse:
    """Capability response from rosclaw-provider."""

    request_id: str = ""
    capability: str = ""
    status: str = "success"  # success | failure | timeout | blocked
    result: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    confidence: float = 0.0
    risk_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability": self.capability,
            "status": self.status,
            "result": self.result,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "risk_estimate": self.risk_estimate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProviderResponse:
        return cls(
            request_id=d.get("request_id", ""),
            capability=d.get("capability", ""),
            status=d.get("status", "success"),
            result=dict(d.get("result", {})),
            latency_ms=d.get("latency_ms", 0),
            confidence=d.get("confidence", 0.0),
            risk_estimate=d.get("risk_estimate", 0.0),
        )
