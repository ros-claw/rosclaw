"""Generic physical feedback frame for embodied agents.

A PhysicalFeedbackFrame is a body-agnostic snapshot of one telemetry sample:
target/actual state, force readings, safety channels, and inferred contact
events. It is intentionally not RH56-specific so it can be reused for any
end-effector or body that exposes per-DOF force/status/temperature data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PhysicalFeedbackFrame:
    """A single body-agnostic telemetry frame.

    All per-DOF maps use the body's native DOF names as keys (e.g. "thumb",
    "index", "joint_0"). Values are floats or arbitrary diagnostic payloads.
    """

    frame_id: str = ""
    body_id: str = ""
    timestamp: float = 0.0

    # Kinematic state
    target: Dict[str, float] = field(default_factory=dict)
    actual: Dict[str, float] = field(default_factory=dict)
    position_error: Dict[str, float] = field(default_factory=dict)

    # Force pipeline (units are body-specific; RH56 uses grams-force)
    force_raw: Dict[str, Optional[float]] = field(default_factory=dict)
    force_baseline: Dict[str, Optional[float]] = field(default_factory=dict)
    force_net: Dict[str, Optional[float]] = field(default_factory=dict)
    force_delta: Dict[str, Optional[float]] = field(default_factory=dict)
    force_derivative: Dict[str, Optional[float]] = field(default_factory=dict)

    # Auxiliary telemetry
    current: Dict[str, Optional[float]] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)
    error: Dict[str, Any] = field(default_factory=dict)
    temperature: Dict[str, Optional[float]] = field(default_factory=dict)

    # Inferred semantics
    primary_event: str = "unknown"
    secondary_tags: List[str] = field(default_factory=list)
    serial_timeout: bool = False

    # Extension point for body-specific fields (e.g. voltage, encoder ticks)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict; safe for JSONL/Parquet."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicalFeedbackFrame":
        """Restore from a plain dict."""
        return cls(**data)

    def dof_names(self) -> List[str]:
        """Return the union of DOF names seen across all per-DOF maps."""
        names: set[str] = set()
        for d in (
            self.target,
            self.actual,
            self.position_error,
            self.force_raw,
            self.force_net,
            self.current,
            self.status,
            self.error,
            self.temperature,
        ):
            names.update(d.keys())
        return sorted(names)
