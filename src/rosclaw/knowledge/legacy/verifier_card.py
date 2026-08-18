"""VerifierCard — evaluation metric prior for Know module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifierCard:
    """Verifier prior: metric definition, threshold, validation method."""

    verifier_id: str = ""
    metric_name: str = ""
    threshold: float = 0.0
    validation_method: str = ""  # simulation | real_robot | hybrid
    objective_direction: str = "maximize"
    required_episodes: int = 50
    safety_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_id": self.verifier_id,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "validation_method": self.validation_method,
            "objective_direction": self.objective_direction,
            "required_episodes": self.required_episodes,
            "safety_checks": self.safety_checks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifierCard:
        return cls(
            verifier_id=d.get("verifier_id", ""),
            metric_name=d.get("metric_name", ""),
            threshold=d.get("threshold", 0.0),
            validation_method=d.get("validation_method", ""),
            objective_direction=d.get("objective_direction", "maximize"),
            required_episodes=d.get("required_episodes", 50),
            safety_checks=list(d.get("safety_checks", [])),
        )
