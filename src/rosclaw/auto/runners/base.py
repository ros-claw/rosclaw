"""Base runner interface for experiment execution."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunnerResult:
    """Result of running an experiment."""
    success: bool = False
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    safety_violations: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "metrics": self.metrics,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "safety_violations": self.safety_violations,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunnerResult":
        return cls(
            success=d.get("success", False),
            metrics=d.get("metrics", {}),
            logs=d.get("logs", []),
            artifacts=d.get("artifacts", {}),
            safety_violations=d.get("safety_violations", []),
            error=d.get("error", ""),
        )


class BaseRunner(ABC):
    """Abstract base for all experiment runners."""

    name: str = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def run(self, experiment_spec: Any) -> RunnerResult:
        """Execute the experiment and return results."""
        ...

    @abstractmethod
    def health(self) -> dict:
        """Return runner health status."""
        ...

    def validate_safety(self, experiment_spec: Any) -> list[str]:
        """Check safety protocol compliance; return list of violations.

        Local runner is allowed to run sandbox_required experiments for
        fast smoke testing.  Sandbox and Darwin runners enforce stricter
        checks.
        """
        safety = getattr(experiment_spec, "safety", {}) if hasattr(experiment_spec, "safety") else {}
        violations = []
        # Only non-local runners reject sandbox_required mismatch
        if safety.get("sandbox_required", False) and self.name not in ("sandbox", "darwin"):
            if self.name != "local":
                violations.append("Sandbox required but running on non-sandbox runner")
        if safety.get("max_force", 999) < 5:
            violations.append("Max force limit too restrictive for meaningful experiment")
        return violations
