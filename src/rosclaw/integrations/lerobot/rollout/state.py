"""Rollout state and result models for P4 proposal/shadow loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RolloutMode(StrEnum):
    """Supported rollout modes."""

    PROPOSAL_ONLY = "proposal_only"
    SHADOW = "shadow"
    EXECUTE = "execute"


class RolloutStopReason(StrEnum):
    """Reason a rollout loop terminated."""

    COMPLETED = "completed"
    RUNTIME_FAILURE = "runtime_failure"
    STALE_OBSERVATION = "stale_observation"
    NAN_INF = "nan_inf"
    INCOMPATIBLE_MAPPING = "incompatible_mapping"
    SANDBOX_BLOCK = "sandbox_block"
    DEADLINE_MISS = "deadline_miss"
    INTERRUPTED = "interrupted"
    POLICY_LOAD_FAILED = "policy_load_failed"
    OBSERVATION_REQUIRED_MISSING = "observation_required_missing"


@dataclass
class RolloutResult:
    """Result of a proposal-only or shadow rollout."""

    mode: RolloutMode
    stop_reason: RolloutStopReason
    steps_completed: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    mapped_actions: list[dict[str, Any]] = field(default_factory=list)
    sandbox_decisions: list[dict[str, Any]] = field(default_factory=list)
    trace_path: str | None = None
    practice_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    hardware_actions_executed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "stop_reason": self.stop_reason.value,
            "steps_completed": self.steps_completed,
            "proposal_count": len(self.proposals),
            "mapped_action_count": len(self.mapped_actions),
            "sandbox_decision_count": len(self.sandbox_decisions),
            "trace_path": self.trace_path,
            "practice_id": self.practice_id,
            "metrics": self.metrics,
            "hardware_actions_executed": self.hardware_actions_executed,
            "errors": self.errors,
            "warnings": self.warnings,
        }
