"""Evidence-gated execution of immutable recovery patches."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from rosclaw.core.event_bus import Event, EventBus, EventPriority

ALLOWED_PATCH_KEYS = frozenset(
    {
        "speed_scale",
        "acceleration_scale",
        "controller_gain",
        "safety_clearance",
        "waypoint_offset",
        "timeout_sec",
        "retry_count",
        "gripper_force_limit",
        "approach_direction",
    }
)


class RetrySubmitter(Protocol):
    def __call__(self, action: dict[str, Any]) -> dict[str, Any]: ...


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class CandidatePatch:
    """Immutable, hash-addressed parameter overlay."""

    parameters: tuple[tuple[str, Any], ...]
    patch_hash: str

    @classmethod
    def create(cls, values: dict[str, Any]) -> CandidatePatch:
        unknown = sorted(set(values) - ALLOWED_PATCH_KEYS)
        if unknown:
            raise ValueError(f"RETRY_PATCH_KEY_FORBIDDEN: {', '.join(unknown)}")
        if not values:
            raise ValueError("RETRY_PATCH_EMPTY")
        normalized = copy.deepcopy(values)
        return cls(tuple(sorted(normalized.items())), _canonical_hash(normalized))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.parameters))


@dataclass(frozen=True)
class RetryExecutionResult:
    executed: bool
    retry_action: dict[str, Any] | None
    receipt: dict[str, Any] | None
    parameter_patch_hash: str
    reason: str


class RetryOrchestrator:
    """Apply one safe patch and submit a real same-scenario retry."""

    def __init__(self, submitter: RetrySubmitter, event_bus: EventBus | None = None) -> None:
        self._submitter = submitter
        self._bus = event_bus

    def execute(
        self,
        original_action: dict[str, Any],
        patch_values: dict[str, Any],
        *,
        retry_budget: int,
        retry_lineage: list[str] | None = None,
        prior_patch_hashes: list[str] | None = None,
    ) -> RetryExecutionResult:
        patch = CandidatePatch.create(patch_values)
        lineage = list(retry_lineage or [])
        if retry_budget <= 0 or len(lineage) >= retry_budget:
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "RETRY_BUDGET_EXHAUSTED"
            )
        if patch.patch_hash in set(prior_patch_hashes or []):
            return RetryExecutionResult(False, None, None, patch.patch_hash, "REPEATED_PATCH")

        scenario = original_action.get("scenario")
        if (
            not isinstance(scenario, dict)
            or "scenario_id" not in scenario
            or "seed" not in scenario
        ):
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "SCENARIO_AND_SEED_REQUIRED"
            )

        candidate = copy.deepcopy(original_action)
        original_id = str(original_action.get("action_id") or "unknown")
        lineage = [original_id, *lineage]
        candidate["action_id"] = (
            f"retry_{_canonical_hash([original_id, lineage, patch.patch_hash])[7:23]}"
        )
        candidate["scenario"] = copy.deepcopy(scenario)
        parameters = candidate.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "ACTION_PARAMETERS_INVALID"
            )
        parameters.update(patch.to_dict())
        candidate["retry_budget"] = retry_budget
        candidate["retry_lineage"] = lineage
        candidate["parameter_patch_hash"] = patch.patch_hash

        receipt = self._submitter(candidate)
        if not isinstance(receipt, dict):
            return RetryExecutionResult(
                False, candidate, None, patch.patch_hash, "RETRY_RECEIPT_MISSING"
            )
        same_scenario = (
            receipt.get("scenario_id") == scenario["scenario_id"]
            and receipt.get("seed") == scenario["seed"]
        )
        reason = "RETRY_EXECUTED" if same_scenario else "RETRY_SCENARIO_DRIFT"
        if self._bus is not None:
            self._bus.publish(
                Event(
                    topic="rosclaw.how.retry.completed",
                    payload={
                        "request_id": original_id,
                        "retry_action_id": candidate["action_id"],
                        "retry_lineage": lineage,
                        "parameter_patch_hash": patch.patch_hash,
                        "result": receipt.get("is_safe"),
                        "reason": reason,
                        "physics_executed": receipt.get("physics_executed", False),
                        "evidence_domain": receipt.get("evidence_domain"),
                    },
                    source="retry_orchestrator",
                    priority=EventPriority.NORMAL,
                )
            )
        return RetryExecutionResult(same_scenario, candidate, receipt, patch.patch_hash, reason)


__all__ = [
    "ALLOWED_PATCH_KEYS",
    "CandidatePatch",
    "RetryExecutionResult",
    "RetryOrchestrator",
]
