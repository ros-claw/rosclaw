"""Evidence-gated execution of immutable recovery patches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.sandbox.evidence import (
    SimulationEvidenceVerification,
    verify_simulation_receipt,
)

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
MAX_RETRY_BUDGET = 10
MAX_RETRY_LINEAGE = 10


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
        _validate_patch_values(normalized)
        try:
            patch_hash = _canonical_hash(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError("RETRY_PATCH_NOT_JSON_SERIALIZABLE") from exc
        return cls(tuple(sorted(normalized.items())), patch_hash)

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
    """Apply one bounded patch to a simulation-only same-scenario retry."""

    def __init__(
        self,
        submitter: RetrySubmitter,
        event_bus: EventBus | None = None,
        *,
        receipt_verifier: Callable[[dict[str, Any]], SimulationEvidenceVerification] | None = None,
    ) -> None:
        self._submitter = submitter
        self._bus = event_bus
        self._receipt_verifier = receipt_verifier or verify_simulation_receipt

    def execute(
        self,
        original_action: dict[str, Any],
        patch_values: dict[str, Any],
        *,
        retry_budget: int,
        retry_lineage: list[str] | None = None,
        prior_patch_hashes: list[str] | None = None,
    ) -> RetryExecutionResult:
        if not isinstance(original_action, dict):
            raise ValueError("RETRY_ACTION_INVALID")
        if (
            isinstance(retry_budget, bool)
            or not isinstance(retry_budget, int)
            or not 1 <= retry_budget <= MAX_RETRY_BUDGET
        ):
            raise ValueError("RETRY_BUDGET_INVALID")
        lineage = _validated_string_list(
            retry_lineage, name="RETRY_LINEAGE", maximum=MAX_RETRY_LINEAGE
        )
        previous_hashes = _validated_string_list(
            prior_patch_hashes, name="PRIOR_PATCH_HASHES", maximum=MAX_RETRY_LINEAGE
        )
        patch = CandidatePatch.create(patch_values)
        if len(lineage) >= retry_budget:
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "RETRY_BUDGET_EXHAUSTED"
            )
        if patch.patch_hash in set(previous_hashes):
            return RetryExecutionResult(False, None, None, patch.patch_hash, "REPEATED_PATCH")

        scenario = original_action.get("scenario")
        execution_mode = str(
            original_action.get("execution_mode") or original_action.get("mode") or ""
        ).upper()
        if execution_mode != "SIMULATION":
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "RETRY_SIMULATION_ONLY"
            )
        if not isinstance(scenario, dict):
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "SCENARIO_AND_SEED_REQUIRED"
            )
        scenario_id = scenario.get("scenario_id")
        seed = scenario.get("seed")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id.strip()
            or len(scenario_id) > 256
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < 2**63
        ):
            return RetryExecutionResult(
                False, None, None, patch.patch_hash, "SCENARIO_AND_SEED_INVALID"
            )

        candidate = copy.deepcopy(original_action)
        raw_action_id = original_action.get("action_id")
        original_id = (
            raw_action_id
            if isinstance(raw_action_id, str) and 0 < len(raw_action_id) <= 256
            else "unknown"
        )
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
        evidence_verified = False
        if same_scenario:
            try:
                evidence_verified = self._receipt_verifier(receipt).verified
            except Exception:  # noqa: BLE001 - recovery must fail closed
                evidence_verified = False
        executed = same_scenario and evidence_verified
        if not same_scenario:
            reason = "RETRY_SCENARIO_DRIFT"
        elif not evidence_verified:
            reason = "RETRY_EVIDENCE_UNVERIFIED"
        else:
            reason = "RETRY_EXECUTED"
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
        return RetryExecutionResult(executed, candidate, receipt, patch.patch_hash, reason)


def _validated_string_list(value: list[str] | None, *, name: str, maximum: int) -> list[str]:
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(item, str) or not item or len(item) > 256 for item in value)
    ):
        raise ValueError(f"{name}_INVALID")
    return list(value)


def _finite_number(value: Any, *, minimum: float, maximum: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"RETRY_PATCH_VALUE_INVALID:{name}")


def _validate_patch_values(values: dict[str, Any]) -> None:
    numeric_bounds = {
        "speed_scale": (0.01, 1.0),
        "acceleration_scale": (0.01, 1.0),
        "controller_gain": (0.01, 10.0),
        "safety_clearance": (0.0, 1.0),
        "timeout_sec": (0.1, 3_600.0),
        "gripper_force_limit": (0.0, 1_000.0),
    }
    for name, (minimum, maximum) in numeric_bounds.items():
        if name in values:
            _finite_number(values[name], minimum=minimum, maximum=maximum, name=name)
    if "retry_count" in values:
        retry_count = values["retry_count"]
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or not 0 <= retry_count <= 10
        ):
            raise ValueError("RETRY_PATCH_VALUE_INVALID:retry_count")
    if "waypoint_offset" in values:
        offset = values["waypoint_offset"]
        if not isinstance(offset, list) or not 1 <= len(offset) <= 12:
            raise ValueError("RETRY_PATCH_VALUE_INVALID:waypoint_offset")
        for component in offset:
            _finite_number(
                component,
                minimum=-0.25,
                maximum=0.25,
                name="waypoint_offset",
            )
    if "approach_direction" in values and values["approach_direction"] not in {
        "direct",
        "home_clearance_route",
        "retreat_then_reapproach",
    }:
        raise ValueError("RETRY_PATCH_VALUE_INVALID:approach_direction")


__all__ = [
    "ALLOWED_PATCH_KEYS",
    "CandidatePatch",
    "RetryExecutionResult",
    "RetryOrchestrator",
]
