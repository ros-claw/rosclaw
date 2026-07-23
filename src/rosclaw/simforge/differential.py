"""Label-based cross-simulator comparison without exact-state assumptions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class DisagreementStatus(StrEnum):
    AGREE = "AGREE"
    NUMERIC_VARIATION = "NUMERIC_VARIATION"
    CRITICAL_DISAGREEMENT = "CRITICAL_DISAGREEMENT"


@dataclass(frozen=True)
class SimulatorLabels:
    safe: bool
    collision: bool
    success: bool
    stopped: bool
    task_verified: bool
    clearance_category: str
    failure_signature: str = ""
    final_error_m: float | None = None
    peak_force_n: float | None = None
    minimum_clearance_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("safe", "collision", "success", "stopped", "task_verified"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"simulator label {name} must be boolean")
        if not self.clearance_category:
            raise ValueError("clearance_category cannot be empty")
        for name in ("final_error_m", "peak_force_n", "minimum_clearance_m"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"simulator numeric label {name} must be finite")


@dataclass(frozen=True)
class SimulatorDisagreementEvent:
    scenario_id: str
    baseline_backend: str
    comparison_backend: str
    status: DisagreementStatus
    critical_fields: tuple[str, ...]
    numeric_deltas: tuple[tuple[str, float], ...]
    promotion_blocked: bool


def compare_simulators(
    *,
    scenario_id: str,
    baseline_backend: str,
    comparison_backend: str,
    baseline: SimulatorLabels,
    comparison: SimulatorLabels,
) -> SimulatorDisagreementEvent:
    critical = tuple(
        name
        for name in (
            "safe",
            "collision",
            "success",
            "stopped",
            "task_verified",
            "clearance_category",
            "failure_signature",
        )
        if getattr(baseline, name) != getattr(comparison, name)
    )
    deltas: list[tuple[str, float]] = []
    for name in ("final_error_m", "peak_force_n", "minimum_clearance_m"):
        left = getattr(baseline, name)
        right = getattr(comparison, name)
        if left is not None and right is not None:
            deltas.append((name, float(right) - float(left)))
    if critical:
        status = DisagreementStatus.CRITICAL_DISAGREEMENT
    elif any(abs(value) > 1e-12 for _, value in deltas):
        status = DisagreementStatus.NUMERIC_VARIATION
    else:
        status = DisagreementStatus.AGREE
    return SimulatorDisagreementEvent(
        scenario_id=scenario_id,
        baseline_backend=baseline_backend,
        comparison_backend=comparison_backend,
        status=status,
        critical_fields=critical,
        numeric_deltas=tuple(deltas),
        promotion_blocked=bool(critical),
    )


__all__ = [
    "DisagreementStatus",
    "SimulatorDisagreementEvent",
    "SimulatorLabels",
    "compare_simulators",
]
