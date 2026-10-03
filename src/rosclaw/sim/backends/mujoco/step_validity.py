"""Per-data validity guards shared by bounded simulation execution paths.

MuJoCo can warn, reset an unstable state and continue with finite zeros.
Finite positions alone therefore cannot establish a valid rollout. Inspect
per-data warning counters rather than changing process-global warning hooks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FINITE_FIELDS = (
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "act",
    "ctrl",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_constraint",
    "qfrc_bias",
    "qfrc_passive",
    "qfrc_smooth",
    "qfrc_applied",
    "xfrc_applied",
    "sensordata",
)


class SimulationDivergedError(ValueError):
    """Rejected execution with bounded diagnostics, never a successful receipt."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def diagnostic_point(data) -> dict[str, Any]:  # noqa: ANN001
    """JSON-safe actual failure data; truncation is explicit, not silently valid."""

    def scalar(value: float) -> float | str:
        number = float(value)
        if math.isfinite(number):
            return number
        return "NaN" if math.isnan(number) else ("+Inf" if number > 0 else "-Inf")

    result: dict[str, Any] = {
        "t": scalar(data.time),
        "warning_counts": [int(w.number) for w in data.warning],
        "warning_lastinfo": [int(w.lastinfo) for w in data.warning],
        "finite_fields": {},
        "truncated_fields": {},
    }
    for field in FINITE_FIELDS:
        values = np.asarray(getattr(data, field)).reshape(-1)
        result["finite_fields"][field] = bool(np.isfinite(values).all())
        result[field] = [scalar(v) for v in values[:4096]]
        if values.size > 4096:
            result["truncated_fields"][field] = {"recorded": 4096, "actual": int(values.size)}
    return result


def runtime_validation(data, *, steps: int, initial_time: float, timestep: float) -> dict[str, Any]:  # noqa: ANN001
    """Call only after the serial guard has validated every executed step."""
    return {
        "schema_version": "rosclaw.sim.runtime_validation.v1",
        "status": "PASS",
        "method": "serial_each_step",
        "steps_checked": steps,
        "initial_time": initial_time,
        "final_time": float(data.time),
        "actual_elapsed_s": float(data.time) - initial_time,
        "expected_elapsed_s": steps * timestep,
        "warning_counts": [int(w.number) for w in data.warning],
        "time_continuity_checked": True,
        "per_step_warning_check": True,
        "checked_finite_fields": list(FINITE_FIELDS),
    }


def validate_step_data(
    data, *, step: int, expected_time: float | None = None, timestep: float = 0.0
) -> None:  # noqa: ANN001
    """Reject solver warnings, time discontinuities and non-finite dynamics."""
    for index, warning in enumerate(data.warning):
        if int(warning.number) > 0:
            raise ValueError(
                f"SIM_DIVERGED: MuJoCo warning index={index} count={int(warning.number)} "
                f"lastinfo={int(warning.lastinfo)} at step {step}"
            )
    actual_time = float(data.time)
    if not math.isfinite(actual_time):
        raise ValueError(f"SIM_DIVERGED: non-finite time at step {step}")
    if expected_time is not None:
        tolerance = max(1e-12, abs(timestep) * 1e-9, 32 * math.ulp(expected_time))
        if abs(actual_time - expected_time) > tolerance:
            raise ValueError(
                f"SIM_DIVERGED: time discontinuity at step {step}: "
                f"expected {expected_time!r}, observed {actual_time!r}"
            )
    for field in FINITE_FIELDS:
        if not np.isfinite(getattr(data, field)).all():
            raise ValueError(f"SIM_DIVERGED: non-finite {field} at step {step}")
