"""Per-data validity guards shared by bounded simulation execution paths.

MuJoCo can warn, reset an unstable state and continue with finite zeros.
Finite positions alone therefore cannot establish a valid rollout. Inspect
per-data warning counters rather than changing process-global warning hooks.
"""

from __future__ import annotations

import math

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
