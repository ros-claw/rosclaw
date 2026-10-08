"""Offline, same-observation diagnostics for a bounded residual policy update.

No policy, runtime, simulator, transport or actuator is instantiated. Callers
authenticate observations, nominal targets, constraints and policy lineage.
The measured previous residual is held fixed for BOTH local projections.
This is not a counterfactual trajectory or an action-authorization surface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MAX_ROWS = 65536
MAX_DIMENSIONS = 64
MAX_COORDINATES = 1048576


def bounded_residual_update_transport(
    behavior_latent: Any,
    candidate_latent: Any,
    observed_previous_residual: Any,
    nominal_target: Any,
    measured_behavior_residual: Any,
    action_limits: Any,
    *,
    residual_cap: float,
    slew_cap: float,
) -> dict[str, Any]:
    """Inspect cap*tanh -> slew -> zero-inclusive nominal residual envelope.

    All rows are retained, including failures; no sampling or optimizer runs.
    Arrays have shape (observed rows, dimensions), with fixed (dimensions,2)
    absolute action limits. Only summary statistics are returned, not targets.
    The source behavior projection must match measured residuals EXACTLY.
    Scalar caps are in the caller's action units, never assumed to be torques.
    """
    if (
        type(residual_cap) not in (float, int)
        or type(slew_cap) not in (float, int)
        or not np.isfinite([residual_cap, slew_cap]).all()
        or not 0 < slew_cap <= residual_cap <= 1e6
    ):
        raise ValueError("finite positive declared residual and slew caps required")
    raw = [
        np.asarray(value)
        for value in (
            behavior_latent,
            candidate_latent,
            observed_previous_residual,
            nominal_target,
            measured_behavior_residual,
            action_limits,
        )
    ]
    shape = raw[0].shape
    if (
        len(shape) != 2
        or not 1 <= shape[0] <= MAX_ROWS
        or not 1 <= shape[1] <= MAX_DIMENSIONS
        or shape[0] * shape[1] > MAX_COORDINATES
        or any(value.shape != shape for value in raw[:5])
        or raw[5].shape != (shape[1], 2)
        or any(
            value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
            or np.any(np.abs(value.astype(np.float64)) > 1e6)
            for value in raw
        )
    ):
        raise ValueError("complete aligned bounded finite numeric rows required")
    old, new, previous, nominal, measured, limits = [
        value.astype(np.float64, copy=False) for value in raw
    ]
    if (
        np.any(limits[:, 0] >= limits[:, 1])
        or np.max(np.abs(previous)) > residual_cap + 1e-5
        or np.max(np.abs(measured)) > residual_cap + 1e-5
    ):
        raise ValueError("ordered absolute limits and measured residual envelope required")
    lower = np.maximum(np.minimum(0.0, limits[:, 0] - nominal), -residual_cap)
    upper = np.minimum(np.maximum(0.0, limits[:, 1] - nominal), residual_cap)
    desired_old, desired_new = residual_cap * np.tanh(old), residual_cap * np.tanh(new)
    proposed_old = previous + np.clip(desired_old - previous, -slew_cap, slew_cap)
    proposed_new = previous + np.clip(desired_new - previous, -slew_cap, slew_cap)
    projected_old, projected_new = (
        np.clip(proposed_old, lower, upper),
        np.clip(proposed_new, lower, upper),
    )
    if not np.array_equal(projected_old, measured):
        raise ValueError("source behavior residuals do not EXACTLY match the declared projection")
    intended = desired_new - desired_old
    transported = projected_new - projected_old
    changed = intended != 0
    magnitude = float(np.linalg.norm(intended))
    digest = hashlib.sha256()
    for value in raw:
        digest.update(json.dumps([value.shape, value.dtype.str], separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "schema": "rosclaw.growth.offline_bounded_residual_update_transport.v1",
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_numeric_hash": "sha256:" + digest.hexdigest(),
        "row_count": shape[0],
        "dimensions": shape[1],
        "coordinate_rows": int(intended.size),
        "residual_cap": residual_cap,
        "slew_cap": slew_cap,
        "same_observed_previous_residual_for_both_projections": True,
        "source_numeric_projection_exact": True,
        "all_supplied_rows_preserved": True,
        "intended_changed_coordinates": int(changed.sum()),
        "changed_coordinates_fully_masked": int(np.sum(changed & (transported == 0))),
        "intended_update_rms": float(np.sqrt(np.mean(intended**2))),
        "transported_update_rms": float(np.sqrt(np.mean(transported**2))),
        "intended_update_max_abs": float(np.max(np.abs(intended))),
        "transported_update_max_abs": float(np.max(np.abs(transported))),
        "transported_to_intended_l2_ratio": float(np.linalg.norm(transported)) / magnitude
        if magnitude
        else None,
        "behavior_slew_active_coordinates": int(np.sum(np.abs(desired_old - previous) > slew_cap)),
        "candidate_slew_active_coordinates": int(np.sum(np.abs(desired_new - previous) > slew_cap)),
        "behavior_limit_active_coordinates": int(
            np.sum((proposed_old < lower) | (proposed_old > upper))
        ),
        "candidate_limit_active_coordinates": int(
            np.sum((proposed_new < lower) | (proposed_new > upper))
        ),
        "execution_ceiling": "OFFLINE_DIAGNOSTIC_NO_EXECUTION",
        "source_physics_authenticated_here": False,
        "counterfactual_dynamics_claimed": False,
        "policy_gain_verified": False,
        "optimizer_steps": 0,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
