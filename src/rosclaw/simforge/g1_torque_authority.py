"""Auditable SIM-only authority projection for G1 torque controllers.

The projection sits between a composed controller and MuJoCo's actuator
input. It never erases the original proposal: callers retain the command and
record every correction so later Growth cycles can reduce projection reliance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class G1TorqueAuthorityProjection:
    """One finite 29-DoF authority projection observation."""

    projected_torque_nm: np.ndarray
    correction_nm: np.ndarray
    active: bool
    projected_joint_count: int
    preprojection_peak_demand_ratio: float
    projected_peak_demand_ratio: float


@dataclass(frozen=True)
class G1AdditiveTorqueAuthorityProjection:
    """Direction-preserving projection for a task-space decoded torque."""

    projected_additive_torque_nm: np.ndarray
    scale: float
    active: bool


def project_g1_torque_authority(
    *,
    commanded_torque_nm: np.ndarray,
    hard_limits_nm: np.ndarray,
    maximum_demand_ratio: float,
) -> G1TorqueAuthorityProjection:
    """Project a finite G1 command into a declared fraction of hard limits."""

    command = np.asarray(commanded_torque_nm, dtype=np.float64)
    limits = np.asarray(hard_limits_nm, dtype=np.float64)
    if (
        command.shape != (29,)
        or limits.shape != (29,)
        or not np.all(np.isfinite(command))
        or not np.all(np.isfinite(limits))
        or np.any(limits <= 0.0)
    ):
        raise ValueError("G1 torque authority projection requires finite 29-DoF vectors")
    if not math.isfinite(maximum_demand_ratio) or not 0.90 <= maximum_demand_ratio <= 0.99:
        raise ValueError("G1 torque authority ratio must be in [0.90, 0.99]")

    bound = limits * maximum_demand_ratio
    projected = np.clip(command, -bound, bound)
    correction = projected - command
    changed = np.abs(correction) > 1e-12
    return G1TorqueAuthorityProjection(
        projected_torque_nm=projected,
        correction_nm=correction,
        active=bool(np.any(changed)),
        projected_joint_count=int(np.count_nonzero(changed)),
        preprojection_peak_demand_ratio=float(np.max(np.abs(command) / limits)),
        projected_peak_demand_ratio=float(np.max(np.abs(projected) / limits)),
    )


def project_g1_additive_torque_authority(
    *,
    parent_torque_nm: np.ndarray,
    additive_torque_nm: np.ndarray,
    hard_limits_nm: np.ndarray,
    maximum_demand_ratio: float,
) -> G1AdditiveTorqueAuthorityProjection:
    """Scale one decoded task torque uniformly into remaining authority.

    A common scale preserves the requested task-space force direction. This is
    preferable to clipping individual Jacobian-transpose joint components,
    which can turn a vertical/lateral football impulse into another wrench.
    Parent over-demand is intentionally not hidden and remains the caller's
    responsibility for the final audited projection.
    """

    parent = np.asarray(parent_torque_nm, dtype=np.float64)
    additive = np.asarray(additive_torque_nm, dtype=np.float64)
    limits = np.asarray(hard_limits_nm, dtype=np.float64)
    if (
        parent.shape != (29,)
        or additive.shape != (29,)
        or limits.shape != (29,)
        or not np.all(np.isfinite(parent))
        or not np.all(np.isfinite(additive))
        or not np.all(np.isfinite(limits))
        or np.any(limits <= 0.0)
    ):
        raise ValueError("G1 additive authority projection requires finite 29-DoF vectors")
    if not math.isfinite(maximum_demand_ratio) or not 0.90 <= maximum_demand_ratio <= 0.99:
        raise ValueError("G1 additive authority ratio must be in [0.90, 0.99]")

    bound = limits * maximum_demand_ratio
    scale = 1.0
    positive = additive > 1e-12
    negative = additive < -1e-12
    if np.any(positive):
        scale = min(scale, float(np.min((bound[positive] - parent[positive]) / additive[positive])))
    if np.any(negative):
        scale = min(
            scale,
            float(np.min((-bound[negative] - parent[negative]) / additive[negative])),
        )
    scale = float(np.clip(scale, 0.0, 1.0))
    return G1AdditiveTorqueAuthorityProjection(
        projected_additive_torque_nm=additive * scale,
        scale=scale,
        active=bool(scale < 1.0 - 1e-12 and np.any(np.abs(additive) > 1e-12)),
    )


__all__ = [
    "G1AdditiveTorqueAuthorityProjection",
    "G1TorqueAuthorityProjection",
    "project_g1_additive_torque_authority",
    "project_g1_torque_authority",
]
