"""Offline per-joint measured tracking diagnostics, without control authority."""

from __future__ import annotations

from typing import Any

import numpy as np


def joint_motion_metrics(
    position_rad: Any,
    velocity_rad_s: Any,
    target_rad: Any,
    actuator_force_nm: Any,
    torque_limits_nm: Any,
    *,
    sample_interval_s: float,
) -> dict[str, list[float]]:
    """Measure every joint of an already recorded, aligned full window.

    Forces have shape (frames, substeps, joints); their temporal differences
    retain the recorded substep resolution. Position, velocity and target have
    shape (frames, joints). These are diagnostics, not a safety verdict or a
    reward: intended athletic follow-through can have large derivatives.
    No simulator, file, model update or actuator is accessed.
    """
    raw = [
        np.asarray(v)
        for v in (position_rad, velocity_rad_s, target_rad, actuator_force_nm, torque_limits_nm)
    ]
    if any(v.dtype.kind not in "fiu" for v in raw):
        raise ValueError("finite numeric measured joint arrays required")
    q, v, target, force, limits = [np.asarray(x, dtype=np.float64) for x in raw]
    if q.ndim != 2 or force.ndim != 3:
        raise ValueError("aligned frame and substep arrays required")
    n, d = q.shape
    substeps = force.shape[1]
    if (
        not 4 <= n <= 200000
        or not 1 <= d <= 512
        or not 1 <= substeps <= 1000
        or n * substeps * d > 20000000
        or v.shape != q.shape
        or target.shape != q.shape
        or force.shape != (n, substeps, d)
        or limits.shape != (d,)
        or np.any(limits <= 0)
        or type(sample_interval_s) not in (int, float)
        or not np.isfinite(sample_interval_s)
        or not 1e-4 <= sample_interval_s <= 1
        or not all(np.isfinite(x).all() for x in (q, v, target, force, limits))
    ):
        raise ValueError("complete finite aligned joint motion required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):

            def rms(x: np.ndarray) -> list[float]:
                scale = np.max(np.abs(x), axis=0)
                divisor = np.where(scale == 0, 1, scale)
                return (scale * np.sqrt(np.mean((x / divisor) ** 2, axis=0))).tolist()

            f = force.reshape(n * substeps, d)
            normalized = np.abs(f) / limits
            result = {
                "measured_velocity_rms_rad_s": rms(v),
                "measured_acceleration_rms_rad_s2": rms(np.diff(v, axis=0) / sample_interval_s),
                "measured_velocity_jerk_rms_rad_s3": rms(
                    np.diff(v, n=2, axis=0) / sample_interval_s**2
                ),
                "target_velocity_rms_rad_s": rms(np.diff(target, axis=0) / sample_interval_s),
                "target_tracking_error_rms_rad": rms(target - q),
                "actuator_force_rms_nm": rms(f),
                "actuator_force_slew_rms_nm_s": rms(
                    np.diff(f, axis=0) / (sample_interval_s / substeps)
                ),
                "actuator_limit_99pct_fraction": np.mean(normalized >= 0.99, axis=0).tolist(),
                "actuator_limit_maximum_fraction": np.max(normalized, axis=0).tolist(),
            }
    except FloatingPointError as error:
        raise ValueError("derived joint diagnostics must remain finite") from error
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("derived joint diagnostics must remain finite")
    return result
