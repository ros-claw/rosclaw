"""Task-neutral offline motion measurements, not a naturalness/safety verdict."""

from __future__ import annotations

from typing import Any

import numpy as np


def recovery_motion_metrics(
    root_position_m: Any,
    root_quaternion_xyzw: Any,
    joint_position_rad: Any,
    *,
    timestep_s: float,
    event_frame: int,
    forward_direction: Any,
) -> dict[str, float]:
    """Measure the full post-event window with an explicitly declared direction.

    Quaternion signs are equivalent. Joint derivatives are finite differences
    of actual measured positions, not commanded targets. Lower motion does not
    by itself mean more human-like motion: intentional follow-through can move.
    No simulator, actuator, file, checkpoint or authority is accessed here.
    """
    raw = [
        np.asarray(v)
        for v in (root_position_m, root_quaternion_xyzw, joint_position_rad, forward_direction)
    ]
    if any(v.dtype.kind not in "fiu" for v in raw):
        raise ValueError("finite numeric measured motion required")
    position, quaternion, joints, direction = [np.asarray(v, dtype=np.float64) for v in raw]
    if position.ndim != 2 or joints.ndim != 2:
        raise ValueError("aligned measured motion arrays required")
    n = len(position)
    if (
        not 4 <= n <= 200000
        or position.shape != (n, 3)
        or quaternion.shape != (n, 4)
        or joints.shape[0] != n
        or not 1 <= joints.shape[1] <= 512
        or direction.shape != (3,)
        or type(event_frame) is not int
        or not 0 <= event_frame <= n - 4
        or type(timestep_s) not in (int, float)
        or not np.isfinite(timestep_s)
        or not 1e-4 <= timestep_s <= 1.0
        or not all(np.isfinite(v).all() for v in (position, quaternion, joints, direction))
    ):
        raise ValueError("complete finite post-event measured motion required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            norms = np.linalg.norm(quaternion, axis=1)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0 or not np.allclose(norms, 1, atol=1e-3, rtol=0):
                raise ValueError("unit measured quaternions and nonzero direction required")
            direction = direction / direction_norm
            q = quaternion[event_frame:] / norms[event_frame:, None]
            p = position[event_frame:]
            joint = joints[event_frame:]
            projected = (p - p[0]) @ direction
            orientation = 2 * np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))
            angular_speed = (
                2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)) / timestep_s
            )

            def rms(values: Any) -> float:
                scale = float(np.max(np.abs(values)))
                return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean((values / scale) ** 2)))

            result = {
                "post_event_duration_s": float((len(p) - 1) * timestep_s),
                "maximum_retreat_m": float(max(0.0, -np.min(projected))),
                "final_forward_displacement_m": float(projected[-1]),
                "root_path_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                "maximum_orientation_excursion_rad": float(orientation.max()),
                "root_angular_speed_rms_rad_s": rms(angular_speed),
                "joint_speed_rms_rad_s": rms(np.diff(joint, axis=0) / timestep_s),
                "joint_acceleration_rms_rad_s2": rms(np.diff(joint, n=2, axis=0) / timestep_s**2),
                "joint_jerk_rms_rad_s3": rms(np.diff(joint, n=3, axis=0) / timestep_s**3),
            }
    except FloatingPointError as error:
        raise ValueError("derived measured motion must remain finite") from error
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("derived measured motion must remain finite")
    return result
