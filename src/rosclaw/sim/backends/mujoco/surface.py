"""Private current-pose surface distances, not contact or clearance certificates.

Simulation owners must serialize stepping and model mutation around this call.
No live solver/cache fields are refreshed. Body origins are deliberately not
used as substitutes for collision-surface distance.
"""

from __future__ import annotations

import math
from typing import Any


def surface_snapshot(
    model: Any,
    data: Any,
    pairs: tuple[tuple[str, str], ...],
    *,
    maximum_distance_m: float = 1.0,
    include_distance_jacobian: bool = False,
) -> dict[str, Any]:
    """Inspect up to 32 explicit geom pairs on private MuJoCo kinematics.

    A native distance cutoff is UNKNOWN, not proof of separation. Even a
    measured gap describes only this configuration, not swept clearance,
    collision filtering, support load, stability, or hardware authorization.
    Optional derivatives are local tangent-space linearizations (distance rate
    equals Jacobian times qvel), not globally smooth gradients or motion plans.
    """
    import mujoco
    import numpy as np

    if (
        not isinstance(model, mujoco.MjModel)
        or not isinstance(data, mujoco.MjData)
        or getattr(data, "model", None) is not model
        or type(pairs) is not tuple
        or not 1 <= len(pairs) <= 32
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(name) is not str or not 1 <= len(name) <= 256 for name in pair)
            or pair[0] == pair[1]
            for pair in pairs
        )
        or type(maximum_distance_m) not in (int, float)
        or not math.isfinite(maximum_distance_m)
        or not 0 < maximum_distance_m <= 10
        or type(include_distance_jacobian) is not bool
        or (include_distance_jacobian and model.nv > 4096)
    ):
        raise ValueError("bounded named surface pairs and matching simulation model/data required")
    if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
        raise ValueError("duplicate surface pairs")
    ids = [
        tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in pair)
        for pair in pairs
    ]
    if any(index < 0 for pair in ids for index in pair):
        raise ValueError("unknown surface geometry")
    time = float(data.time)
    source = {name: getattr(data, name).copy() for name in ("qpos", "mocap_pos", "mocap_quat")}
    if (
        not math.isfinite(time)
        or time < 0
        or any(not np.isfinite(value).all() for value in source.values())
    ):
        raise ValueError("finite simulation pose required")
    quaternions = list(source["mocap_quat"])
    for kind, address in zip(model.jnt_type, model.jnt_qposadr, strict=True):
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            quaternions.append(source["qpos"][address + 3 : address + 7])
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            quaternions.append(source["qpos"][address : address + 4])
    if any(abs(float(np.linalg.norm(q)) - 1) > 1e-5 for q in quaternions):
        raise ValueError("unit simulation pose quaternion required")
    private = mujoco.MjData(model)
    private.time = time
    for name, value in source.items():
        getattr(private, name)[:] = value
    mujoco.mj_kinematics(model, private)
    if include_distance_jacobian:
        mujoco.mj_comPos(model, private)
    rows = []
    for names, (first, second) in zip(pairs, ids, strict=True):
        segment = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(model, private, first, second, maximum_distance_m, segment)
        )
        if not math.isfinite(distance) or not np.isfinite(segment).all():
            raise ValueError("nonfinite native surface result")
        censored = distance >= maximum_distance_m
        row: dict[str, Any] = {
            "geometries": list(names),
            "status": "UNKNOWN" if censored else "MEASURED",
            "signed_distance_m": None if censored else distance,
            "nearest_segment_world_m": None if censored else segment.tolist(),
            "reason": "cutoff_or_unsupported_pair" if censored else "native_pose_geometry",
        }
        if include_distance_jacobian:
            derivative = None
            jacobian_reason = "cutoff_or_unsupported_pair" if censored else "near_zero_distance"
            if not censored and abs(distance) > 1e-9:
                first_jacobian = np.zeros((3, model.nv), dtype=np.float64)
                second_jacobian = np.zeros_like(first_jacobian)
                mujoco.mj_jac(
                    model, private, first_jacobian, None, segment[:3], int(model.geom_bodyid[first])
                )
                mujoco.mj_jac(
                    model,
                    private,
                    second_jacobian,
                    None,
                    segment[3:],
                    int(model.geom_bodyid[second]),
                )
                normal = (segment[:3] - segment[3:]) / distance
                if abs(float(np.linalg.norm(normal)) - 1.0) <= 1e-5:
                    derivative = normal @ (first_jacobian - second_jacobian)
                    if not np.isfinite(derivative).all():
                        raise ValueError("nonfinite native surface linearization")
                    jacobian_reason = "closest_surface_branch"
                else:
                    jacobian_reason = "degenerate_native_segment"
            row.update(
                distance_jacobian_qvel=None if derivative is None else derivative.tolist(),
                jacobian_status="UNKNOWN" if derivative is None else "LOCAL_LINEARIZATION",
                jacobian_reason=jacobian_reason,
            )
        rows.append(row)
    if data.time != time or any(
        not np.array_equal(getattr(data, name), value) for name, value in source.items()
    ):
        raise RuntimeError("simulation pose changed during surface inspection")
    return {
        "schema": (
            "rosclaw.sim.surface_snapshot.v2"
            if include_distance_jacobian
            else "rosclaw.sim.surface_snapshot.v1"
        ),
        "time_sec": time,
        "maximum_distance_m": maximum_distance_m,
        "pairs": rows,
        "contact_forces_measured": False,
        "continuous_clearance_verified": False,
        "live_state_modified": False,
        "activation_ceiling": "SIM_ONLY",
    }
