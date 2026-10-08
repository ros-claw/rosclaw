"""Read-only evaluated contact wrenches, not motion or task-success authority.

The simulation owner must serialize stepping, inspection and model mutation.
Post-step qpos/qvel are not the state at which Euler contact forces were solved.
Callers supply that pre-integration state; cached kinematics must match it.
"""

from __future__ import annotations

import math
from typing import Any


def contact_snapshot(
    model: Any,
    data: Any,
    *,
    evaluated_qpos: tuple[float, ...],
    evaluated_qvel: tuple[float, ...],
    evaluated_time_sec: float,
    geometry_names: tuple[str, ...],
) -> dict[str, Any]:
    """Inspect native rigid-geom contacts touching explicit named geometries.

    Supports same-time forward evaluation or one Euler/implicit step. Does not
    integrate, recompute live forces, infer impulse from force, certify contact
    success, or grant hardware authority. Flex contacts are rejected explicitly.
    Concurrent model changes are forbidden by the owner contract, not sandboxed.
    """
    import mujoco
    import numpy as np

    if (
        not isinstance(model, mujoco.MjModel)
        or not isinstance(data, mujoco.MjData)
        or getattr(data, "model", None) is not model
        or model.nv > 4096
        or int(model.opt.integrator) not in (0, 2, 3)
        or type(geometry_names) is not tuple
        or not 1 <= len(geometry_names) <= 32
        or any(type(name) is not str or not 1 <= len(name) <= 256 for name in geometry_names)
        or len(set(geometry_names)) != len(geometry_names)
        or type(evaluated_time_sec) not in (int, float)
        or not math.isfinite(evaluated_time_sec)
        or evaluated_time_sec < 0
    ):
        raise ValueError("bounded explicit evaluated contact query required")
    for value, size in ((evaluated_qpos, model.nq), (evaluated_qvel, model.nv)):
        if (
            type(value) is not tuple
            or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
        ):
            raise ValueError("finite immutable evaluated state required")
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geometry_names}
    if -1 in ids:
        raise ValueError("unknown contact geometry")
    time = float(data.time)
    elapsed = time - evaluated_time_sec
    if not math.isfinite(time) or not (
        abs(elapsed) <= 1e-10 or abs(elapsed - float(model.opt.timestep)) <= 1e-10
    ):
        raise ValueError("contact evaluation time must be current or one integration step earlier")
    names = (
        "qpos",
        "qvel",
        "ctrl",
        "qacc",
        "qacc_warmstart",
        "qfrc_constraint",
        "geom_xpos",
        "geom_xmat",
        "cvel",
        "mocap_pos",
        "mocap_quat",
        "efc_force",
    )
    before = {name: getattr(data, name).copy() for name in names}
    if any(not np.isfinite(value).all() for value in before.values()):
        raise ValueError("finite live simulation state required")
    count = int(data.ncon)
    if not 0 <= count <= 4096:
        raise ValueError("bounded contact count required")

    def contact_signature() -> tuple[bytes, ...]:
        return tuple(
            np.asarray(getattr(data.contact, name)).tobytes()
            for name in ("geom1", "geom2", "pos", "frame", "dist", "dim", "efc_address")
        )

    contact_bytes = contact_signature()
    private = mujoco.MjData(model)
    private.qpos[:] = evaluated_qpos
    private.qvel[:] = evaluated_qvel
    private.mocap_pos[:] = before["mocap_pos"]
    private.mocap_quat[:] = before["mocap_quat"]
    quaternions = list(private.mocap_quat)
    for kind, address in zip(model.jnt_type, model.jnt_qposadr, strict=True):
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            quaternions.append(private.qpos[address + 3 : address + 7])
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            quaternions.append(private.qpos[address : address + 4])
    if any(abs(float(np.linalg.norm(q)) - 1) > 1e-5 for q in quaternions):
        raise ValueError("unit evaluated quaternion required")
    mujoco.mj_kinematics(model, private)
    mujoco.mj_comPos(model, private)
    mujoco.mj_comVel(model, private)
    if any(
        not np.allclose(getattr(private, name), before[name], rtol=0, atol=1e-10)
        for name in ("geom_xpos", "geom_xmat", "cvel")
    ):
        raise ValueError("evaluated state does not match native contact kinematics")
    rows = []
    for index in range(count):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in ids and second not in ids:
            continue
        if first < 0 or second < 0:
            raise ValueError("flex contacts are outside this rigid geometry contract")
        frame = np.asarray(contact.frame).reshape(3, 3).copy()
        point = np.asarray(contact.pos).copy()
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        if (
            not np.isfinite(wrench).all()
            or not np.isfinite(point).all()
            or not np.isfinite(frame).all()
            or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-6)
            or abs(float(np.linalg.det(frame)) - 1) > 1e-6
            or not math.isfinite(float(contact.dist))
        ):
            raise ValueError("finite orthonormal native contact result required")
        velocities = []
        for geom in (first, second):
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, private, jacobian, None, point, int(model.geom_bodyid[geom]))
            velocities.append(jacobian @ private.qvel)
        relative = velocities[1] - velocities[0]
        normal_speed = float(frame[0] @ relative)
        force = frame.T @ wrench[:3]
        torque = frame.T @ wrench[3:]
        tangential_speed = float(np.linalg.norm(relative - normal_speed * frame[0]))
        if (
            any(not np.isfinite(v).all() for v in (*velocities, force, torque))
            or not math.isfinite(normal_speed)
            or not math.isfinite(tangential_speed)
        ):
            raise ValueError("nonfinite transformed contact measurement")
        rows.append(
            {
                "contact_index": index,
                "geometry_ids": [first, second],
                "geometry_names": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (first, second)
                ],
                "point_world_m": point.tolist(),
                "normal_first_to_second": frame[0].tolist(),
                "signed_distance_m": float(contact.dist),
                "constraint_active": int(contact.efc_address) >= 0,
                "force_on_second_world_n": force.tolist(),
                "torque_on_second_at_contact_world_nm": torque.tolist(),
                "first_point_velocity_world_mps": velocities[0].tolist(),
                "second_point_velocity_world_mps": velocities[1].tolist(),
                "relative_normal_velocity_mps": normal_speed,
                "relative_tangential_speed_mps": tangential_speed,
            }
        )
    if (
        data.time != time
        or int(data.ncon) != count
        or contact_signature() != contact_bytes
        or any(not np.array_equal(getattr(data, name), value) for name, value in before.items())
    ):
        raise RuntimeError("simulation changed during contact inspection")
    return {
        "schema": "rosclaw.sim.contact_snapshot.v1",
        "simulation_time_sec": time,
        "evaluated_time_sec": float(evaluated_time_sec),
        "contacts": rows,
        "live_state_modified": False,
        "impulse_measured": False,
        "task_success_verified": False,
        "activation_ceiling": "SIM_ONLY",
    }
