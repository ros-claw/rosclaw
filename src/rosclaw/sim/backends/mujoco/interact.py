"""可执行交互运行时（MH12，0916 优化 §十二-§十四，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

Interaction Contract ≠ Interaction Execution。官方 executor registry
（不允许 WorldSpec 写 arbitrary Python）；每个 executor 遵循
validate → precondition → execute → observe → postcondition → receipt。

Grasp 物理诚实（§十四）：approach → close gripper → contact evidence
→ capability check → **measured relative transform** → activate weld；
weld 必须标记 `constraint_assisted_grasp = true`（task abstraction，
不冒充完整 contact-dynamics grasp）；release 必须有"物体受重力响应"
证据（GRASP_HONESTY）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 官方 executor 注册表（WorldSpec 不允许 arbitrary Python）。
EXECUTORS = (
    "joint_target",
    "actuator_setpoint",
    "gripper_close",
    "gripper_open",
    "constraint_attach",
    "constraint_release",
)


def _body_id(model, name: str) -> int:  # noqa: ANN001
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: body {name!r}")
    return body_id


def _joint_for_actuator(model, actuator_name: str) -> int:  # noqa: ANN001
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: actuator {actuator_name!r}")
    return actuator_id


def _step(model, data, seconds: float, *, visit=None) -> None:  # noqa: ANN001
    import mujoco

    steps = max(1, math.ceil(seconds / float(model.opt.timestep)))
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise ValueError(f"SIM_DIVERGED: non-finite state during interaction at step {step + 1}")
        if visit is not None:
            visit(data, step)


def exec_joint_target(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """关节目标位：经该关节的 position 执行器（无执行器诚实拒绝）。"""
    import mujoco

    joint_name = interaction["target"]["name"]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: joint {joint_name!r}")
    actuator_id = -1
    for i in range(model.actuator_trnid.shape[0]):
        if int(model.actuator_trnid[i][0]) == joint_id:
            actuator_id = i
            break
    if actuator_id < 0:
        raise ValueError(
            f"CAPABILITY_UNAVAILABLE: joint {joint_name!r} has no actuator for joint_target"
        )
    target = payload.get("target")
    if not isinstance(target, (int, float)) or isinstance(target, bool) or not math.isfinite(target):
        raise ValueError(f"INTERACTION_PAYLOAD_INVALID: target must be a finite number")
    duration = float(payload.get("duration_s", 0.5))
    adr = int(model.jnt_qposadr[joint_id])
    before = float(data.qpos[adr])
    data.ctrl[actuator_id] = float(target)
    _step(model, data, duration)
    after = float(data.qpos[adr])
    return {
        "before": before,
        "after": after,
        "target": float(target),
        "reached": abs(after - float(target)) <= float(payload.get("tolerance", 0.02)),
    }


def exec_actuator_setpoint(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """执行器 setpoint（pos/vel/ff/ctrl，按 control schema 寻址）。"""
    actuator_name = interaction["target"]["name"]
    _joint_for_actuator(model, actuator_name)
    spec = backend._spec_from_manifest(backend._manifest(interaction["_model_ref"]))
    from rosclaw.sim.backends.mujoco.inspect import _control_channels

    channels = _control_channels(model, spec)
    channel_map = {
        (c["actuator"], c["role"]): c["index"] for c in channels if c["actuator"] == actuator_name
    }
    if not channel_map:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: no control channel for {actuator_name!r}")
    applied: dict[str, float] = {}
    for role, value in payload.get("setpoints", {}).items():
        key = (actuator_name, role)
        if key not in channel_map:
            raise ValueError(f"CONTROLLER_SCHEMA_MISMATCH: no channel {actuator_name}:{role}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"INTERACTION_PAYLOAD_INVALID: setpoint {role} must be finite")
        data.ctrl[channel_map[key]] = float(value)
        applied[role] = float(value)
    duration = float(payload.get("duration_s", 0.5))
    _step(model, data, duration)
    return {"applied": applied}


def exec_gripper_motion(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any], *, close: bool) -> dict[str, Any]:  # noqa: ANN001
    """gripper_close / gripper_open：驱动夹爪执行器并记录接触证据。"""
    import mujoco

    actuator_name = interaction["target"]["name"]
    actuator_id = _joint_for_actuator(model, actuator_name)
    direction = payload.get("close_target" if close else "open_target")
    if direction is None:
        direction = 1.0 if close else 0.0
    if not isinstance(direction, (int, float)) or isinstance(direction, bool) or not math.isfinite(direction):
        raise ValueError("INTERACTION_PAYLOAD_INVALID: gripper target must be finite")
    data.ctrl[actuator_id] = float(direction)
    duration = float(payload.get("duration_s", 0.5))
    contacts: list[dict[str, Any]] = []

    def visit(d, step) -> None:  # noqa: ANN001
        for i in range(d.ncon):
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(d.contact[i].geom1)) or "?"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(d.contact[i].geom2)) or "?"
            contacts.append({"geom1": g1, "geom2": g2, "dist": float(d.contact[i].dist)})

    _step(model, data, duration, visit=visit)
    return {
        "target": float(direction),
        "contact_evidence": contacts[:50],
        "contact_count": len(contacts),
    }


def exec_constraint_attach(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """constraint_attach（GRASP_HONESTY §十四）：
    contact evidence → capability check → measured relative transform →
    activate weld（必须预先在模型中声明 equality weld），标记
    constraint_assisted_grasp=true。
    """
    import mujoco

    weld_name = payload.get("weld") or interaction.get("weld")
    if not weld_name:
        raise ValueError("INTERACTION_PAYLOAD_INVALID: constraint_attach requires 'weld' name")
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        raise ValueError(
            f"INTERACTION_NO_WELD_DECLARED: equality weld {weld_name!r} not declared in model"
        )
    body1_id = int(model.eq_obj1id[eq_id])
    body2_id = int(model.eq_obj2id[eq_id])
    body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
    body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

    # 接触/近距证据：两 body 的 geom 间最小距离。
    geoms1 = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body1_id]
    geoms2 = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body2_id]
    min_dist = min(
        (
            float(mujoco.mj_geomDistance(model, data, g1, g2, 0.5, None))
            for g1 in geoms1
            for g2 in geoms2
        ),
        default=float("inf"),
    )
    attach_threshold = float(payload.get("attach_threshold_m", 0.01))
    if min_dist > attach_threshold:
        raise ValueError(
            f"INTERACTION_PRECONDITION_FAILED: {body1}↔{body2} min distance "
            f"{min_dist:.4f}m > {attach_threshold}m——无接触证据不得 attach"
        )

    # 实测相对位姿（body2 在 body1 坐标系）。
    rotation = np.asarray(data.xmat[body1_id], dtype=float).reshape(3, 3)
    rel_pos = rotation.T @ (np.asarray(data.xpos[body2_id]) - np.asarray(data.xpos[body1_id]))
    rel_quat = np.asarray(data.xquat[body2_id], dtype=float)  # 简化：保持 body2 当前朝向

    model.eq_data[eq_id, 0:3] = rel_pos
    model.eq_data[eq_id, 3:7] = [1.0, 0.0, 0.0, 0.0]
    model.eq_data[eq_id, 7:11] = rel_quat
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)
    return {
        "weld": weld_name,
        "bodies": [body1, body2],
        "min_distance_m": min_dist,
        "measured_relpose": {
            "pos": [float(v) for v in rel_pos],
            "quat": [float(v) for v in rel_quat],
        },
        "constraint_assisted_grasp": True,  # task abstraction，不冒充接触动力学抓取
    }


def exec_constraint_release(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """constraint_release：deactivate weld → step → 物体必须受重力响应
    （GRASP_HONESTY：释放证据）。"""
    import mujoco

    weld_name = payload.get("weld") or interaction.get("weld")
    if not weld_name:
        raise ValueError("INTERACTION_PAYLOAD_INVALID: constraint_release requires 'weld' name")
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        raise ValueError(
            f"INTERACTION_NO_WELD_DECLARED: equality weld {weld_name!r} not declared in model"
        )
    body2_id = int(model.eq_obj2id[eq_id])
    z_before = float(data.xpos[body2_id][2])
    qvel_before = float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0
    data.eq_active[eq_id] = 0
    duration = float(payload.get("duration_s", 0.3))
    _step(model, data, duration)
    z_after = float(data.xpos[body2_id][2])
    qvel_after = float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0
    gravity_response = (z_after < z_before - 1e-4) or (qvel_after > qvel_before + 1e-6)
    return {
        "weld": weld_name,
        "z_before": z_before,
        "z_after": z_after,
        "gravity_response": gravity_response,
    }
