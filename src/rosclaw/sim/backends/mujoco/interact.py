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
    """关节目标位：经 ControlMapper 写 (actuator, pos) 槽位
    （MH20-B——actuator 序号 ≠ ctrl 槽位，PID 多槽会错位）。"""
    import mujoco

    from rosclaw.sim.backends.mujoco import control as control_mod

    joint_name = interaction["target"]["name"]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: joint {joint_name!r}")
    spec = backend._spec_from_manifest(backend._manifest(interaction["_model_ref"]))
    channel_map = control_mod.channel_map_for(model, spec)
    target = payload.get("target")
    if not isinstance(target, (int, float)) or isinstance(target, bool) or not math.isfinite(target):
        raise ValueError("INTERACTION_PAYLOAD_INVALID: target must be a finite number")
    duration = float(payload.get("duration_s", 0.5))
    adr = int(model.jnt_qposadr[joint_id])
    before = float(data.qpos[adr])
    control_mod.write_joint_target(
        model, data, channel_map, joint_name=joint_name, value=float(target)
    )
    _step(model, data, duration)
    after = float(data.qpos[adr])
    return {
        "before": before,
        "after": after,
        "target": float(target),
        "reached": abs(after - float(target)) <= float(payload.get("tolerance", 0.02)),
    }


def exec_actuator_setpoint(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """执行器 setpoint（pos/vel/ff/ctrl，经 ControlMapper 按
    control schema 寻址）。"""
    from rosclaw.sim.backends.mujoco import control as control_mod

    actuator_name = interaction["target"]["name"]
    _joint_for_actuator(model, actuator_name)
    spec = backend._spec_from_manifest(backend._manifest(interaction["_model_ref"]))
    channel_map = {
        key: slot for key, slot in control_mod.channel_map_for(model, spec).items() if key[0] == actuator_name
    }
    if not channel_map:
        raise ValueError(f"INTERACTION_TARGET_NOT_FOUND: no control channel for {actuator_name!r}")
    applied: dict[str, float] = {}
    for role, value in payload.get("setpoints", {}).items():
        applied[role] = control_mod.write_setpoint(
            model, data, channel_map, actuator=actuator_name, role=role, value=value
        )
    duration = float(payload.get("duration_s", 0.5))
    _step(model, data, duration)
    return {"applied": applied}


def exec_gripper_motion(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any], *, close: bool) -> dict[str, Any]:  # noqa: ANN001
    """gripper_close / gripper_open：经 ControlMapper 驱动夹爪执行器
    并记录接触证据（MH20-B——actuator 序号 ≠ ctrl 槽位）。"""
    import mujoco

    from rosclaw.sim.backends.mujoco import control as control_mod

    actuator_name = interaction["target"]["name"]
    _joint_for_actuator(model, actuator_name)
    spec = backend._spec_from_manifest(backend._manifest(interaction["_model_ref"]))
    channel_map = control_mod.channel_map_for(model, spec)
    direction = payload.get("close_target" if close else "open_target")
    if direction is None:
        direction = 1.0 if close else 0.0
    if not isinstance(direction, (int, float)) or isinstance(direction, bool) or not math.isfinite(direction):
        raise ValueError("INTERACTION_PAYLOAD_INVALID: gripper target must be finite")
    control_mod.write_setpoint(
        model,
        data,
        channel_map,
        actuator=actuator_name,
        role=control_mod.resolve_position_role(channel_map, actuator_name),
        value=float(direction),
    )
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


def relative_body_pose(pos1, quat1, pos2, quat2):  # noqa: ANN001, ANN202
    """body2 在 body1 坐标系下的相对位姿（MH20-C §9）。

    pos = R1^T (p2 - p1)；quat = inverse(q1) ⊗ q2——world-frame
    body2 quat 不是相对朝向（body1 旋转即错）。
    """
    import mujoco

    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat1, dtype=float))
    rel_pos = rotation.reshape(3, 3).T @ (np.asarray(pos2, dtype=float) - np.asarray(pos1, dtype=float))
    q1 = np.asarray(quat1, dtype=float)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(rel_quat, q1_inv, np.asarray(quat2, dtype=float))
    return rel_pos, rel_quat


def contact_evidence_between(model, data, body1_id: int, body2_id: int) -> dict[str, Any]:  # noqa: ANN001
    """两 body 间的真实接触证据（MH20-C §8.2）：contact 对数量、
    最深穿透、法向力合计、是否持续。"""
    import mujoco

    count = 0
    min_dist = float("inf")
    max_penetration = 0.0
    normal_force_total = 0.0
    force6 = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        if set(bodies) != {body1_id, body2_id}:
            continue
        count += 1
        dist = float(contact.dist)
        min_dist = min(min_dist, dist)
        max_penetration = max(max_penetration, -dist)
        mujoco.mj_contactForce(model, data, index, force6)
        normal_force_total += abs(float(force6[0]))
    return {
        "contact_count": count,
        "min_distance_m": min_dist if count else None,
        "max_penetration_m": max_penetration,
        "normal_force": normal_force_total,
    }


def set_weld_relpose(model, data, eq_id: int, rel_pos, rel_quat) -> None:  # noqa: ANN001
    """写 weld relpose 的唯一 helper（MH20-C §10）。

    eq_data 布局（3.13 实测）：anchor[0:3] / pos[3:6] / quat[6:10] /
    torquescale[10]——anchor 与 torquescale 不动。官方对 eq_data
    运行时修改标记 Safe with mj_setConst：写后必须 mj_setConst →
    eq_active → mj_forward（不让其他模块散落布局知识）。"""
    import mujoco

    model.eq_data[eq_id, 3:6] = np.asarray(rel_pos, dtype=float)
    model.eq_data[eq_id, 6:10] = np.asarray(rel_quat, dtype=float)
    mujoco.mj_setConst(model, data)
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)


#: 证据三级（MH20-C §8.1）：靠近 ≠ 接触 ≠ 承重抓取。
EVIDENCE_PROXIMITY = "PROXIMITY_ASSISTED_ATTACH"
EVIDENCE_CONTACT = "CONTACT"
EVIDENCE_LOAD_BEARING = "LOAD_BEARING_CONTACT"


def exec_constraint_attach(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """constraint_attach（GRASP_HONESTY v2 §8-§10）：
    **真接触证据**（data.contact 实际 pair）→ capability check →
    measured relative pose（含正确相对四元数）→ set_weld_relpose
    （mj_setConst）→ activate weld。

    默认要求 CONTACT 级证据；proximity abstraction 必须显式声明
    evidence_level=PROXIMITY_ASSISTED_ATTACH（诚实降级命名）。
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

    evidence = contact_evidence_between(model, data, body1_id, body2_id)
    requested_level = payload.get("evidence_level")
    if evidence["contact_count"] >= 1:
        evidence_level = EVIDENCE_CONTACT
    else:
        # 无真实接触：仅当显式声明 proximity abstraction 才允许，
        # 且仍要求近距（geom 最小距离 ≤ attach_threshold）。
        if requested_level != EVIDENCE_PROXIMITY:
            raise ValueError(
                f"INTERACTION_NO_CONTACT_EVIDENCE: {body1}↔{body2} 无实际接触对"
                "（proximity ≠ contact；proximity abstraction 需显式声明"
                " evidence_level=PROXIMITY_ASSISTED_ATTACH）"
            )
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
                f"{min_dist:.4f}m > {attach_threshold}m——proximity attach 也需近距"
            )
        evidence["proximity_min_distance_m"] = min_dist
        evidence_level = EVIDENCE_PROXIMITY

    # 实测相对位姿（位置与朝向都在 body1 坐标系——§9）。
    rel_pos, rel_quat = relative_body_pose(
        np.asarray(data.xpos[body1_id], dtype=float),
        np.asarray(data.xquat[body1_id], dtype=float),
        np.asarray(data.xpos[body2_id], dtype=float),
        np.asarray(data.xquat[body2_id], dtype=float),
    )
    set_weld_relpose(model, data, eq_id, rel_pos, rel_quat)
    return {
        "weld": weld_name,
        "bodies": [body1, body2],
        "evidence_level": evidence_level,
        "contact_evidence": evidence,
        "measured_relpose": {
            "pos": [float(v) for v in rel_pos],
            "quat": [float(v) for v in rel_quat],
        },
        "constraint_assisted_grasp": True,  # task abstraction，不冒充接触动力学抓取
    }


def exec_constraint_release(backend, model, data, interaction: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """constraint_release：deactivate weld → step → **payload 自身**
    必须有重力响应（MH20-C §11：Evidence 必须指向它声称证明的
    对象——不看全局 max qvel，只看被释放 body 的速度/位移/高度）。"""
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
    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
    pos_before = np.asarray(data.xpos[body2_id], dtype=float).copy()
    data.eq_active[eq_id] = 0
    duration = float(payload.get("duration_s", 0.3))
    _step(model, data, duration)
    pos_after = np.asarray(data.xpos[body2_id], dtype=float)
    linear_velocity = float(
        np.linalg.norm(np.asarray(data.cvel[body2_id][3:6], dtype=float))
    )
    angular_velocity = float(
        np.linalg.norm(np.asarray(data.cvel[body2_id][0:3], dtype=float))
    )
    displacement = float(np.linalg.norm(pos_after - pos_before))
    velocity_threshold = float(payload.get("velocity_threshold", 1e-3))
    gravity_response = bool(
        pos_after[2] < pos_before[2] - 1e-4
        or linear_velocity > velocity_threshold
        or angular_velocity > velocity_threshold
        or displacement > 1e-4
    )
    return {
        "weld": weld_name,
        "target_body": body2_name,
        "z_before": float(pos_before[2]),
        "z_after": float(pos_after[2]),
        "payload_linear_velocity": linear_velocity,
        "payload_angular_velocity": angular_velocity,
        "payload_displacement": displacement,
        "gravity_response": gravity_response,
    }
