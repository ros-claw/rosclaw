"""动力学/控制类 audit（PR-MH4，规格 §17.1/§18）：A03/A04/A15/A16/A17。"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.sim.audit.context import AuditContext


def a03_servo_hold(ctx: AuditContext) -> dict[str, Any]:
    """A03：position 伺服 command qpos0 保持 policy.servo_hold_s 秒，
    关节漂移超 sag 阈值即 FAIL（铰链 1°、滑轨 2mm）。"""
    import mujoco

    model = ctx.model
    servos = [
        i
        for i in range(model.nu)
        if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and float(model.actuator_gainprm[i][0]) > 0
    ]
    if not servos:
        return {"status": "PASS", "violations": [], "detail": {"note": "no_position_servos"}}

    data = ctx.fresh_data()
    qpos0 = np.asarray(data.qpos, dtype=float).copy()
    for actuator_id in servos:
        joint_id = int(model.actuator_trnid[actuator_id][0])
        data.ctrl[actuator_id] = qpos0[int(model.jnt_qposadr[joint_id])]
    mujoco.mj_forward(model, data)

    import math

    steps = max(1, math.ceil(ctx.policy.servo_hold_s / float(model.opt.timestep)))
    for _ in range(steps):
        mujoco.mj_step(model, data)

    violations: list[dict[str, Any]] = []
    max_linear = 0.0
    max_angular = 0.0
    checked: set[int] = set()
    for actuator_id in servos:
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id in checked:
            continue
        checked.add(joint_id)
        adr = int(model.jnt_qposadr[joint_id])
        drift = abs(float(data.qpos[adr]) - float(qpos0[adr]))
        joint_type = int(model.jnt_type[joint_id])
        limit = (
            ctx.policy.servo_sag_rad
            if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)
            else ctx.policy.servo_sag_m
        )
        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            max_angular = max(max_angular, drift)
        else:
            max_linear = max(max_linear, drift)
        if drift > limit:
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            )
            violations.append({"joint": name, "drift": drift, "limit": limit})
    status = "FAIL" if violations else "PASS"
    return {
        "status": status,
        "violations": violations,
        "max_linear_drift_m": max_linear,
        "max_angular_drift_rad": max_angular,
    }


def a04_link_continuity(ctx: AuditContext) -> dict[str, Any]:
    """A04：jointed body 与其最近有 geom 的祖先之间，几何最小缝隙
    超 continuity_gap_m 即 TORN（free joint 排除）。"""
    import mujoco

    model = ctx.model

    def _geoms_of(body_id: int) -> list[int]:
        return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]

    def _has_free_joint(body_id: int) -> bool:
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) == body_id and int(model.jnt_type[j]) == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                return True
        return False

    data = ctx.fresh_data()
    violations: list[dict[str, Any]] = []
    for body_id in range(1, model.nbody):
        if int(model.body_jntnum[body_id]) == 0 or _has_free_joint(body_id):
            continue
        child_geoms = _geoms_of(body_id)
        if not child_geoms:
            continue
        ancestor = int(model.body_parentid[body_id])
        while ancestor > 0 and not _geoms_of(ancestor):
            ancestor = int(model.body_parentid[ancestor])
        parent_geoms = _geoms_of(ancestor)
        if not parent_geoms:
            continue
        gap = min(
            float(mujoco.mj_geomDistance(model, data, child, parent, 0.5, None))
            for child in child_geoms
            for parent in parent_geoms
        )
        if gap > ctx.policy.continuity_gap_m:
            violations.append(
                {
                    "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or f"body_{body_id}",
                    "gap_m": gap,
                    "limit_m": ctx.policy.continuity_gap_m,
                }
            )
    status = "FAIL" if violations else "PASS"
    return {"status": status, "violations": violations}


def _hold_sweep(ctx: AuditContext, seconds: float) -> dict[str, Any]:
    """hold 扫描共享 helper：发散/最大速度/能量轨迹。"""
    energy0 = None
    energy_end = None
    max_qvel = 0.0

    def visit(data, step) -> None:  # noqa: ANN001
        nonlocal energy0, energy_end, max_qvel
        import mujoco

        mujoco.mj_energyPos(ctx.model, data)
        potential = float(data.energy[0])
        mujoco.mj_energyVel(ctx.model, data)
        total = potential + float(data.energy[1])
        if energy0 is None:
            energy0 = total
        energy_end = total
        max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0)

    out = ctx.sweep(seconds, visit, ctrl={"kind": "hold"})
    return {
        "diverged": out["diverged"],
        "max_qvel": max_qvel,
        "energy0": energy0,
        "energy_end": energy_end,
    }


def a15_nan_inf(ctx: AuditContext) -> dict[str, Any]:
    """A15：NaN/Inf 哨兵——(a) 提供的 trace 状态全部有限；
    (b) hold 扫描中状态不发散。

    注：float64 CPU MuJoCo 下物理自然产生真 NaN 极难（3.11 实测
    极限环而非溢出），因此 trace 数据完整性是本检查的核心面。
    """
    import math

    if ctx.trace_record is not None:
        for index, snapshot in enumerate(ctx.trace_record.get("states", [])):
            for key in ("t", "qpos", "qvel", "ctrl"):
                value = snapshot.get(key)
                items = value if isinstance(value, list) else [value]
                for v in items:
                    if not isinstance(v, (int, float)) or not math.isfinite(v):
                        return {
                            "status": "FAIL",
                            "violations": [
                                {
                                    "reason": "non_finite_trace_state",
                                    "state_index": index,
                                    "field": key,
                                }
                            ],
                        }
    result = _hold_sweep(ctx, min(0.5, ctx.policy.sweep_s))
    status = "FAIL" if result["diverged"] else "PASS"
    return {
        "status": status,
        "violations": [] if status == "PASS" else [{"reason": "non_finite_state"}],
    }


def a16_physics_divergence(ctx: AuditContext) -> dict[str, Any]:
    """A16：hold 下物理发散（NaN 或 |qvel| 超阈值）即 FAIL。"""
    result = _hold_sweep(ctx, ctx.policy.sweep_s)
    violations = []
    if result["diverged"]:
        violations.append({"reason": "nan_divergence"})
    if result["max_qvel"] > ctx.policy.max_qvel:
        violations.append({"reason": "qvel_explosion", "max_qvel": result["max_qvel"]})
    status = "FAIL" if violations else "PASS"
    return {"status": status, "violations": violations, "max_qvel": result["max_qvel"]}


def a17_energy_explosion(ctx: AuditContext) -> dict[str, Any]:
    """A17：hold 下总能量增长超 factor/abs 阈值即 FAIL。"""
    result = _hold_sweep(ctx, ctx.policy.sweep_s)
    violations = []
    if result["diverged"]:
        violations.append({"reason": "diverged_before_energy_check"})
    elif result["energy0"] is not None:
        growth = abs(result["energy_end"]) - abs(result["energy0"])
        if growth > ctx.policy.energy_explosion_abs or abs(
            result["energy_end"]
        ) > ctx.policy.energy_explosion_factor * max(abs(result["energy0"]), 1e-9):
            violations.append(
                {
                    "reason": "energy_explosion",
                    "energy0": result["energy0"],
                    "energy_end": result["energy_end"],
                }
            )
    status = "FAIL" if violations else "PASS"
    return {"status": status, "violations": violations}
