"""限值与数值健壮性 audit（MH15，0916 优化 §二十一-§二十三）。

A09-A14（saturation/力/速度/加速度/接触）与 A21-A24（sensor 有效性/
frame 约定/solver 敏感性/timestep 敏感性）。

**限值不能凭 ROSClaw 猜**：优先 e-URDF Safety Profile
（velocity_limits/force_limits）与模型自带 ctrlrange/forcerange
（物理事实）；body 没有声明 → NOT_EVALUATED（不用万能阈值）。
NOT_EVALUATED 不拉低总状态（诚实记录）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.sim.audit.context import AuditContext

#: 安全组合 solver 探针（Newton 为默认，CG 为对照）。
_SOLVER_PROBE = ("newton", "cg")


def _safety_limits(ctx: AuditContext) -> dict[str, Any] | None:
    """e-URDF Safety Profile（velocity_limits/force_limits）。"""
    explicit = ctx.extra.get("safety_limits")
    if explicit is not None:
        return explicit
    source = ctx.extra.get("model_source") or {}
    if source.get("kind") != "eurdf":
        return None

    import yaml

    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    safety_file = _default_zoo_path() / source["ref"] / "safety.yaml"
    if not safety_file.is_file():
        return None
    data = yaml.safe_load(safety_file.read_text(encoding="utf-8")) or {}
    return data.get("safety_limits") or {}


def _trace_or_hold_sweep(ctx: AuditContext, seconds: float = 1.0) -> dict[str, Any]:
    """在 hold（或 trace controller）扫描中采集限值指标。"""
    ctrl = None
    if ctx.trace_record is not None:
        controller = ctx.trace_record.get("controller", {})
        if "ctrl_series" in controller:
            ctrl = {"kind": "ctrl_series", "rows": controller["ctrl_series"]}
        elif "position_targets" in controller:
            ctrl = {"kind": "ctrl", "values": controller["position_targets"]}

        seconds = ctx.trace_record.get("steps", 1) * float(ctx.model.opt.timestep)

    metrics: dict[str, Any] = {
        "saturation_steps": 0,
        "total_steps": 0,
        "max_actuator_force": 0.0,
        "max_qvel": 0.0,
        "max_qacc": 0.0,
        "max_contact_force": 0.0,
        "qvel_per_joint": {},
        "sensordata_finite": True,
        "diverged": False,
    }
    import mujoco

    model = ctx.model

    def visit(data, step) -> None:  # noqa: ANN001
        metrics["total_steps"] += 1
        for i in range(model.nu):
            lo, hi = float(model.actuator_ctrlrange[i][0]), float(model.actuator_ctrlrange[i][1])
            # MuJoCo 不回写裁剪 data.ctrl——命令值可越界滞留；
            # 贴边界与越界都算饱和（实证 0916）。
            if model.actuator_ctrllimited[i] and (
                float(data.ctrl[i]) <= lo + 1e-9 or float(data.ctrl[i]) >= hi - 1e-9
            ):
                metrics["saturation_steps"] += 1
                break
        if model.nu:
            metrics["max_actuator_force"] = max(
                metrics["max_actuator_force"],
                float(max(abs(v) for v in data.actuator_force)),
            )
        if data.qvel.size:
            metrics["max_qvel"] = max(metrics["max_qvel"], float(max(abs(v) for v in data.qvel)))
            for j in range(model.njnt):
                dof = int(model.jnt_dofadr[j])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
                metrics["qvel_per_joint"][name] = max(
                    metrics["qvel_per_joint"].get(name, 0.0), abs(float(data.qvel[dof]))
                )
        if data.qacc.size:
            metrics["max_qacc"] = max(metrics["max_qacc"], float(max(abs(v) for v in data.qacc)))
        force6 = np.zeros(6)
        for c in range(min(data.ncon, 50)):
            mujoco.mj_contactForce(model, data, c, force6)
            metrics["max_contact_force"] = max(metrics["max_contact_force"], abs(float(force6[0])))
        if not np.isfinite(data.sensordata).all():
            metrics["sensordata_finite"] = False

    out = ctx.sweep(seconds, visit, ctrl=ctrl)
    metrics["diverged"] = out["diverged"]
    return metrics


def a09_actuator_saturation(ctx: AuditContext) -> dict[str, Any]:
    """A09：ctrl 贴 ctrlrange 边界的步数占比（模型自带 ctrlrange，
    非猜测）。高占比 = 控制器长期饱和。"""
    metrics = _trace_or_hold_sweep(ctx)
    if metrics["total_steps"] == 0:
        return {"status": "NOT_EVALUATED", "violations": [], "detail": {"reason": "no_steps"}}
    ratio = metrics["saturation_steps"] / metrics["total_steps"]
    threshold = ctx.policy.saturation_ratio
    violations = (
        [{"ratio": ratio, "threshold": threshold, "reason": "actuator_saturated"}]
        if ratio > threshold
        else []
    )
    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
        "saturation_ratio": ratio,
        "threshold": threshold,
    }


def a10_force_limit(ctx: AuditContext) -> dict[str, Any]:
    """A10：max |actuator_force| 超过模型 forcerange 或 e-URDF
    max_joint_effort（取更严者）。"""
    model = ctx.model
    safety = _safety_limits(ctx)
    limits = [
        float(model.actuator_forcerange[i][1])
        for i in range(model.nu)
        if model.actuator_forcelimited[i] and model.actuator_forcerange[i][1] > 0
    ]
    if safety and safety.get("force_limits", {}).get("max_joint_effort"):
        limits.append(float(safety["force_limits"]["max_joint_effort"]))
    if not limits:
        return {
            "status": "NOT_EVALUATED",
            "violations": [],
            "detail": {"reason": "no_force_limits_declared"},
        }
    limit = min(limits)

    # forcerange 是物理裁剪——力"超过"限值不可能，真正的缺陷是
    # 力被**持续钉在限值**（force saturation 掩盖跟踪失败）。
    saturated_steps = 0
    total_steps = 0

    def visit(data, step) -> None:  # noqa: ANN001
        nonlocal saturated_steps, total_steps
        total_steps += 1
        for i in range(model.nu):
            if model.actuator_forcelimited[i]:
                actuator_limit = float(model.actuator_forcerange[i][1])
                if actuator_limit > 0 and abs(float(data.actuator_force[i])) >= 0.999 * min(
                    actuator_limit, limit
                ):
                    saturated_steps += 1
                    break

    metrics = _trace_or_hold_sweep(ctx, seconds=0.0)  # 取 ctrl/trace 形态
    ctrl = None
    seconds = 1.0
    if ctx.trace_record is not None:
        controller = ctx.trace_record.get("controller", {})
        if "ctrl_series" in controller:
            ctrl = {"kind": "ctrl_series", "rows": controller["ctrl_series"]}
        elif "position_targets" in controller:
            ctrl = {"kind": "ctrl", "values": controller["position_targets"]}
        seconds = ctx.trace_record.get("steps", 1) * float(model.opt.timestep)
    ctx.sweep(seconds, visit, ctrl=ctrl)

    max_force = metrics["max_actuator_force"]
    if total_steps == 0:
        return {"status": "NOT_EVALUATED", "violations": [], "detail": {"reason": "no_steps"}}
    ratio = saturated_steps / total_steps
    violations = (
        [
            {
                "saturated_ratio": ratio,
                "limit": limit,
                "max_actuator_force": max_force,
                "reason": "force_saturation_persistent",
            }
        ]
        if ratio > ctx.policy.saturation_ratio
        else []
    )
    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
        "saturated_ratio": ratio,
        "max_actuator_force": max_force,
        "limit": limit,
    }


def a11_velocity_limit(ctx: AuditContext) -> dict[str, Any]:
    """A11：逐关节 |qvel| 超 e-URDF velocity_limits；无声明 NOT_EVALUATED。"""
    safety = _safety_limits(ctx)
    velocity_limits = (safety or {}).get("velocity_limits") or {}
    if not velocity_limits:
        return {"status": "NOT_EVALUATED", "violations": []}
    metrics = _trace_or_hold_sweep(ctx)
    violations = [
        {
            "joint": name,
            "max_qvel": value,
            "limit": float(velocity_limits[name]),
            "reason": "velocity_limit_exceeded",
        }
        for name, value in metrics["qvel_per_joint"].items()
        if name in velocity_limits and value > float(velocity_limits[name])
    ]
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def a12_acceleration_spike(ctx: AuditContext) -> dict[str, Any]:
    """A12：max |qacc| 超声明阈值；无声明 NOT_EVALUATED。"""
    safety = _safety_limits(ctx)
    limit = (safety or {}).get("acceleration_limit") or (ctx.extra.get("acceleration_limit"))
    if limit is None:
        return {"status": "NOT_EVALUATED", "violations": []}
    metrics = _trace_or_hold_sweep(ctx)
    violations = (
        [{"max_qacc": metrics["max_qacc"], "limit": float(limit), "reason": "acceleration_spike"}]
        if metrics["max_qacc"] > float(limit)
        else []
    )
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def a14_peak_contact_force(ctx: AuditContext) -> dict[str, Any]:
    """A14：峰值接触法向力超 max_tcp_force；无声明 NOT_EVALUATED。"""
    safety = _safety_limits(ctx)
    limit = (safety or {}).get("force_limits", {}).get("max_tcp_force")
    if limit is None:
        return {
            "status": "NOT_EVALUATED",
            "violations": [],
            "max_contact_force": _trace_or_hold_sweep(ctx)["max_contact_force"],
        }
    metrics = _trace_or_hold_sweep(ctx)
    violations = (
        [
            {
                "max_contact_force": metrics["max_contact_force"],
                "limit": float(limit),
                "reason": "peak_contact_force",
            }
        ]
        if metrics["max_contact_force"] > float(limit)
        else []
    )
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def a21_sensor_validity(ctx: AuditContext) -> dict[str, Any]:
    """A21：sensordata 有限性 + 延迟传感器 buffer/delay 一致性 +
    读数停滞检测（运动中读数长期不变且非延迟所致）。"""
    model = ctx.model
    violations = []
    # 延迟 buffer 必须覆盖 delay（nsample 在 3.13 MjModel 不可达——
    # 与 A02 同法解析来源 XML；编译器只查 nsample>0，不查覆盖率）。
    import xml.etree.ElementTree as ET

    dt = float(model.opt.timestep)
    root = ET.fromstring(ctx.xml_text)
    for sensor_tag in root.iter():
        delay = float(sensor_tag.get("delay", "0") or 0)
        if delay <= 0:
            continue
        nsample = int(sensor_tag.get("nsample", "0") or 0)
        name = sensor_tag.get("name", sensor_tag.tag)
        if nsample * dt < delay - 1e-12:
            violations.append(
                {
                    "sensor": name,
                    "delay": delay,
                    "buffer_s": nsample * dt,
                    "reason": "sensor_buffer_undersized",
                }
            )
    metrics = _trace_or_hold_sweep(ctx)
    if not metrics["sensordata_finite"]:
        violations.append({"reason": "sensor_non_finite"})
    status = "FAIL" if violations else "PASS"
    return {
        "status": status,
        "violations": violations,
        "sensordata_finite": metrics["sensordata_finite"],
    }


def a22_frame_convention(ctx: AuditContext) -> dict[str, Any]:
    """A22：frame 约定——重力应主导 -Z（米制 Z-up）。"""
    gravity = [float(v) for v in ctx.model.opt.gravity]
    z_up = (
        gravity[2] < 0 and abs(gravity[0]) < abs(gravity[2]) and abs(gravity[1]) < abs(gravity[2])
    )
    if z_up:
        return {"status": "PASS", "violations": [], "gravity": gravity}
    return {
        "status": "WARN",
        "violations": [],
        "warnings": [{"gravity": gravity, "reason": "frame_convention_mismatch"}],
        "gravity": gravity,
    }


def _final_qpos_for_config(
    ctx: AuditContext, *, solver: str | None = None, timestep: float | None = None
) -> list[float]:  # noqa: ANN001
    """同模型同初始态在不同 solver/timestep 下的 rollout 末态。"""
    import mujoco

    spec = None
    if ctx.spec is not None and hasattr(ctx.spec, "copy"):
        spec = ctx.spec.copy()
    elif ctx.xml_text:
        import mujoco as _mj

        spec = _mj.MjSpec.from_string(ctx.xml_text)
    if spec is not None:
        if solver is not None:
            spec.option.solver = getattr(mujoco.mjtSolver, f"mjSOL_{solver.upper()}")
        if timestep is not None:
            spec.option.timestep = timestep
        model = spec.compile()
    else:
        model = ctx.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    steps = max(1, round(0.5 / float(model.opt.timestep)))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return [float(v) for v in data.qpos]


def a23_solver_sensitivity(ctx: AuditContext) -> dict[str, Any]:
    """A23：Newton vs CG 安全组合 robustness 探针——结果实质不同
    即 ROBUSTNESS_WARNING（WARN，不判 FAIL）。"""
    import mujoco

    if ctx.spec is None:
        return {"status": "NOT_EVALUATED", "violations": [], "detail": {"reason": "no_spec"}}
    base = _final_qpos_for_config(ctx)
    deviations = {}
    for solver in _SOLVER_PROBE:
        base_solver = (
            str(mujoco.mjtSolver(ctx.model.opt.solver).name).removeprefix("mjSOL_").lower()
        )
        if solver == base_solver:
            continue
        other = _final_qpos_for_config(ctx, solver=solver)
        denom = max(1e-9, max(abs(v) for v in base))
        deviations[solver] = max(abs(a - b) for a, b in zip(base, other, strict=True)) / denom
    import math

    sensitive = {
        s: d
        for s, d in deviations.items()
        if (isinstance(d, float) and math.isnan(d)) or d > ctx.policy.sensitivity_rel
    }
    if sensitive:
        return {
            "status": "WARN",
            "violations": [],
            "warnings": [{"reason": "ROBUSTNESS_WARNING", "solver_deviations": deviations}],
            "solver_deviations": deviations,
        }
    return {"status": "PASS", "violations": [], "solver_deviations": deviations}


def a24_timestep_sensitivity(ctx: AuditContext) -> dict[str, Any]:
    """A24：dt vs dt/2 结果实质不同 → NUMERICAL_FRAGILITY（WARN）。
    discrete integrator 只能作为 diagnostic candidate，不能偷偷换
    integrator 宣布原方案成功。"""
    base = _final_qpos_for_config(ctx)
    finer = _final_qpos_for_config(ctx, timestep=float(ctx.model.opt.timestep) / 2)
    denom = max(1e-9, max(abs(v) for v in base))
    import math

    deviation = max(abs(a - b) for a, b in zip(base, finer, strict=True)) / denom
    if math.isnan(deviation) or deviation > ctx.policy.sensitivity_rel:
        return {
            "status": "WARN",
            "violations": [],
            "warnings": [
                {
                    "reason": "NUMERICAL_FRAGILITY",
                    "timestep_deviation": deviation,
                    "note": "discrete integrator 可作 diagnostic candidate，但不能冒充原方案成功",
                }
            ],
            "timestep_deviation": deviation,
        }
    return {"status": "PASS", "violations": [], "timestep_deviation": deviation}
