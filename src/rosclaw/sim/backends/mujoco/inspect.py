"""全量模型检查（PR-MH2，ADR-0014，规格 §12.2）。

Maturity: experimental（ADR-0000 §4）。

在编译后 ``MjModel`` 上推导（与 ``sim.model_inspect.inspect_mjcf``
同一原则：从实际编译结果推导，不从机器人名字猜）。
``inspect_mjcf`` 形状冻结（tests/agentd/test_w02_contracts.py 精确
相等护栏），本模块是叠加层，不改它。

``summary_cn`` 仅供 Agent 理解；真实判断必须使用结构化字段。
"""

from __future__ import annotations

from typing import Any

_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}  # 与 model_inspect 保持一致


def _name(model, objtype, index: int, fallback: str) -> str:  # noqa: ANN001, ANN202
    import mujoco

    return mujoco.mj_id2name(model, objtype, index) or fallback


def inspect_model_full(model, *, model_digest: str = "", spec=None) -> dict[str, Any]:  # noqa: ANN001
    """对编译后 MjModel 做 §12.2 全量检查，返回纯结构化 dict。

    geom 的 mass/density 在编译时被并入 body 惯性，MjModel 不再保留
    逐 geom 值；传入来源 ``spec``（可选）时按名字补齐这两个字段。
    """
    import mujoco

    joints_detail: list[dict[str, Any]] = []
    for i in range(model.njnt):
        dof_addr = int(model.jnt_dofadr[i])
        entry: dict[str, Any] = {
            "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, i, f"joint_{i}"),
            "type": _JOINT_TYPES.get(int(model.jnt_type[i]), str(int(model.jnt_type[i]))),
            "qpos_addr": int(model.jnt_qposadr[i]),
            "dof_addr": dof_addr,
            "range": [float(model.jnt_range[i][0]), float(model.jnt_range[i][1])],
            "limited": bool(model.jnt_limited[i]),
        }
        # 单 dof 关节（hinge/slide）的阻尼落在 dof_damping。
        if entry["type"] in ("hinge", "slide"):
            entry["damping"] = float(model.dof_damping[dof_addr])
        joints_detail.append(entry)

    actuators_detail: list[dict[str, Any]] = []
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i][0])
        gaintype = mujoco.mjtGain(model.actuator_gaintype[i])
        biastype = mujoco.mjtBias(model.actuator_biastype[i])
        kp = 0.0
        kv = 0.0
        if gaintype == mujoco.mjtGain.mjGAIN_FIXED:
            kp = float(model.actuator_gainprm[i][0])
            if kp == 0.0 and biastype == mujoco.mjtBias.mjBIAS_AFFINE:
                kp = -float(model.actuator_biasprm[i][1])
        if biastype == mujoco.mjtBias.mjBIAS_AFFINE:
            kv = -float(model.actuator_biasprm[i][2])
        actuators_detail.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i, f"actuator_{i}"),
                "trntype": str(mujoco.mjtTrn(model.actuator_trntype[i]).name)
                .removeprefix("mjTRN_")
                .lower(),
                "target": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, f"joint_{joint_id}")
                if joint_id >= 0
                else "",
                "ctrlrange": [
                    float(model.actuator_ctrlrange[i][0]),
                    float(model.actuator_ctrlrange[i][1]),
                ],
                "forcerange": [
                    float(model.actuator_forcerange[i][0]),
                    float(model.actuator_forcerange[i][1]),
                ],
                "forcelimited": bool(model.actuator_forcelimited[i]),
                "gear": [float(v) for v in model.actuator_gear[i]],
                "kp": kp,
                "kv": kv,
            }
        )

    import math

    spec_geom_mass: dict[str, tuple[float | None, float | None]] = {}
    if spec is not None:
        for element in spec.geoms:
            raw_mass = getattr(element, "mass", None)
            mass = float(raw_mass) if raw_mass is not None else None
            if mass is not None and math.isnan(mass):
                mass = None  # 未显式声明（编译器按 density 推导）
            raw_density = getattr(element, "density", None)
            density = float(raw_density) if raw_density is not None else None
            spec_geom_mass[element.name or ""] = (mass, density)

    geoms: list[dict[str, Any]] = []
    for i in range(model.ngeom):
        body_id = int(model.geom_bodyid[i])
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, i, f"geom_{i}")
        entry: dict[str, Any] = {
            "name": geom_name,
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
            "type": str(mujoco.mjtGeom(model.geom_type[i]).name).removeprefix("mjGEOM_").lower(),
            "size": [float(v) for v in model.geom_size[i]],
            "friction": [float(v) for v in model.geom_friction[i]],
            "rgba": [float(v) for v in model.geom_rgba[i]],
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
        }
        if geom_name in spec_geom_mass:
            mass, density = spec_geom_mass[geom_name]
            entry["mass"] = mass  # None = 未显式声明
            entry["density"] = density
        geoms.append(entry)

    bodies: list[dict[str, Any]] = []
    for i in range(model.nbody):
        parent_id = int(model.body_parentid[i])
        bodies.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, i, f"body_{i}"),
                "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id, "")
                if parent_id >= 0
                else "",
                "pos": [float(v) for v in model.body_pos[i]],
                "quat": [float(v) for v in model.body_quat[i]],
            }
        )

    children_of: dict[int, list[int]] = {}
    for i in range(1, model.nbody):
        children_of.setdefault(int(model.body_parentid[i]), []).append(i)

    def _tree(body_id: int) -> dict[str, Any]:
        return {
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
            "children": [_tree(c) for c in children_of.get(body_id, [])],
        }

    sensors_detail = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_SENSOR, i, f"sensor_{i}"),
            "type": str(mujoco.mjtSensor(model.sensor_type[i]).name)
            .removeprefix("mjSENS_")
            .lower(),
        }
        for i in range(model.nsensor)
    ]
    cameras_detail = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_CAMERA, i, f"camera_{i}"),
            "pos": [float(v) for v in model.cam_pos[i]],
            "quat": [float(v) for v in model.cam_quat[i]],
        }
        for i in range(model.ncam)
    ]
    sites_detail = [
        {"name": _name(model, mujoco.mjtObj.mjOBJ_SITE, i, f"site_{i}")} for i in range(model.nsite)
    ]

    equality = []
    for i in range(model.neq):
        eq: dict[str, Any] = {
            "type": str(mujoco.mjtEq(model.eq_type[i]).name).removeprefix("mjEQ_").lower(),
            "active": bool(model.eq_active0[i]),
        }
        equality.append(eq)

    tendons = [
        _name(model, mujoco.mjtObj.mjOBJ_TENDON, i, f"tendon_{i}") for i in range(model.ntendon)
    ]

    contact_excludes = []
    for i in range(model.nexclude):
        signature = int(model.exclude_signature[i])
        body1, body2 = signature & 0xFFFF, signature >> 16
        contact_excludes.append(
            {
                "body1": _name(model, mujoco.mjtObj.mjOBJ_BODY, body1, f"body_{body1}"),
                "body2": _name(model, mujoco.mjtObj.mjOBJ_BODY, body2, f"body_{body2}"),
            }
        )

    keyframes = [_name(model, mujoco.mjtObj.mjOBJ_KEY, i, f"key_{i}") for i in range(model.nkey)]

    opt = model.opt
    options = {
        "timestep": float(opt.timestep),
        "solver": str(mujoco.mjtSolver(opt.solver).name).removeprefix("mjSOL_").lower(),
        "integrator": str(mujoco.mjtIntegrator(opt.integrator).name).removeprefix("mjINT_").lower(),
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
    }

    detail: dict[str, Any] = {
        "model_digest": model_digest,
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "body_tree": _tree(0),
        "bodies": bodies,
        "joints_detail": joints_detail,
        "actuators_detail": actuators_detail,
        "geoms": geoms,
        "sensors_detail": sensors_detail,
        "cameras_detail": cameras_detail,
        "sites_detail": sites_detail,
        "equality": equality,
        "tendons": tendons,
        "contact_excludes": contact_excludes,
        "keyframes": keyframes,
        "options": options,
    }
    detail["summary_cn"] = summarize_cn(detail)
    return detail


def summarize_cn(detail: dict[str, Any]) -> str:
    """Agent 友好的中文摘要——仅供理解，真实判断用结构化字段。"""
    type_count: dict[str, int] = {}
    for joint in detail["joints_detail"]:
        type_count[joint["type"]] = type_count.get(joint["type"], 0) + 1
    type_desc = "、".join(f"{count} 个 {jtype}" for jtype, count in type_count.items()) or "无"
    actuated = {a["target"] for a in detail["actuators_detail"] if a["target"]}
    gripper = any("gripper" in target for target in actuated)
    cameras = "、".join(c["name"] for c in detail["cameras_detail"]) or "无"
    sites = "、".join(s["name"] for s in detail["sites_detail"]) or "无"
    return (
        f"该模型有 {detail['njnt']} 个关节（{type_desc}），"
        f"{detail['nq']} 个广义坐标、{detail['nv']} 个自由度速度维、"
        f"{detail['nu']} 个执行器；"
        f"{'有' if gripper else '无'}夹爪执行器；"
        f"相机：{cameras}；site：{sites}；"
        f"timestep={detail['options']['timestep']}，"
        f"solver={detail['options']['solver']}，"
        f"integrator={detail['options']['integrator']}。"
        f"（摘要仅供理解；权威数据见结构化字段。）"
    )
