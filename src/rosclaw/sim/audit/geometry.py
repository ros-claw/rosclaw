"""几何类 audit（PR-MH4，规格 §17.1）：A01/A02/A07/A08。

吸收 Text2Mujoco（MIT）碰撞边界/显式质量/自重叠/marker 接地
四项检查的思想，按 ROSClaw 后端模型改写。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from rosclaw.sim.audit.context import AuditContext


def _body_name(model, body_id: int) -> str:  # noqa: ANN001
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def _geom_name(model, geom_id: int) -> str:  # noqa: ANN001
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def a01_collision_coverage(ctx: AuditContext) -> dict[str, Any]:
    """A01：所有 moving body 必须有有效 collider。

    视觉 geom 必须被同 body 某碰撞体的包围球覆盖，否则报
    uncovered_visual_geoms。
    """
    model = ctx.model
    violations: list[dict[str, Any]] = []
    for body_id in range(1, model.nbody):
        if int(model.body_weldid[body_id]) == 0:
            continue  # 不动的 body 不需要 collider
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]
        if not geoms:
            continue
        colliders = [
            g
            for g in geoms
            if int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0
        ]
        if not colliders:
            violations.append(
                {"body": _body_name(model, body_id), "reason": "collision_filtered_away"}
            )
            continue
        for geom_id in geoms:
            if geom_id in colliders:
                continue
            covered = any(
                float(np.linalg.norm(model.geom_pos[geom_id] - model.geom_pos[c]))
                <= float(model.geom_rbound[c]) + 1e-9
                for c in colliders
            )
            if not covered:
                violations.append(
                    {
                        "body": _body_name(model, body_id),
                        "geom": _geom_name(model, geom_id),
                        "reason": "uncovered_visual_geom",
                    }
                )
    status = "FAIL" if violations else "PASS"
    return {
        "status": status,
        "violations": violations,
        "detail": {"checked_bodies": model.nbody - 1},
    }


def a02_explicit_mass(ctx: AuditContext) -> dict[str, Any]:
    """A02：带 joint/freejoint 的 body 下所有 geom 必须显式声明
    mass 或 density——不允许无意依赖 MuJoCo 默认密度。"""
    root = ET.fromstring(ctx.xml_text)
    violations: list[dict[str, Any]] = []
    for body in root.iter("body"):
        has_joint = body.find("joint") is not None or body.find("freejoint") is not None
        if not has_joint:
            continue
        body_name = body.get("name", "?")
        for geom in body.iter("geom"):
            if geom.get("mass") is None and geom.get("density") is None:
                violations.append(
                    {
                        "body": body_name,
                        "geom": geom.get("name", "?"),
                        "reason": "implicit_mass_or_density",
                    }
                )
    status = "FAIL" if violations else "PASS"
    return {"status": status, "violations": violations, "detail": {}}


def _declared_exclusions(model) -> set[tuple[int, int]]:  # noqa: ANN001
    pairs = set()
    for i in range(model.nexclude):
        signature = int(model.exclude_signature[i])
        body_low, body_high = signature & 0xFFFF, signature >> 16
        pairs.add((min(body_low, body_high), max(body_low, body_high)))
    return pairs


def a07_undeclared_self_overlap(ctx: AuditContext) -> dict[str, Any]:
    """A07：被 collision filter 静默隐藏、但既非焊接邻居也无显式
    <contact><exclude> 声明的 geom 对——实测重叠即 FAIL。"""
    import mujoco

    model = ctx.model
    data = ctx.fresh_data()
    exclusions = _declared_exclusions(model)
    violations: list[dict[str, Any]] = []
    masked_benign: list[dict[str, Any]] = []
    for g1 in range(model.ngeom):
        for g2 in range(g1 + 1, model.ngeom):
            contype1, conaff1 = int(model.geom_contype[g1]), int(model.geom_conaffinity[g1])
            contype2, conaff2 = int(model.geom_contype[g2]), int(model.geom_conaffinity[g2])
            filtered = (contype1 & conaff2) == 0 and (contype2 & conaff1) == 0
            if not filtered:
                continue
            body1, body2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if body1 == body2 or int(model.body_weldid[body1]) == int(model.body_weldid[body2]):
                continue  # 同体 / 焊接邻居
            pair = (min(body1, body2), max(body1, body2))
            if pair in exclusions:
                continue  # 显式声明的排除
            dist = float(mujoco.mj_geomDistance(model, data, g1, g2, 0.02, None))
            entry = {
                "geom1": _geom_name(model, g1),
                "geom2": _geom_name(model, g2),
                "body1": _body_name(model, body1),
                "body2": _body_name(model, body2),
                "distance_m": dist,
            }
            if dist < ctx.policy.self_overlap_m:
                violations.append(entry)
            else:
                masked_benign.append(entry)
    if violations:
        status = "FAIL"
    elif masked_benign:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "violations": violations,
        "warnings": masked_benign,
        "detail": {"hidden_pairs": len(violations) + len(masked_benign)},
    }


def _point_burial_depth(model, data, geom_id: int, point: np.ndarray) -> float:  # noqa: ANN001
    """点在 geom 内部的深度（正值=埋入；不支持的类型返回 0）。"""
    import mujoco

    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    local = mat.T @ (point - pos)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return float(-local[2])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0] - np.linalg.norm(local))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.min(size - np.abs(local)))
    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        radial = float(np.linalg.norm(local[:2]))
        axial_gap = abs(local[2]) - size[1]
        radial_gap = size[0] - radial
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return float(min(radial_gap, size[1] - abs(local[2])))
        # capsule：端帽半球
        if axial_gap <= 0:
            return float(radial_gap)
        cap_center = np.array([0.0, 0.0, np.sign(local[2]) * size[1]])
        return float(size[0] - np.linalg.norm(local - cap_center))
    return 0.0


def a08_marker_grounding(ctx: AuditContext) -> dict[str, Any]:
    """A08：交互 marker（site group==policy.marker_group）不能漂浮、
    不能埋入，必须对应真实 surface。"""
    import mujoco

    model = ctx.model
    data = ctx.fresh_data()
    markers = [s for s in range(model.nsite) if int(model.site_group[s]) == ctx.policy.marker_group]
    if not markers:
        return {"status": "PASS", "violations": [], "detail": {"note": "no_marker_sites"}}
    violations: list[dict[str, Any]] = []
    down = np.array([0.0, 0.0, -1.0])
    for site_id in markers:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or f"site_{site_id}"
        point = np.asarray(data.site_xpos[site_id], dtype=float)
        geomid = np.zeros(1, dtype=np.int32)
        dist = float(mujoco.mj_ray(model, data, point, down, None, 1, -1, geomid))
        if dist < 0 or dist > ctx.policy.marker_clearance_m:
            violations.append(
                {"site": name, "reason": "floating", "clearance_m": dist if dist >= 0 else None}
            )
            continue
        for geom_id in range(model.ngeom):
            depth = _point_burial_depth(model, data, geom_id, point)
            if depth > ctx.policy.marker_burial_m:
                violations.append(
                    {
                        "site": name,
                        "reason": "buried",
                        "geom": _geom_name(model, geom_id),
                        "depth_m": depth,
                    }
                )
                break
    status = "FAIL" if violations else "PASS"
    return {"status": status, "violations": violations, "detail": {"markers": len(markers)}}
