"""SIM 动力学闭环正式能力（十四审 PR-14.6，总纲 §5.3/§7 PR-14.6）。

四个注册能力（全部 COMPUTE 类、SIMULATION 限定、确定性、无副作用）：

1. trajectory.generate_planar_path——形状参数化平面轨迹（star5/circle
   同一组合，不为每个形状写新仿真器）；WP-5 起产出 SE(3)
   PoseTrajectorySpecV1（位置+朝向+approach/contact/lift 语义段）；
2. ur5e.simulate_cartesian_trajectory——真实 MuJoCo 动力学 rollout
   （SIM_DYN_ROLLOUT 证据，不是命令回放）：6-DOF DLS-IK 把位姿轨迹
   转成关节轨迹，MujocoCpuBackend 物理推演，FK 还原实际 eef 位姿
   （位置+朝向）；
3. simulation.render_trace——实际 eef 轨迹渲染 GIF（可打开的产物）；
4. simulation.verify_tracking——跟踪误差阈值判定（位置+朝向，
   PASS/FAIL 诚实）。

交付物全部落盘（rosclaw_home/sim/）：trace.json/trace.csv/metrics.json/
GIF——部分成果产品化，不是只有 bash-log。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

#: R0-1.5（金丝雀实证：模型传 'xz'/'vertical' 被拒）——命名平面
#: xy/xz/yz 映射为法向（任意法向走 plane_normal_xyz）。
_NAMED_PLANE_NORMALS = {
    "xy": [0.0, 0.0, 1.0],
    "xz": [0.0, 1.0, 0.0],
    "yz": [1.0, 0.0, 0.0],
}

#: 与 ur5e_mcp 一致的安全工作空间（规划即拒越界）。
_SAFE_RADIUS = (0.10, 0.80)
_SAFE_Z = (0.02, 1.20)
#: UR5e home 关节角（与 sandbox keyframe 一致）。
_HOME_QPOS = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
_SHAPES = ("star5", "circle")
#: approach/lift 抬升高度（接触平面上方，米）。
_APPROACH_LIFT_M = 0.15


# ----------------------------------------------------------------------
# SE(3) 数学助手（WP-5）——纯 stdlib，规划路径不依赖 mujoco/numpy。
# ----------------------------------------------------------------------
def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        raise ValueError(f"zero vector not normalizable: {v}")
    return [x / n for x in v]


def _mat_to_quat_xyzw(r: list[list[float]]) -> list[float]:
    """行主序 3x3 旋转矩阵 → xyzw 单位四元数。"""
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return [
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
            0.25 * s,
        ]
    if r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        return [
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[2][1] - r[1][2]) / s,
        ]
    if r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        return [
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
            (r[0][2] - r[2][0]) / s,
        ]
    s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
    return [
        (r[0][2] + r[2][0]) / s,
        (r[1][2] + r[2][1]) / s,
        0.25 * s,
        (r[1][0] - r[0][1]) / s,
    ]


def _quat_xyzw_to_mat(q: list[float]) -> list[list[float]]:
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _quat_mul_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _tool_down_orientation(normal_xyz: list[float]) -> list[float]:
    """工具 z 轴对准 -法向（垂直接触平面）的目标朝向（xyzw）。

    x 轴取世界 x 在平面上的投影（退化时取世界 y）——绕工具轴的
    旋转对平面作业无关，取确定性约定。
    """
    z_d = _normalize([-n for n in normal_xyz])
    ref = [1.0, 0.0, 0.0]
    if abs(_dot(ref, z_d)) > 0.9:
        ref = [0.0, 1.0, 0.0]
    d = _dot(ref, z_d)
    x_d = _normalize([ref[i] - d * z_d[i] for i in range(3)])
    y_d = _cross(z_d, x_d)
    return _mat_to_quat_xyzw(
        [
            [x_d[0], y_d[0], z_d[0]],
            [x_d[1], y_d[1], z_d[1]],
            [x_d[2], y_d[2], z_d[2]],
        ]
    )


def _tool_z_of_quat(q: list[float]) -> list[float]:
    """四元数（xyzw）表达的工具 z 轴在世界系的方向。"""
    x, y, z, w = q
    return [
        2 * (x * z + y * w),
        2 * (y * z - x * w),
        1 - 2 * (x * x + y * y),
    ]


def _plane_basis(normal_xyz: list[float]) -> tuple[list[float], list[float]]:
    """法向 → 平面正交基（e1,e2——任意平面的一等表达，§15.4）。

    normal=[0,0,1] 时 e1=[1,0,0]/e2=[0,1,0]（xy 默认行为不变）。
    """
    normal = _normalize(normal_xyz)
    # 取一个不与法向平行的参考轴叉乘出 e1。
    reference = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = _normalize(_cross(reference, normal))
    e2 = _normalize(_cross(normal, e1))
    return e1, e2


def _build_pose_spec(points: list[dict], *, plane: str) -> dict:
    """接触点列 → PoseTrajectorySpecV1（approach/contact/lift 显式段）。

    当前只支持 xy 水平接触平面（法向 +z）——规格本身支持任意
    平面，生成器对非 xy 诚实拒绝（见 generate_planar_path）。
    """
    if plane != "xy":
        raise ValueError(f"unsupported plane {plane!r} (supported: xy)")
    normal = [0.0, 0.0, 1.0]
    return _build_pose_spec_3d(points, normal_xyz=normal)


def _build_pose_spec_3d(
    points: list[dict],
    *,
    normal_xyz: list[float],
    geometry: str = "shape",
    max_segment_m: float | None = None,
    speed_mps: float | None = None,
    accel_mps2: float | None = None,
) -> dict:
    """任意平面 PoseTrajectorySpecV1（§15.4）——approach/contact/lift
    沿平面法向；xy 是 normal=[0,0,1] 的特例。"""
    normal = _normalize(normal_xyz)
    quat = _tool_down_orientation(normal)
    start, end = points[0], points[-1]
    # 平面 offset = 首点在法向上的投影（n·p）。
    offset = _dot([start["x"], start["y"], start["z"]], normal)

    def _at(point: dict, lift: float) -> list[float]:
        return [
            point["x"] + normal[0] * lift,
            point["y"] + normal[1] * lift,
            point["z"] + normal[2] * lift,
        ]

    waypoints: list[dict] = [
        {"position_m": _at(start, _APPROACH_LIFT_M), "orientation_xyzw": quat, "kind": "approach"},
        {"position_m": _at(start, 0.0), "orientation_xyzw": quat, "kind": "approach"},
    ]
    waypoints += [
        {"position_m": [p["x"], p["y"], p["z"]], "orientation_xyzw": quat, "kind": "contact"}
        for p in points
    ]
    waypoints.append(
        {
            "position_m": _at(end, _APPROACH_LIFT_M),
            "orientation_xyzw": quat,
            "kind": "lift",
        }
    )
    digest = hashlib.sha256(json.dumps(waypoints, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": "rosclaw.pose_trajectory_spec.v1",
        "frame_id": "world",
        "tool_frame": "attachment_site",
        "contact_plane": {
            "schema_version": "rosclaw.contact_plane.v1",
            "normal_xyz": normal,
            "offset_m": offset,
        },
        "waypoints": waypoints,
        "digest": f"sha256:{digest}",
        "geometry": geometry,
        "plane_pose": {
            "position_m": [normal[0] * offset, normal[1] * offset, normal[2] * offset],
            "orientation_xyzw": quat,
        },
        "tool_pose_constraint": {"axis": "tool_z", "tolerance_deg": 5.0},
        "sampling": {"max_segment_m": max_segment_m},
        "timing": {"speed_mps": speed_mps, "accel_mps2": accel_mps2},
    }


def _workspace_check(point: dict) -> None:
    radius = math.hypot(point["x"], point["y"])
    if not (_SAFE_RADIUS[0] <= radius <= _SAFE_RADIUS[1]):
        raise ValueError(
            f"point ({point['x']:.3f},{point['y']:.3f}) radius {radius:.3f}m "
            f"outside safe workspace {_SAFE_RADIUS}"
        )
    if not (_SAFE_Z[0] <= point["z"] <= _SAFE_Z[1]):
        raise ValueError(f"point z={point['z']}m outside safe window {_SAFE_Z}")


def _shape_vertices(shape: str, center: list[float], scale: float) -> list[dict]:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    vertices: list[dict] = []
    if shape == "star5":
        inner = scale * 0.381966
        for k in range(10):
            angle = math.radians(90 + k * 36)
            r = scale if k % 2 == 0 else inner
            vertices.append({"x": cx + r * math.cos(angle), "y": cy + r * math.sin(angle), "z": cz})
    elif shape == "circle":
        # 解析圆弧采样（弦插值会落入圆内——圆走精确弧，不走折线）。
        step = max(1, math.ceil(2 * math.pi * scale / 0.02))
        for k in range(step):
            angle = 2 * math.pi * k / step
            vertices.append(
                {"x": cx + scale * math.cos(angle), "y": cy + scale * math.sin(angle), "z": cz}
            )
    else:
        raise ValueError(f"unsupported shape {shape!r} (supported: {', '.join(_SHAPES)})")
    vertices.append(dict(vertices[0]))  # 闭合
    return vertices


def _sample_segments(vertices: list[dict], max_segment_m: float) -> list[dict]:
    points: list[dict] = [vertices[0]]
    for a, b in zip(vertices, vertices[1:], strict=False):
        seg = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
        steps = max(1, math.ceil(seg / max_segment_m))
        for i in range(1, steps + 1):
            ratio = i / steps
            points.append(
                {
                    "x": a["x"] + (b["x"] - a["x"]) * ratio,
                    "y": a["y"] + (b["y"] - a["y"]) * ratio,
                    "z": a["z"] + (b["z"] - a["z"]) * ratio,
                }
            )
    return points


def _ik_waypoints_6d(model, data, poses: list[dict], site_id: int) -> list[list[float]]:
    """6-DOF 阻尼最小二乘 IK（WP-5）：SE(3) 位姿航点 → 关节轨迹。

    位置 + 朝向同时跟踪（朝向误差用世界系四元数误差
    2·vec(q_target ⊗ q_cur⁻¹)，单调到 180° 不退化）。从 home
    顺序求解，每点以上一点为初值——轨迹连续不跳变。不收敛即
    编译期失败（零执行），诚实报错。
    """
    import mujoco
    import numpy as np

    nu = int(model.nu)
    q = np.array(_HOME_QPOS[:nu], dtype=float)
    waypoints: list[list[float]] = []
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lam = 0.05
    for pose in poses:
        target_p = np.array(pose["position_m"], dtype=float)
        target_q = list(pose["orientation_xyzw"])
        for _ in range(240):
            data.qpos[:nu] = q
            mujoco.mj_forward(model, data)
            err_p = target_p - np.array(data.site_xpos[site_id])
            r_cur = np.array(data.site_xmat[site_id]).reshape(3, 3).tolist()
            q_cur = _mat_to_quat_xyzw(r_cur)
            q_err = _quat_mul_xyzw(target_q, [-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]])
            if q_err[3] < 0.0:  # 半球固定——走最短弧
                q_err = [-v for v in q_err]
            err_r = 2.0 * np.array(q_err[:3])
            if float(np.linalg.norm(err_p)) < 1e-3 and float(np.linalg.norm(err_r)) < 0.02:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, :nu], jacr[:, :nu]])
            err = np.concatenate([err_p, err_r])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * lam * np.eye(6), err)
            dq = np.clip(dq, -0.10, 0.10)
            q = q + dq
            q = np.clip(
                q,
                model.actuator_ctrlrange[:nu, 0],
                model.actuator_ctrlrange[:nu, 1],
            )
        else:
            pos = pose["position_m"]
            raise ValueError(
                f"IK 未收敛于 ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                "（位置+朝向）——路径不可达（编译期失败，零执行）"
            )
        waypoints.append(q.copy().tolist())
    return waypoints


class SimTrajectoryService:
    """SIM 轨迹能力的确定性实现（文件落盘，重启可审计）。"""

    def __init__(self, home: Path, *, runtime_manager=None) -> None:
        self._home = Path(home)
        self._plans_dir = self._home / "sim" / "plans"
        self._traces_dir = self._home / "sim" / "traces"
        # 十六审 P0-C：渲染依赖（Pillow）由托管 runtime 提供——不是
        # agentd 环境碰运气，更不是 Worker 去装。
        self._runtime_manager = runtime_manager

    def _import_pil(self):
        """PIL 导入（P0-F）：Pillow 是主包依赖（安装阶段闭包）——
        任务期间绝不安装；缺失 → RENDER_DEPS_MISSING 诚实失败
        （安装损坏，重装一致构建，不是任务期 pip install）。"""
        try:
            from PIL import Image, ImageDraw

            return Image, ImageDraw
        except ImportError as exc:
            raise ValueError(
                "RENDER_DEPS_MISSING: Pillow 不可用——安装阶段依赖"
                "闭包破损（请重新安装一致构建；任务期间不安装依赖）"
            ) from exc

    # --------------------------------------------------------------
    # 1. trajectory.generate_planar_path
    # --------------------------------------------------------------
    def generate_planar_path(
        self,
        *,
        shape: str | None = None,
        center_m: list[float] | None = None,
        scale_m: float | None = None,
        plane: str = "xy",
        max_segment_m: float = 0.02,
        plane_normal_xyz: list[float] | None = None,
        plane_offset_m: float | None = None,
        waypoints: list[dict] | None = None,
    ) -> dict[str, Any]:
        # 大道至简 R1-1：任意路径点是通用入口——画什么由 Pi 决定，
        # 形状枚举只是便捷参数（内部同样生成 waypoints）。
        if waypoints is not None:
            return self.generate_waypoint_path(
                waypoints=waypoints, max_segment_m=max_segment_m,
            )
        if shape is None or center_m is None or scale_m is None:
            raise ValueError(
                "shape+center_m+scale_m 或 waypoints 必须给一组——"
                "无默认形状（0905 实证：静默兜底 star5 会把 hello 画成"
                "五角星）"
            )
        if shape not in _SHAPES:
            raise ValueError(f"unsupported shape {shape!r} (supported: {', '.join(_SHAPES)})")
        if plane != "xy" and plane_normal_xyz is None:
            normal = _NAMED_PLANE_NORMALS.get(plane)
            if normal is None:
                raise ValueError(
                    f"unsupported plane {plane!r} (supported: "
                    f"{', '.join(_NAMED_PLANE_NORMALS)} 或 plane_normal_xyz)"
                )
            plane_normal_xyz = normal
            if plane_offset_m is None:
                # 命名竖直面的 offset = 中心在法向轴上的投影
                # （axis = 不在面名中的那个轴，xyz 轴序索引）。
                axis = "xyz".index(
                    next(c for c in "xyz" if c not in plane)
                )
                plane_offset_m = float(list(center_m)[axis])
        if not isinstance(center_m, list | tuple) or len(center_m) != 3:
            raise ValueError("center_m must be [x, y, z]")
        if not (0.02 <= float(scale_m) <= 0.35):
            raise ValueError(f"scale_m {scale_m} outside range [0.02, 0.35]")
        if not (0.005 <= float(max_segment_m) <= 0.1):
            raise ValueError("max_segment_m outside range [0.005, 0.1]")
        if plane_normal_xyz is None:
            vertices = _shape_vertices(shape, list(center_m), float(scale_m))
            # 圆已是精确弧采样；其余走线段插值。
            points = (
                vertices if shape == "circle" else _sample_segments(vertices, float(max_segment_m))
            )
            for point in points:
                _workspace_check(point)
            # WP-5：SE(3) 位姿规格——approach/contact/lift 显式段 +
            # 工具朝向 + 接触平面，随 plan 持久化（内容寻址可反查）。
            spec = _build_pose_spec(points, plane=plane)
        else:
            # §15.4 任意平面：法向单位化 → 正交基 → 局部 (u,v) 顶点
            # 映到世界坐标。center 在平面上的投影是形状中心。
            norm = math.sqrt(sum(v * v for v in plane_normal_xyz))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(f"plane normal 非单位向量: {plane_normal_xyz}")
            normal = _normalize(plane_normal_xyz)
            e1, e2 = _plane_basis(normal)
            offset = (
                float(plane_offset_m)
                if plane_offset_m is not None
                else _dot(list(center_m), normal)
            )
            # 形状中心在平面内：center - (center·n - offset) n
            proj = _dot(list(center_m), normal) - offset
            center_on_plane = [
                float(center_m[0]) - proj * normal[0],
                float(center_m[1]) - proj * normal[1],
                float(center_m[2]) - proj * normal[2],
            ]
            local_vertices = _shape_vertices(shape, [0.0, 0.0, 0.0], float(scale_m))
            local_points = (
                local_vertices
                if shape == "circle"
                else _sample_segments(local_vertices, float(max_segment_m))
            )
            points = [
                {
                    "x": center_on_plane[0] + p["x"] * e1[0] + p["y"] * e2[0],
                    "y": center_on_plane[1] + p["x"] * e1[1] + p["y"] * e2[1],
                    "z": center_on_plane[2] + p["x"] * e1[2] + p["y"] * e2[2],
                }
                for p in local_points
            ]
            for point in points:
                _workspace_check(point)
            spec = _build_pose_spec_3d(
                points,
                normal_xyz=normal,
                max_segment_m=max_segment_m,
            )
        digest = hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest()
        plan_id = f"plan_{digest[:16]}"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        (self._plans_dir / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "shape": shape,
                    "center_m": list(center_m),
                    "scale_m": scale_m,
                    "plane": plane,
                    "max_segment_m": max_segment_m,
                    "points": points,
                    "hash": digest,
                    "spec": spec,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "plan_id": plan_id,
            "hash": digest,
            "points": points,
            "point_count": len(points),
            "spec": spec,
            "summary": (
                f"{shape}：中心 ({center_m[0]}, {center_m[1]}, {center_m[2]})m，"
                f"半径 {scale_m}m，{len(points)} 个插值点，已闭合"
            ),
        }

    def generate_waypoint_path(
        self,
        *,
        waypoints: list[dict],
        max_segment_m: float = 0.02,
    ) -> dict[str, Any]:
        """任意路径点规划（大道至简 R1-1——「plan_cartesian 必须接收
        任意路径点，而不是只接受 star5/circle」）。

        waypoints: [{x, y, z, contact?}]——contact=False 的目标是
        抬笔移动段（多笔画文字/图案的笔段间隔）。每个点过安全工作
        空间校验（规划即拒越界），按 max_segment_m 插值，canonical
        hash + PlanStore 持久化（与形状路径同一消费面：simulate/
        render/verify 不需要知道路径是不是形状）。
        """
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("waypoints 至少需要 2 个点")
        if not (0.005 <= float(max_segment_m) <= 0.1):
            raise ValueError("max_segment_m outside range [0.005, 0.1]")
        canonical: list[dict] = []
        for w in waypoints:
            try:
                point = {
                    "x": round(float(w["x"]), 6),
                    "y": round(float(w["y"]), 6),
                    "z": round(float(w["z"]), 6),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"waypoint 必须是 {{x, y, z}} 数值点：{w!r}（{exc}）"
                ) from exc
            _workspace_check(point)
            canonical.append({**point, "contact": bool(w.get("contact", True))})
        # 线段插值；每个插值点继承其段目标的 contact 标记（抬笔段
        # 整段 contact=False——渲染/验收层可区分绘制与移动）。
        points: list[dict] = [dict(canonical[0])]
        for a, b in zip(canonical, canonical[1:], strict=False):
            seg = math.dist(
                (a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]),
            )
            steps = max(1, math.ceil(seg / float(max_segment_m)))
            for i in range(1, steps + 1):
                ratio = i / steps
                points.append(
                    {
                        "x": round(a["x"] + (b["x"] - a["x"]) * ratio, 6),
                        "y": round(a["y"] + (b["y"] - a["y"]) * ratio, 6),
                        "z": round(a["z"] + (b["z"] - a["z"]) * ratio, 6),
                        "contact": bool(b["contact"]),
                    }
                )
        digest = hashlib.sha256(
            json.dumps(points, sort_keys=True).encode()
        ).hexdigest()
        plan_id = f"plan_{digest[:16]}"
        spec = _build_pose_spec(
            [{"x": p["x"], "y": p["y"], "z": p["z"]} for p in points],
            plane="xy",
        )
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        (self._plans_dir / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "shape": "custom",
                    "waypoints": canonical,
                    "max_segment_m": max_segment_m,
                    "points": points,
                    "hash": digest,
                    "spec": spec,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        lifts = sum(1 for w in canonical if not w["contact"])
        return {
            "ok": True,
            "plan_id": plan_id,
            "hash": digest,
            "points": points,
            "point_count": len(points),
            "spec": spec,
            "summary": (
                f"自定义路径：{len(canonical)} 个路径点"
                + (f"（{lifts} 段抬笔移动）" if lifts else "")
                + f"，{len(points)} 个插值点"
            ),
        }

    def get_plan_payload(self, plan_id: str) -> dict[str, Any]:
        """PlanStore 记录的公开读取面（审计/测试——内容寻址可反查）。"""
        return self._load_plan(plan_id)

    def _load_plan(self, plan_id: str) -> dict:
        """WP-2：统一引用——kit envelope 记录（PersistentPlanStore）
        与 native 原始记录互通；不可解码给诚实错误码。"""
        path = self._plans_dir / f"{plan_id}.json"
        if not path.exists():
            raise ValueError(f"REF_NOT_FOUND: plan {plan_id!r} 不在共享 PlanStore")
        record = json.loads(path.read_text(encoding="utf-8"))
        # kit envelope：{plan_id, digest, trajectory, summary, status}
        if "trajectory" in record:
            record = record["trajectory"]
        if "points" not in record or "hash" not in record:
            raise ValueError(
                f"REF_FORMAT_UNKNOWN: plan {plan_id!r} 的记录格式不可解码"
                "（缺 points/hash）——生产者与消费者的引用格式不兼容"
            )
        return record

    @staticmethod
    def _spec_of(plan: dict) -> dict:
        """WP-5 前的旧 plan（无 spec）→ 从接触点合成默认 SE(3) 规格
        （工具轴垂直接触平面，approach/lift 与新版同构）。"""
        spec = plan.get("spec")
        if spec:
            return spec
        return _build_pose_spec(plan["points"], plane="xy")

    # --------------------------------------------------------------
    # 2. ur5e.simulate_cartesian_trajectory
    # --------------------------------------------------------------
    def generate_reach_path(
        self,
        target_m: list[float],
        *,
        approach_m: float = 0.05,
        normal_xyz: list[float] | None = None,
    ) -> dict[str, Any]:
        """arm reach（0824 §23 验收）：approach→contact 到目标点。"""
        if not isinstance(target_m, list | tuple) or len(target_m) != 3:
            raise ValueError("target_m must be [x, y, z]")
        normal = _normalize(normal_xyz or [0.0, 0.0, 1.0])
        target = {"x": float(target_m[0]), "y": float(target_m[1]), "z": float(target_m[2])}
        _workspace_check(target)
        approach = {
            "x": target["x"] + normal[0] * approach_m,
            "y": target["y"] + normal[1] * approach_m,
            "z": target["z"] + normal[2] * approach_m,
        }
        spec = _build_pose_spec_3d([approach, target], normal_xyz=normal)
        # reach：contact_plane 过 target（接触判定面向目标点，不是
        # approach 起点）；waypoints 只需 approach→contact。
        quat = spec["waypoints"][0]["orientation_xyzw"]
        spec["waypoints"] = [
            {
                "position_m": [approach["x"], approach["y"], approach["z"]],
                "orientation_xyzw": quat,
                "kind": "approach",
            },
            {
                "position_m": [target["x"], target["y"], target["z"]],
                "orientation_xyzw": quat,
                "kind": "contact",
            },
        ]
        spec["contact_plane"] = dict(spec["contact_plane"])
        spec["contact_plane"]["offset_m"] = _dot([target["x"], target["y"], target["z"]], normal)
        digest = hashlib.sha256(json.dumps(spec["waypoints"], sort_keys=True).encode()).hexdigest()
        spec["digest"] = f"sha256:{digest}"
        plan_id = f"plan_{digest[:16]}"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        (self._plans_dir / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "shape": "reach",
                    "target_m": list(target_m),
                    "approach_m": approach_m,
                    "points": [approach, target],
                    "hash": digest,
                    "spec": spec,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "plan_id": plan_id,
            "hash": digest,
            "points": [approach, target],
            "point_count": 2,
            "spec": spec,
            "summary": f"reach 路径：approach→{list(target_m)}",
        }

    def simulate_cartesian_trajectory(
        self,
        plan_id: str,
        *,
        controller: str = "jacobian",
        timestep_sec: float | None = None,
    ) -> dict[str, Any]:
        """真实 MuJoCo 动力学 rollout（SIM_DYN_ROLLOUT）。"""
        del controller, timestep_sec  # 当前后端固定参数——显式忽略
        import numpy as np

        plan = self._load_plan(plan_id)
        points = plan["points"]
        spec = self._spec_of(plan)
        waypoints = spec["waypoints"]
        from rosclaw.sandbox.backends import (
            MujocoCpuBackend,
            RolloutRequest,
            ScenarioSpec,
        )
        from rosclaw.sandbox.backends.fingerprints import file_hash
        from rosclaw.sandbox.sandbox_api import Sandbox

        trace_id = f"trace_{plan['hash'][:12]}"
        out_dir = self._traces_dir / trace_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # world=empty：无桌面障碍世界的动力学推演。桌面世界需要碰撞
        # 感知姿态规划（上臂连杆会扫过桌面——十四审实证 geom11↔tabletop），
        # 那是独立能力；当前为世界选择诚实标注。
        world = "empty"
        sandbox = Sandbox.create("ur5e", world, "mujoco")
        if not sandbox.has_physics:
            raise ValueError(f"MuJoCo 物理不可用: {sandbox.load_error}")
        try:
            model = sandbox.physics_model
            data = sandbox.physics_data
            import mujoco as _mj
            import numpy as _np

            site_id = -1
            for site_name in ("attachment_site", "tcp"):
                site_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, site_name)
                if site_id >= 0:
                    break
            if site_id < 0:
                raise ValueError("ur5e model has no end-effector site")
            # 安全转场：home（桌面上方）→ 抬升到 approach 高度 →
            # spec 航点（approach 降下 → contact → lift 抬升，WP-5
            # 显式建模在规格里，不再是这里的临时拼接）。
            data.qpos[: int(model.nu)] = _np.array(_HOME_QPOS[: int(model.nu)])
            _mj.mj_forward(model, data)
            home_eef = [float(v) for v in data.site_xpos[site_id]]
            approach_z = float(waypoints[0]["position_m"][2])
            approach_quat = list(waypoints[0]["orientation_xyzw"])
            transit = [
                {
                    "position_m": [home_eef[0], home_eef[1], approach_z],
                    "orientation_xyzw": approach_quat,
                    "kind": "transit",
                }
            ]
            joint_trajectory = _ik_waypoints_6d(model, data, transit + waypoints, site_id)
            resource_proof = sandbox.resource_manifest()
            scenario = ScenarioSpec(
                scenario_id=f"sim-trajectory-{plan['hash'][:8]}",
                robot_id="ur5e",
                world_id=world,
                body_snapshot_hash=f"sha256:{plan['hash']}",
                model_hash=file_hash(sandbox.model_path),
                seed=0,
                # N4.1：执行资源证明随 scenario/receipt 走。
                metadata={"resource": resource_proof},
            )
            backend = MujocoCpuBackend(sandbox)
            # 慢速插值（0.0005 rad/控制步）——每个 IK 航点约 10 步，
            # 物理臂有时间真实跟踪（默认 0.005 = 每航点一步，2ms 过点
            # 等于命令回放，臂永远追不上）。
            receipt = backend.rollout(
                RolloutRequest(
                    scenario=scenario,
                    trajectory=joint_trajectory,
                    max_joint_delta_rad=0.0003,
                    artifact_dir=out_dir,
                )
            )
            # 实际 eef 位姿：对 states 的 qpos 做 FK（不是命令回放——
            # 是物理推演后的真实位置+朝向）。
            import mujoco

            states_path = out_dir / "trajectory_states.json"
            states = json.loads(states_path.read_text(encoding="utf-8"))["states"]
            actual: list[dict] = []
            for sample in states:
                data.qpos[: int(model.nu)] = np.array(sample["qpos"])
                mujoco.mj_forward(model, data)
                pos = data.site_xpos[site_id]
                r_mat = np.array(data.site_xmat[site_id]).reshape(3, 3).tolist()
                quat = _mat_to_quat_xyzw(r_mat)
                actual.append(
                    {
                        "t": round(float(sample["time"]), 4),
                        "x": round(float(pos[0]), 6),
                        "y": round(float(pos[1]), 6),
                        "z": round(float(pos[2]), 6),
                        "quat_xyzw": [round(float(v), 6) for v in quat],
                    }
                )
            # 期望工具轴 = -接触平面法向（画图姿态）。
            normal = spec["contact_plane"]["normal_xyz"]
            desired_tool_z = [-float(n) for n in normal]
            metrics = self._tracking_metrics(
                points,
                actual,
                desired_tool_z=desired_tool_z,
                contact_plane=spec.get("contact_plane"),
            )
            # 落盘交付物：trace.json / trace.csv / metrics.json。
            (out_dir / "trace.json").write_text(
                json.dumps(
                    {
                        "schema_version": "rosclaw.sim_trace.v1",
                        # WP-3：states digest 锚点——渲染/验证据此拒绝被
                        # 篡改的 trajectory_states。
                        "states_digest": "sha256:"
                        + hashlib.sha256(states_path.read_bytes()).hexdigest(),
                        "plan_hash": plan["hash"],
                        # WP-5：SE(3) 规格锚点——朝向/接触平面可反查。
                        "spec_digest": spec["digest"],
                        "planned": points,
                        "actual": actual,
                        "evidence_level": "SIM_DYN_ROLLOUT",
                        # N4.1：资源证明进 trace（可反查 manifest/digest）。
                        "resource": resource_proof,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_dir / "trace.csv").write_text(
                "t,x,y,z\n"
                + "\n".join(f"{p['t']},{p['x']},{p['y']},{p['z']}" for p in actual)
                + "\n",
                encoding="utf-8",
            )
            (out_dir / "metrics.json").write_text(
                json.dumps(
                    {**metrics, "resource": resource_proof},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {
                "ok": True,
                "trace_id": trace_id,
                "evidence_level": "SIM_DYN_ROLLOUT",
                "physics_executed": receipt.physics_executed,
                "is_safe": receipt.is_safe,
                "violations": list(receipt.violations),
                "point_count": len(actual),
                "tracking": metrics,
                "resource": resource_proof,
                "artifacts": {
                    "trace_json": str(out_dir / "trace.json"),
                    "trace_csv": str(out_dir / "trace.csv"),
                    "metrics_json": str(out_dir / "metrics.json"),
                    "simulation_receipt": str(out_dir / "simulation_receipt.json"),
                },
            }
        finally:
            sandbox.close()

    @staticmethod
    def _tracking_metrics(
        planned: list[dict],
        actual: list[dict],
        *,
        desired_tool_z: list[float] | None = None,
        contact_plane: dict | None = None,
    ) -> dict:
        """实际 eef 到规划路径（最近点）的跟踪误差——只统计接触段
        （到接触平面 ≤ 2cm 且在路径邻域内；approach/lift 过渡是
        转场不是接触跟踪）。P1-C3：接触判定平面感知（任意法向
        n·p−offset ≤ tol——xy 是 normal=[0,0,1] 的特例）。WP-5：
        朝向误差同段统计。"""
        window = 0.05
        plane_tol = 0.02
        plane_z = float(planned[0]["z"]) if planned else 0.0
        plane_normal = (
            list(contact_plane.get("normal_xyz") or [0.0, 0.0, 1.0])
            if contact_plane
            else [0.0, 0.0, 1.0]
        )
        plane_offset = float(contact_plane.get("offset_m", plane_z)) if contact_plane else plane_z

        def _near(a: dict) -> float:
            return min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"])) for p in planned
            )

        def _on_plane(a: dict) -> bool:
            return abs(_dot([a["x"], a["y"], a["z"]], plane_normal) - plane_offset) <= plane_tol

        start = 0
        for idx, a in enumerate(actual):
            if _near(a) < window and _on_plane(a):
                start = idx
                break
        end = len(actual)
        for idx in range(len(actual) - 1, -1, -1):
            if _near(actual[idx]) < window and _on_plane(actual[idx]):
                end = idx + 1
                break
        actual = actual[start:end]
        errors = []
        orient_errors: list[float] = []
        for a in actual:
            best = min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"])) for p in planned
            )
            errors.append(best)
            if desired_tool_z is not None and "quat_xyzw" in a:
                tool_z = _tool_z_of_quat(a["quat_xyzw"])
                cos = max(-1.0, min(1.0, _dot(tool_z, desired_tool_z)))
                orient_errors.append(math.degrees(math.acos(cos)))
        # R0-5：分布/闭合/平面指标（低质量 PASS 防线——max/mean
        # 之外，RMSE/p95/闭合/平面偏差全部落账）。
        sorted_errors = sorted(errors)
        p95 = (
            sorted_errors[min(len(sorted_errors) - 1,
                              int(len(sorted_errors) * 0.95))]
            if sorted_errors else 0.0
        )
        rmse = (
            math.sqrt(sum(e * e for e in errors) / len(errors))
            if errors else 0.0
        )
        closure = 0.0
        if len(actual) >= 2:
            first, last = actual[0], actual[-1]
            closure = math.dist(
                (first["x"], first["y"], first["z"]),
                (last["x"], last["y"], last["z"]),
            )
        plane_deviations = [
            abs(_dot([a["x"], a["y"], a["z"]], plane_normal) - plane_offset)
            for a in actual
        ]
        metrics = {
            "max_error_m": round(max(errors, default=0.0), 6),
            "mean_error_m": round(sum(errors) / len(errors) if errors else 0.0, 6),
            "rmse_error_m": round(rmse, 6),
            "p95_error_m": round(p95, 6),
            "closure_error_m": round(closure, 6),
            "plane_max_deviation_m": round(
                max(plane_deviations, default=0.0), 6
            ),
            "contact_samples": len(actual),
            "samples": len(actual),
            "planned_points": len(planned),
        }
        if orient_errors:
            metrics["max_orientation_error_deg"] = round(max(orient_errors), 3)
            metrics["mean_orientation_error_deg"] = round(
                sum(orient_errors) / len(orient_errors), 3
            )
        return metrics

    # --------------------------------------------------------------
    # 3. simulation.render_trace
    # --------------------------------------------------------------
    def render_trace(self, trace_id: str, *, format: str = "gif", fps: int = 12) -> dict[str, Any]:
        if format != "gif":
            raise ValueError(f"unsupported format {format!r} (supported: gif)")
        out_dir = self._traces_dir / trace_id
        trace_path = out_dir / "trace.json"
        if not trace_path.exists():
            raise ValueError(f"unknown trace {trace_id!r}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actual = trace["actual"]
        image_mod, imagedraw_mod = self._import_pil()

        xs = [p["x"] for p in actual]
        ys = [p["y"] for p in actual]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        size = 480

        def _px(pt: dict) -> tuple[float, float]:
            return (
                20 + (size - 40) * (pt["x"] - min_x) / span,
                size - 20 - (size - 40) * (pt["y"] - min_y) / span,
            )

        frames = max(30, min(len(actual), 120))
        images = []
        for f in range(frames):
            upto = max(2, int(len(actual) * (f + 1) / frames))
            img = image_mod.new("RGB", (size, size), "white")
            draw = imagedraw_mod.Draw(img)
            pts = [_px(pt) for pt in actual[:upto]]
            draw.line(pts, fill=(20, 20, 20), width=2)
            ex, ey = pts[-1]
            draw.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), outline=(200, 30, 30), width=2)
            images.append(img)
        out = out_dir / f"{trace_id}.gif"
        # 原子写（0827 真实 K3 复验实证）：同内容修订重跑同 trace_id
        # 目录——原位写 GIF 的截断窗口被 verify_artifacts 抓成
        # "artifact 为空"瞬态 FAIL。先写临时文件再 os.replace。
        import os as _os

        tmp_out = out_dir / f".{trace_id}.gif.tmp.gif"
        images[0].save(
            tmp_out,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )
        _os.replace(tmp_out, out)
        # P0-F 闭包（金丝雀实证）：2D 预览与场景渲染同样产出
        # MP4——两条渲染路径都覆盖完整视频格式集（用户要 MP4 时
        # 选哪条都有交付物，同一 TraceRef）。
        import imageio.v3 as iio
        import numpy as _np

        mp4 = out_dir / f"{trace_id}.mp4"
        tmp_mp4 = out_dir / f".{trace_id}.mp4.tmp.mp4"
        iio.imwrite(tmp_mp4, [_np.asarray(img) for img in images], fps=max(fps, 1))
        _os.replace(tmp_mp4, mp4)
        return {
            "ok": True,
            "artifact": {
                "path": str(out),
                "frames": frames,
                "format": "gif",
                "bytes": out.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            },
            "mp4_artifact": {
                "path": str(mp4),
                "frames": frames,
                "format": "mp4",
                "bytes": mp4.stat().st_size,
                "evidence_level": "SIM_DYN_ROLLOUT",
            },
        }

    # --------------------------------------------------------------
    # 4. simulation.verify_tracking
    # --------------------------------------------------------------
    def verify_tracking(
        self,
        trace_id: str,
        *,
        max_tracking_error_m: float,
        max_orientation_error_deg: float | None = None,
    ) -> dict[str, Any]:
        out_dir = self._traces_dir / trace_id
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            raise ValueError(f"unknown trace {trace_id!r}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        verdict = "PASS" if metrics["max_error_m"] <= max_tracking_error_m else "FAIL"
        orientation: dict[str, Any] | None = None
        if max_orientation_error_deg is not None:
            if "max_orientation_error_deg" not in metrics:
                raise ValueError(
                    f"TRACE_ORIENTATION_MISSING: trace {trace_id!r} 无朝向"
                    "指标（WP-5 前的旧 trace）——无法按朝向阈值验收"
                )
            orientation = {
                "threshold_deg": max_orientation_error_deg,
                "max_error_deg": metrics["max_orientation_error_deg"],
                "mean_error_deg": metrics.get("mean_orientation_error_deg"),
            }
            if metrics["max_orientation_error_deg"] > max_orientation_error_deg:
                verdict = "FAIL"
        return {
            "ok": True,
            "verdict": verdict,
            "threshold_m": max_tracking_error_m,
            "metrics": metrics,
            "orientation": orientation,
            "evidence_level": "SIM_DYN_ROLLOUT",
        }
