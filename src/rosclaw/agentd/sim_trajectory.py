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
    return _mat_to_quat_xyzw([
        [x_d[0], y_d[0], z_d[0]],
        [x_d[1], y_d[1], z_d[1]],
        [x_d[2], y_d[2], z_d[2]],
    ])


def _tool_z_of_quat(q: list[float]) -> list[float]:
    """四元数（xyzw）表达的工具 z 轴在世界系的方向。"""
    x, y, z, w = q
    return [
        2 * (x * z + y * w),
        2 * (y * z - x * w),
        1 - 2 * (x * x + y * y),
    ]


def _build_pose_spec(points: list[dict], *, plane: str) -> dict:
    """接触点列 → PoseTrajectorySpecV1（approach/contact/lift 显式段）。

    当前只支持 xy 水平接触平面（法向 +z）——规格本身支持任意
    平面，生成器对非 xy 诚实拒绝（见 generate_planar_path）。
    """
    if plane != "xy":
        raise ValueError(f"unsupported plane {plane!r} (supported: xy)")
    normal = [0.0, 0.0, 1.0]
    quat = _tool_down_orientation(normal)
    offset = float(points[0]["z"])
    start, end = points[0], points[-1]
    waypoints: list[dict] = [
        # approach：接触点上方 → 接触点（降下到位）。
        {
            "position_m": [start["x"], start["y"], start["z"] + _APPROACH_LIFT_M],
            "orientation_xyzw": quat,
            "kind": "approach",
        },
        {
            "position_m": [start["x"], start["y"], start["z"]],
            "orientation_xyzw": quat,
            "kind": "approach",
        },
    ]
    waypoints += [
        {
            "position_m": [p["x"], p["y"], p["z"]],
            "orientation_xyzw": quat,
            "kind": "contact",
        }
        for p in points
    ]
    # lift：末点抬升回 approach 高度。
    waypoints.append({
        "position_m": [end["x"], end["y"], end["z"] + _APPROACH_LIFT_M],
        "orientation_xyzw": quat,
        "kind": "lift",
    })
    digest = hashlib.sha256(
        json.dumps(waypoints, sort_keys=True).encode()
    ).hexdigest()
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
            vertices.append(
                {"x": cx + r * math.cos(angle), "y": cy + r * math.sin(angle), "z": cz}
            )
    elif shape == "circle":
        # 解析圆弧采样（弦插值会落入圆内——圆走精确弧，不走折线）。
        step = max(1, math.ceil(2 * math.pi * scale / 0.02))
        for k in range(step):
            angle = 2 * math.pi * k / step
            vertices.append(
                {"x": cx + scale * math.cos(angle),
                 "y": cy + scale * math.sin(angle), "z": cz}
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
            points.append({
                "x": a["x"] + (b["x"] - a["x"]) * ratio,
                "y": a["y"] + (b["y"] - a["y"]) * ratio,
                "z": a["z"] + (b["z"] - a["z"]) * ratio,
            })
    return points


def _ik_waypoints_6d(
    model, data, poses: list[dict], site_id: int
) -> list[list[float]]:
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
            if float(np.linalg.norm(err_p)) < 1e-3 and float(
                np.linalg.norm(err_r)
            ) < 0.02:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, :nu], jacr[:, :nu]])
            err = np.concatenate([err_p, err_r])
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + lam * lam * np.eye(6), err
            )
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
        """PIL 导入：宿主环境优先；缺失 → 托管 rosclaw-simulation
        runtime ensure+activate 后重试；托管也不可用 → RuntimeNotReadyError
        （诚实——调用方映射 BLOCKED，不是裸 ModuleNotFoundError）。"""
        try:
            from PIL import Image, ImageDraw

            return Image, ImageDraw
        except ImportError:
            if self._runtime_manager is None:
                raise
            handle = self._runtime_manager.ensure("rosclaw-simulation")
            self._runtime_manager.activate(handle)
            from PIL import Image, ImageDraw

            return Image, ImageDraw

    # --------------------------------------------------------------
    # 1. trajectory.generate_planar_path
    # --------------------------------------------------------------
    def generate_planar_path(
        self,
        *,
        shape: str,
        center_m: list[float],
        scale_m: float,
        plane: str = "xy",
        max_segment_m: float = 0.02,
    ) -> dict[str, Any]:
        if shape not in _SHAPES:
            raise ValueError(
                f"unsupported shape {shape!r} (supported: {', '.join(_SHAPES)})"
            )
        if plane != "xy":
            raise ValueError(f"unsupported plane {plane!r} (supported: xy)")
        if not isinstance(center_m, list | tuple) or len(center_m) != 3:
            raise ValueError("center_m must be [x, y, z]")
        if not (0.02 <= float(scale_m) <= 0.35):
            raise ValueError(f"scale_m {scale_m} outside range [0.02, 0.35]")
        if not (0.005 <= float(max_segment_m) <= 0.1):
            raise ValueError("max_segment_m outside range [0.005, 0.1]")
        vertices = _shape_vertices(shape, list(center_m), float(scale_m))
        # 圆已是精确弧采样；其余走线段插值。
        points = (
            vertices
            if shape == "circle"
            else _sample_segments(vertices, float(max_segment_m))
        )
        for point in points:
            _workspace_check(point)
        # WP-5：SE(3) 位姿规格——approach/contact/lift 显式段 +
        # 工具朝向 + 接触平面，随 plan 持久化（内容寻址可反查）。
        spec = _build_pose_spec(points, plane=plane)
        digest = hashlib.sha256(
            json.dumps(points, sort_keys=True).encode()
        ).hexdigest()
        plan_id = f"plan_{digest[:16]}"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        (self._plans_dir / f"{plan_id}.json").write_text(
            json.dumps({
                "shape": shape, "center_m": list(center_m), "scale_m": scale_m,
                "plane": plane, "max_segment_m": max_segment_m,
                "points": points, "hash": digest,
                "spec": spec,
            }, ensure_ascii=False),
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
                site_id = _mj.mj_name2id(
                    model, _mj.mjtObj.mjOBJ_SITE, site_name
                )
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
            transit = [{
                "position_m": [home_eef[0], home_eef[1], approach_z],
                "orientation_xyzw": approach_quat,
                "kind": "transit",
            }]
            joint_trajectory = _ik_waypoints_6d(
                model, data, transit + waypoints, site_id
            )
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
                    max_joint_delta_rad=0.0005,
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
                r_mat = (
                    np.array(data.site_xmat[site_id]).reshape(3, 3).tolist()
                )
                quat = _mat_to_quat_xyzw(r_mat)
                actual.append({
                    "t": round(float(sample["time"]), 4),
                    "x": round(float(pos[0]), 6),
                    "y": round(float(pos[1]), 6),
                    "z": round(float(pos[2]), 6),
                    "quat_xyzw": [round(float(v), 6) for v in quat],
                })
            # 期望工具轴 = -接触平面法向（画图姿态）。
            normal = spec["contact_plane"]["normal_xyz"]
            desired_tool_z = [-float(n) for n in normal]
            metrics = self._tracking_metrics(
                points, actual, desired_tool_z=desired_tool_z
            )
            # 落盘交付物：trace.json / trace.csv / metrics.json。
            (out_dir / "trace.json").write_text(
                json.dumps({
                    "schema_version": "rosclaw.sim_trace.v1",
                    # WP-3：states digest 锚点——渲染/验证据此拒绝被
                    # 篡改的 trajectory_states。
                    "states_digest": "sha256:" + hashlib.sha256(
                        states_path.read_bytes()
                    ).hexdigest(),
                    "plan_hash": plan["hash"],
                    # WP-5：SE(3) 规格锚点——朝向/接触平面可反查。
                    "spec_digest": spec["digest"],
                    "planned": points,
                    "actual": actual,
                    "evidence_level": "SIM_DYN_ROLLOUT",
                    # N4.1：资源证明进 trace（可反查 manifest/digest）。
                    "resource": resource_proof,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "trace.csv").write_text(
                "t,x,y,z\n"
                + "\n".join(
                    f"{p['t']},{p['x']},{p['y']},{p['z']}" for p in actual
                )
                + "\n",
                encoding="utf-8",
            )
            (out_dir / "metrics.json").write_text(
                json.dumps(
                    {**metrics, "resource": resource_proof},
                    ensure_ascii=False, indent=2,
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
    ) -> dict:
        """实际 eef 到规划路径（最近点）的跟踪误差——剔除转场段
        （从首次进入路径邻域起算，到最后一次离开邻域为止；WP-5 的
        lift 抬升段刻意离开接触路径，不计入接触跟踪误差）。WP-5：
        朝向误差（实际工具轴 vs 期望工具轴，度）同段统计。"""
        window = 0.05

        def _near(a: dict) -> float:
            return min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"]))
                for p in planned
            )

        start = 0
        for idx, a in enumerate(actual):
            if _near(a) < window:
                start = idx
                break
        end = len(actual)
        for idx in range(len(actual) - 1, -1, -1):
            if _near(actual[idx]) < window:
                end = idx + 1
                break
        actual = actual[start:end]
        errors = []
        orient_errors: list[float] = []
        for a in actual:
            best = min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"]))
                for p in planned
            )
            errors.append(best)
            if desired_tool_z is not None and "quat_xyzw" in a:
                tool_z = _tool_z_of_quat(a["quat_xyzw"])
                cos = max(-1.0, min(1.0, _dot(tool_z, desired_tool_z)))
                orient_errors.append(math.degrees(math.acos(cos)))
        metrics = {
            "max_error_m": round(max(errors, default=0.0), 6),
            "mean_error_m": round(
                sum(errors) / len(errors) if errors else 0.0, 6
            ),
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
    def render_trace(self, trace_id: str, *, format: str = "gif",
                     fps: int = 12) -> dict[str, Any]:
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
        images[0].save(
            out, save_all=True, append_images=images[1:],
            duration=int(1000 / max(fps, 1)), loop=0,
        )
        return {
            "ok": True,
            "artifact": {
                "path": str(out),
                "frames": frames,
                "format": "gif",
                "bytes": out.stat().st_size,
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
        verdict = (
            "PASS" if metrics["max_error_m"] <= max_tracking_error_m else "FAIL"
        )
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
