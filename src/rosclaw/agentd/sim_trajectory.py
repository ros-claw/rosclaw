"""SIM 动力学闭环正式能力（十四审 PR-14.6，总纲 §5.3/§7 PR-14.6）。

四个注册能力（全部 COMPUTE 类、SIMULATION 限定、确定性、无副作用）：

1. trajectory.generate_planar_path——形状参数化平面轨迹（star5/circle
   同一组合，不为每个形状写新仿真器）；
2. ur5e.simulate_cartesian_trajectory——真实 MuJoCo 动力学 rollout
   （SIM_DYN_ROLLOUT 证据，不是命令回放）：DLS-IK 把笛卡尔路径转成
   关节轨迹，MujocoCpuBackend 物理推演，FK 还原实际 eef 轨迹；
3. simulation.render_trace——实际 eef 轨迹渲染 GIF（可打开的产物）；
4. simulation.verify_tracking——跟踪误差阈值判定（PASS/FAIL 诚实）。

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


def _ik_waypoints(model, data, points: list[dict]) -> list[list[float]]:
    """阻尼最小二乘 IK：笛卡尔路径 → 关节轨迹（从 home 顺序求解，
    每点以上一点为初值——轨迹连续不跳变）。"""
    import mujoco
    import numpy as np

    site_id = -1
    for site_name in ("attachment_site", "tcp"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            break
    if site_id < 0:
        raise ValueError("ur5e model has no end-effector site")
    nu = int(model.nu)
    q = np.array(_HOME_QPOS[:nu], dtype=float)
    waypoints: list[list[float]] = []
    jac = np.zeros((3, model.nv))
    lam = 0.05
    for point in points:
        target = np.array([point["x"], point["y"], point["z"]], dtype=float)
        for _ in range(120):
            data.qpos[:nu] = q
            mujoco.mj_forward(model, data)
            err = target - np.array(data.site_xpos[site_id])
            if float(np.linalg.norm(err)) < 1e-3:
                break
            mujoco.mj_jacSite(model, data, jac, None, site_id)
            jac_pos = jac[:, :nu]
            dq = jac_pos.T @ np.linalg.solve(
                jac_pos @ jac_pos.T + lam * lam * np.eye(3), err
            )
            dq = np.clip(dq, -0.10, 0.10)
            q = q + dq
            q = np.clip(
                q,
                model.actuator_ctrlrange[:nu, 0],
                model.actuator_ctrlrange[:nu, 1],
            )
        else:
            raise ValueError(
                f"IK 未收敛于 ({point['x']:.3f},{point['y']:.3f},{point['z']:.3f})"
                "——路径不可达（编译期失败，零执行）"
            )
        waypoints.append(q.copy().tolist())
    return waypoints


class SimTrajectoryService:
    """SIM 轨迹能力的确定性实现（文件落盘，重启可审计）。"""

    def __init__(self, home: Path) -> None:
        self._home = Path(home)
        self._plans_dir = self._home / "sim" / "plans"
        self._traces_dir = self._home / "sim" / "traces"

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
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "plan_id": plan_id,
            "hash": digest,
            "points": points,
            "point_count": len(points),
            "summary": (
                f"{shape}：中心 ({center_m[0]}, {center_m[1]}, {center_m[2]})m，"
                f"半径 {scale_m}m，{len(points)} 个插值点，已闭合"
            ),
        }

    def _load_plan(self, plan_id: str) -> dict:
        path = self._plans_dir / f"{plan_id}.json"
        if not path.exists():
            raise ValueError(f"unknown plan {plan_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

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
            # 安全转场：home（桌面上方）→ 抬升 → 路径起点上方 → 下降
            # 到起点——否则 home→起点的开环插值会拖着连杆扫过桌面
            # （COLLISION 实证）。
            import mujoco as _mj
            import numpy as _np

            data.qpos[: int(model.nu)] = _np.array(_HOME_QPOS[: int(model.nu)])
            _mj.mj_forward(model, data)
            _site = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, "attachment_site")
            home_eef = [float(v) for v in data.site_xpos[_site]]
            start = points[0]
            lift_z = max(0.50, float(start["z"]) + 0.20)
            transit = [
                {"x": home_eef[0], "y": home_eef[1], "z": lift_z},
                {"x": start["x"], "y": start["y"], "z": lift_z},
            ]
            joint_trajectory = _ik_waypoints(model, data, transit + points)
            scenario = ScenarioSpec(
                scenario_id=f"sim-trajectory-{plan['hash'][:8]}",
                robot_id="ur5e",
                world_id=world,
                body_snapshot_hash=f"sha256:{plan['hash']}",
                model_hash=file_hash(sandbox.model_path),
                seed=0,
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
            # 实际 eef 轨迹：对 states 的 qpos 做 FK（不是命令回放——
            # 是物理推演后的真实位置）。
            import mujoco

            states_path = out_dir / "trajectory_states.json"
            states = json.loads(states_path.read_text(encoding="utf-8"))["states"]
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )
            actual: list[dict] = []
            for sample in states:
                data.qpos[: int(model.nu)] = np.array(sample["qpos"])
                mujoco.mj_forward(model, data)
                pos = data.site_xpos[site_id]
                actual.append({
                    "t": round(float(sample["time"]), 4),
                    "x": round(float(pos[0]), 6),
                    "y": round(float(pos[1]), 6),
                    "z": round(float(pos[2]), 6),
                })
            metrics = self._tracking_metrics(points, actual)
            # 落盘交付物：trace.json / trace.csv / metrics.json。
            (out_dir / "trace.json").write_text(
                json.dumps({
                    "schema_version": "rosclaw.sim_trace.v1",
                    "plan_hash": plan["hash"],
                    "planned": points,
                    "actual": actual,
                    "evidence_level": "SIM_DYN_ROLLOUT",
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
                json.dumps(metrics, ensure_ascii=False, indent=2),
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
    def _tracking_metrics(planned: list[dict], actual: list[dict]) -> dict:
        """实际 eef 到规划路径（最近点）的跟踪误差——剔除 home →
        路径起点的转场段（从首次进入路径邻域起算）。"""
        start = 0
        for idx, a in enumerate(actual):
            near = min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"]))
                for p in planned
            )
            if near < 0.05:
                start = idx
                break
        actual = actual[start:]
        errors = []
        for a in actual:
            best = min(
                math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"]))
                for p in planned
            )
            errors.append(best)
        return {
            "max_error_m": round(max(errors, default=0.0), 6),
            "mean_error_m": round(
                sum(errors) / len(errors) if errors else 0.0, 6
            ),
            "samples": len(actual),
            "planned_points": len(planned),
        }

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
        from PIL import Image, ImageDraw

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
            img = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(img)
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
        self, trace_id: str, *, max_tracking_error_m: float
    ) -> dict[str, Any]:
        out_dir = self._traces_dir / trace_id
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            raise ValueError(f"unknown trace {trace_id!r}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        verdict = (
            "PASS" if metrics["max_error_m"] <= max_tracking_error_m else "FAIL"
        )
        return {
            "ok": True,
            "verdict": verdict,
            "threshold_m": max_tracking_error_m,
            "metrics": metrics,
            "evidence_level": "SIM_DYN_ROLLOUT",
        }
