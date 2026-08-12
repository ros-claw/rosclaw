"""UR5e 确定性 SIM MCP server（六审 §6.3/PR-SIX-3）。

第一方仿真本体：为 `sim/ur5e` body 提供真实的机械臂能力面——
从此"运行机械臂仿真"有可执行回答，而不是模型编造的动作名。

工具（全部是确定性内存状态机，无外部依赖）：

- ``ur5e.get_joint_state``（OBSERVE）——6 关节角（rad）；
- ``ur5e.get_end_effector_pose``（OBSERVE）——末端位姿 x/y/z/rpy；
- ``ur5e.move_joints``（PHYSICAL_ACTION）——关节空间运动（限位校验）；
- ``ur5e.move_to_pose``（PHYSICAL_ACTION）——末端位姿运动（安全工作
  空间校验：半径/高度边界内）；
- ``ur5e.stop``（PHYSICAL_ACTION）——停止并保持当前状态。

模型永远不能经 ToolCatalog 调用动作工具；agentd 的 SimActionChannel
（SIM 物理权威）在 EXACT_ACTION grant 验证后才直接调用它们。
"""

from __future__ import annotations

import json
import math
import time

from mcp.server.fastmcp import FastMCP

# 十审 W0：WARNING——FastMCP 默认 INFO 会把 "Processing request of type
# ListToolsRequest" 打到 stderr（经 errlog 泄漏进 TUI 第一屏）。内部诊断
# 只在 --debug/doctor/log 出现。
server = FastMCP("ur5e-sim", log_level="WARNING")

# UR5e 近似关节限位（rad）——教学级 SIM 边界，不是真实 DH 模型。
_JOINT_LIMITS = (-2 * math.pi, 2 * math.pi)
# 安全工作空间（米）：以基座为原点的圆柱壳 + 高度窗口。
_SAFE_RADIUS = (0.10, 0.80)
_SAFE_Z = (0.02, 1.20)

# 确定性初始状态（零位附近的一个稳定姿态）。
_state = {
    "joints": [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0],
    "pose": {"x": 0.30, "y": 0.20, "z": 0.40, "roll": 0.0, "pitch": 3.1416, "yaw": 0.0},
    "moving": False,
    "last_motion": None,
    # 七审 PR-SEVEN-4：轨迹级状态——时间序列 trace + 计划缓存。
    "trace": [],
    "plans": {},
}

# 七审 §6 PR-SEVEN-4.9 + 九审 §21.6：本产品面是确定性命令回放
# 沙盒（规划点写入 state 再自证）——不暗示运动学/动力学仿真。
SIM_KIND = "command-replay"


class PlanStore:
    """执行器侧计划仓库（八审 §1.4/P0-3）：不透明 plan_id。

    完整轨迹只存在于受信执行器进程内——模型只见 plan_id + digest +
    摘要。单次消费、TTL、容量限制；载荷复验（canonical hash）+
    工作空间校验在执行时重做一次。任何未知/过期/已消费/篡改一律
    fail closed。
    """

    def __init__(self, *, ttl_s: float = 1800.0, capacity: int = 32) -> None:
        self._ttl_s = ttl_s
        self._capacity = capacity
        self._records: dict[str, dict] = {}

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def put(self, trajectory: dict, summary: str) -> dict:
        # 九审 §17.3：plan_instance_id 随机——digest 只做内容寻址；
        # 同一任务可合法重复执行（每个实例单次消费）。
        import uuid as _uuid

        digest = str(trajectory["hash"])
        plan_id = f"plan_{_uuid.uuid4().hex[:16]}"
        if plan_id not in self._records:
            if len(self._records) >= self._capacity:
                oldest = min(
                    self._records, key=lambda k: self._records[k]["created_mono"]
                )
                del self._records[oldest]
            self._records[plan_id] = {
                "plan_id": plan_id,
                "digest": digest,
                "trajectory": trajectory,
                "summary": summary,
                "created_mono": self._now(),
                "status": "PLANNED",
            }
        return self._records[plan_id]

    def get_for_execute(self, plan_id: str) -> dict:
        record = self._records.get(plan_id)
        if record is None:
            raise ValueError(f"unknown plan_id {plan_id!r} (fail closed)")
        if record["status"] != "PLANNED":
            raise ValueError(
                f"plan {plan_id} already consumed — single-use (fail closed)"
            )
        if self._now() - record["created_mono"] > self._ttl_s:
            raise ValueError(f"plan {plan_id} expired (fail closed)")
        return record

    def consume(self, plan_id: str) -> None:
        self._records[plan_id]["status"] = "CONSUMED"

    def by_digest(self, digest: str) -> dict | None:
        for record in self._records.values():
            if record["digest"] == digest:
                return record
        return None

    def clear(self) -> None:
        self._records.clear()


def _make_plan_store():
    """WP-P0-7：ROSCLAW_HOME 可用 → 落盘 PlanStore（executor 重启
    不丢 plan、已消费不复活）；否则内存（单测直 import）。"""
    import os as _os
    from pathlib import Path as _Path

    home = _os.environ.get("ROSCLAW_HOME")
    if home:
        from rosclaw.sim.plan_store import PersistentPlanStore

        return PersistentPlanStore(_Path(home) / "sim" / "plans")
    return PlanStore()


_PLAN_STORE = _make_plan_store()


def _canonical_point(point: dict) -> dict:
    return {
        "x": round(float(point["x"]), 6),
        "y": round(float(point["y"]), 6),
        "z": round(float(point["z"]), 6),
    }


def _trajectory_hash(points: list[dict]) -> str:
    import hashlib

    canonical = json.dumps([_canonical_point(p) for p in points], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _workspace_check(point: dict) -> None:
    radius = math.hypot(point["x"], point["y"])
    r_lo, r_hi = _SAFE_RADIUS
    z_lo, z_hi = _SAFE_Z
    if not (r_lo <= radius <= r_hi):
        raise ValueError(
            f"point ({point['x']:.3f},{point['y']:.3f}) radius {radius:.3f}m "
            f"outside safe workspace [{r_lo}, {r_hi}]"
        )
    if not (z_lo <= point["z"] <= z_hi):
        raise ValueError(f"point z={point['z']}m outside safe window [{z_lo}, {z_hi}]")


@server.tool(
    name="ur5e.plan_cartesian_path",
    description="规划笛卡尔轨迹（COMPUTE，无副作用）：五角星等平面图形"
    "生成顶点+插值+canonical hash；全部点经安全工作空间校验。",
    annotations={"readOnlyHint": True},
)
def plan_cartesian_path(
    shape: str,
    center_x: float,
    center_y: float,
    z: float,
    outer_radius: float,
    max_segment_m: float = 0.02,
    include_payload: bool = False,
) -> str:
    if shape != "star5":
        raise ValueError(f"unsupported shape {shape!r} (supported: star5)")
    if not (0.02 <= outer_radius <= 0.35):
        raise ValueError("outer_radius out of range [0.02, 0.35]")
    if not (0.005 <= max_segment_m <= 0.1):
        raise ValueError("max_segment_m out of range [0.005, 0.1]")
    # 五角星轮廓：5 外顶点 + 5 内顶点每 36° 交替，回到起点。
    inner_radius = outer_radius * 0.381966
    waypoints: list[dict] = []
    for k in range(10):
        angle = math.radians(90 + k * 36)
        radius = outer_radius if k % 2 == 0 else inner_radius
        waypoints.append(
            _canonical_point(
                {
                    "x": center_x + radius * math.cos(angle),
                    "y": center_y + radius * math.sin(angle),
                    "z": z,
                }
            )
        )
    waypoints.append(dict(waypoints[0]))  # 闭合
    # 全部点经安全工作空间校验（规划即拒越界）。
    for point in waypoints:
        _workspace_check(point)
    # 按最大线段插值。
    points: list[dict] = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:], strict=False):
        seg = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
        steps = max(1, math.ceil(seg / max_segment_m))
        for i in range(1, steps + 1):
            ratio = i / steps
            points.append(
                _canonical_point(
                    {
                        "x": a["x"] + (b["x"] - a["x"]) * ratio,
                        "y": a["y"] + (b["y"] - a["y"]) * ratio,
                        "z": a["z"] + (b["z"] - a["z"]) * ratio,
                    }
                )
            )
    trajectory = {
        "shape": shape,
        "waypoints": waypoints,
        "points": points,
        "max_segment_m": max_segment_m,
        "hash": _trajectory_hash(points),
    }
    # 八审 §1.4/P0-3：完整载荷只进 PlanStore——模型收到不透明句柄
    # （plan_id + digest + 摘要），不再搬运 points/hash。
    summary = (
        f"{shape} 五角星：中心 ({center_x}, {center_y}, {z})m，"
        f"外半径 {outer_radius}m，{len(points)} 个插值点，已闭合"
    )
    record = _PLAN_STORE.put(trajectory, summary)
    result = {
        "ok": True,
        "sim_kind": SIM_KIND,
        "plan_id": record["plan_id"],
        "digest": record["digest"],
        "summary": summary,
        "point_count": len(points),
        "workspace_ok": True,
    }
    if include_payload:  # dev/几何单测后门——绝不默认开
        result["trajectory"] = trajectory
    return json.dumps(result, ensure_ascii=False)


@server.tool(
    name="ur5e.execute_plan",
    description="执行已规划的轨迹（物理动作；SIM 下为仿真执行）——"
    "只接受 plan_id：载荷从 PlanStore 取回并复验 canonical hash + "
    "工作空间；单次消费。模型不得也不需传轨迹/hash。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def execute_plan(plan_id: str) -> str:
    record = _PLAN_STORE.get_for_execute(plan_id)
    trajectory = record["trajectory"]
    points = trajectory["points"]
    # 执行前复验（TOCTOU）：canonical hash + 工作空间。
    actual = _trajectory_hash(points)
    if actual != record["digest"]:
        raise ValueError(f"plan {plan_id} payload hash mismatch (fail closed)")
    for point in points:
        _workspace_check(point)
    _PLAN_STORE.consume(plan_id)
    result = json.loads(_execute_trajectory(trajectory))
    result["plan_id"] = plan_id
    result["digest"] = record["digest"]
    return json.dumps(result, ensure_ascii=False)


@server.tool(
    name="ur5e.execute_cartesian_path",
    description="[deprecated——dev 兼容层] 直接执行完整轨迹对象；"
    "正式路径是 plan_cartesian_path → execute_plan(plan_id)。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def execute_cartesian_path(trajectory: dict) -> str:
    return _execute_trajectory(trajectory)


def _execute_trajectory(trajectory: dict) -> str:
    points = trajectory.get("points") if isinstance(trajectory, dict) else None
    if not points or not isinstance(points, list):
        raise ValueError("trajectory.points required")
    claimed = str(trajectory.get("hash", ""))
    actual = _trajectory_hash(points)
    if not claimed or claimed != actual:
        raise ValueError(
            f"trajectory hash mismatch: claimed {claimed[:16]} != actual {actual[:16]} "
            "— refuse to execute a tampered trajectory (fail closed)"
        )
    for point in points:
        _workspace_check(point)
    # 确定性执行：时间序列 trace（dt=50ms 每插值点）。
    trace = []
    for i, point in enumerate(points):
        trace.append({"t": round(i * 0.05, 3), **_canonical_point(point)})
    _state["trace"] = trace
    last = points[-1]
    _state["pose"] = {
        "x": last["x"], "y": last["y"], "z": last["z"],
        "roll": 0.0, "pitch": 3.1416, "yaw": 0.0,
    }
    _state["last_motion"] = {
        "kind": "execute_cartesian_path",
        "trajectory_hash": actual,
        "points": len(points),
        "executed_at": _ts(),
    }
    return json.dumps(
        {
            "ok": True,
            "driver": "completed",
            "evidence_domain": "simulation",
            "sim_kind": SIM_KIND,
            # WP-P0-6（总纲 §8.3）：本执行器把规划点写入 sandbox
            # state——证据等级是 COMMAND_REPLAY（路径自洽），不是
            # 运动学/动力学仿真，更不是真实机械臂观测。
            "evidence_level": "COMMAND_REPLAY",
            "limitation": (
                "trace 与计划来自同一 sandbox——能证明路径数据自洽，"
                "不能证明 MuJoCo 动力学或真实机械臂完成运动"
            ),
            "trajectory_hash": actual,
            "points_executed": len(points),
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


def _trace_svg(trace: list[dict]) -> str:
    """trace → 简易 SVG（x/y 投影；证据用，不是渲染品）。"""
    if not trace:
        return ""
    xs = [p["x"] for p in trace]
    ys = [p["y"] for p in trace]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1e-6)

    def _px(p: dict) -> tuple[float, float]:
        return (
            10 + 280 * (p["x"] - min_x) / span,
            290 - 280 * (p["y"] - min_y) / span,
        )

    points_attr = " ".join(f"{_px(p)[0]:.1f},{_px(p)[1]:.1f}" for p in trace)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">'
        f'<polyline points="{points_attr}" fill="none" stroke="black" stroke-width="1.5"/>'
        "</svg>"
    )


@server.tool(
    name="ur5e.get_cartesian_trace",
    description="读取最近执行的笛卡尔轨迹 trace（时间序列 + hash + SVG，"
    "SIM 观测）。",
    annotations={"readOnlyHint": True},
)
def get_cartesian_trace(include_points: bool = False) -> str:
    trace = list(_state["trace"])
    # 八审 P0-3/P0-7：模型视图是摘要（点数/hash/端点/边界/SVG）——
    # 完整时间序列是证据，默认不进模型上下文（include_points 为
    # dev/几何单测后门）。
    trace_view: dict = {
        "trajectory_hash": _trajectory_hash(trace) if trace else "",
        "point_count": len(trace),
        "first_point": trace[0] if trace else None,
        "last_point": trace[-1] if trace else None,
        "svg": _trace_svg(trace),
        # WP-P0-6：trace 来源标注——command replay 不是独立观测。
        "origin": "command_replay",
    }
    if include_points:
        trace_view["points"] = trace
    return json.dumps(
        {
            "ok": True,
            "evidence_domain": "simulation",
            "evidence_level": "COMMAND_REPLAY",
            "sim_kind": SIM_KIND,
            "trace": trace_view,
            "observed_at": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="ur5e.verify_drawing",
    description="后验几何验证（COMPUTE）：trace 对目标轨迹的端点误差/"
    "RMSE/最大误差/闭合误差——全部过阈值才 PASS。默认验证最近执行的"
    " plan（无需模型传 hash）。",
    annotations={"readOnlyHint": True},
)
def verify_drawing(expected_trajectory_hash: str = "") -> str:
    # 八审 P0-3：hash 不该由模型搬运——默认取最近执行 plan 的 digest。
    if not expected_trajectory_hash:
        last = _state.get("last_motion") or {}
        expected_trajectory_hash = str(last.get("trajectory_hash", ""))
        if not expected_trajectory_hash:
            raise ValueError("no executed plan to verify (fail closed)")
    record = _PLAN_STORE.by_digest(expected_trajectory_hash)
    if record is None:
        raise ValueError(f"unknown trajectory hash {expected_trajectory_hash[:16]}")
    expected = record["trajectory"]
    trace = list(_state["trace"])
    expected_points = expected["points"]
    if len(trace) != len(expected_points):
        verdict = {
            "verdict": "FAIL",
            "reason": f"trace length {len(trace)} != expected {len(expected_points)}",
        }
        return json.dumps({"ok": True, "verification": verdict})
    errors = [
        math.dist(
            (t["x"], t["y"], t["z"]), (e["x"], e["y"], e["z"])
        )
        for t, e in zip(trace, expected_points, strict=True)
    ]
    endpoint_error = errors[-1] if errors else 0.0
    rmse = math.sqrt(sum(e * e for e in errors) / max(1, len(errors)))
    max_error = max(errors, default=0.0)
    closure_error = (
        math.dist(
            (trace[0]["x"], trace[0]["y"], trace[0]["z"]),
            (trace[-1]["x"], trace[-1]["y"], trace[-1]["z"]),
        )
        if trace
        else 0.0
    )
    threshold = 0.005  # 5mm
    passed = (
        endpoint_error < threshold
        and rmse < threshold
        and max_error < threshold
        and closure_error < threshold
    )
    verdict = {
        "verdict": "PASS" if passed else "FAIL",
        "endpoint_error_m": endpoint_error,
        "rmse_m": rmse,
        "max_error_m": max_error,
        "closure_error_m": closure_error,
        "threshold_m": threshold,
        "trajectory_hash": expected_trajectory_hash,
    }
    return json.dumps({"ok": True, "verification": verdict})


@server.tool(
    name="ur5e.reset_simulation",
    description="重置仿真状态（CONTROL——按策略经授权链或维护通道）。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def reset_simulation() -> str:
    _state["trace"] = []
    _state["plans"] = {}
    _PLAN_STORE.clear()
    _state["moving"] = False
    return json.dumps(
        {
            "ok": True,
            "driver": "completed",
            "evidence_domain": "simulation",
            "reset": True,
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@server.tool(
    name="ur5e.get_joint_state",
    description="读取 UR5e 当前 6 关节角（rad，SIM 观测）。",
    annotations={"readOnlyHint": True},
)
def get_joint_state() -> str:
    return json.dumps(
        {
            "ok": True,
            "evidence_domain": "simulation",
            "joints": list(_state["joints"]),
            "moving": _state["moving"],
            "observed_at": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="ur5e.get_end_effector_pose",
    description="读取 UR5e 末端执行器位姿（x/y/z 米 + roll/pitch/yaw rad，SIM 观测）。",
    annotations={"readOnlyHint": True},
)
def get_end_effector_pose() -> str:
    return json.dumps(
        {
            "ok": True,
            "evidence_domain": "simulation",
            "pose": dict(_state["pose"]),
            "observed_at": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="ur5e.move_joints",
    description="关节空间运动到目标关节角（物理动作；SIM 下为仿真执行）。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def move_joints(
    joints: list[float],
    duration_sec: float = 2.0,
    velocity_fraction: float = 0.1,
) -> str:
    if len(joints) != 6:
        raise ValueError("joints must have exactly 6 values")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in joints):
        raise ValueError("joints must be finite numbers")
    lo, hi = _JOINT_LIMITS
    for v in joints:
        if not (lo <= v <= hi):
            raise ValueError(f"joint value {v} out of limits [{lo}, {hi}]")
    if not (0.1 <= duration_sec <= 30.0):
        raise ValueError("duration_sec out of range [0.1, 30]")
    if not (0.01 <= velocity_fraction <= 1.0):
        raise ValueError("velocity_fraction out of range [0.01, 1.0]")
    _state["joints"] = [float(v) for v in joints]
    _state["moving"] = False
    _state["last_motion"] = {
        "kind": "move_joints",
        "target": list(_state["joints"]),
        "duration_sec": duration_sec,
        "executed_at": _ts(),
    }
    return json.dumps(
        {
            "ok": True,
            "driver": "completed",
            "evidence_domain": "simulation",
            "joints": list(_state["joints"]),
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="ur5e.move_to_pose",
    description="末端执行器移动到目标位姿（物理动作；SIM 下为仿真执行；"
    "仅安全工作空间内目标可执行）。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def move_to_pose(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 3.1416,
    yaw: float = 0.0,
    velocity_fraction: float = 0.1,
    timeout_sec: float = 10.0,
) -> str:
    values = (x, y, z, roll, pitch, yaw, velocity_fraction, timeout_sec)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("pose values must be finite numbers")
    radius = math.hypot(x, y)
    r_lo, r_hi = _SAFE_RADIUS
    z_lo, z_hi = _SAFE_Z
    if not (r_lo <= radius <= r_hi):
        raise ValueError(
            f"target radius {radius:.3f}m outside safe workspace [{r_lo}, {r_hi}]"
        )
    if not (z_lo <= z <= z_hi):
        raise ValueError(f"target z {z}m outside safe window [{z_lo}, {z_hi}]")
    if not (0.01 <= velocity_fraction <= 1.0):
        raise ValueError("velocity_fraction out of range [0.01, 1.0]")
    if not (0.5 <= timeout_sec <= 60.0):
        raise ValueError("timeout_sec out of range [0.5, 60]")
    _state["pose"] = {
        "x": float(x), "y": float(y), "z": float(z),
        "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
    }
    _state["moving"] = False
    _state["last_motion"] = {
        "kind": "move_to_pose",
        "target": dict(_state["pose"]),
        "executed_at": _ts(),
    }
    return json.dumps(
        {
            "ok": True,
            "driver": "completed",
            "evidence_domain": "simulation",
            "pose": dict(_state["pose"]),
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="ur5e.stop",
    description="停止当前运动并保持状态（物理动作；SIM 下为仿真执行）。",
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def stop() -> str:
    _state["moving"] = False
    return json.dumps(
        {
            "ok": True,
            "driver": "completed",
            "evidence_domain": "simulation",
            "stopped": True,
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


def main() -> None:
    # 六审 §4.4.5：第一方 SIM 工具声明严格对象边界（函数签名即完整
    # 参数边界；FastMCP 默认不产出 additionalProperties:false）。
    for tool in server._tool_manager._tools.values():
        if isinstance(tool.parameters, dict):
            tool.parameters["additionalProperties"] = False
    server.run()


if __name__ == "__main__":
    main()
