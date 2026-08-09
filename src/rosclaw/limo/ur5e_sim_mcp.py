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

server = FastMCP("ur5e-sim")

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
}


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
