"""LIMO 仿真 MCP server（PR-12）：SIMULATION 证据域的 LIMO 身体。

观测工具（readOnlyHint=true → OBSERVE）：
- ``limo.localization.get_pose`` — 仿真位姿（SIMULATED 证据，永不证明 REAL）
- ``limo.health`` — 仿真健康状态

动作工具（无 readOnlyHint + 动作动词 → PHYSICAL_ACTION）：
- ``limo.speaker.play_tone`` — 仿真扬声器
- ``limo.localization.set_initial_pose`` — 设置仿真初始位姿

模型永远不能经 ToolCatalog 调用动作工具；agentd 的 SimActionChannel
（SIM 物理权威）在 EXACT_ACTION grant 验证后才直接调用它们。
"""

from __future__ import annotations

import json
import math
import time

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("limo-sim")

_state = {
    "pose": {"x": 1.25, "y": -0.5, "theta": 0.03},
    "tones": [],
    "boot_time": time.time(),
}


def _ts() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


@server.tool(
    name="limo.localization.get_pose",
    description="读取 LIMO 当前定位位姿（仿真观测，SIMULATED 证据域）。",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_pose(frame: str = "map") -> str:
    return json.dumps(
        {
            "frame": frame,
            "evidence_domain": "simulation",
            **dict(_state["pose"]),
            "covariance": [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03],
            "timestamp": _ts(),
            "fresh": True,
        },
        ensure_ascii=False,
    )


@server.tool(
    name="limo.health",
    description="LIMO 仿真健康状态（电量/驱动/定位在线）。",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def health() -> str:
    return json.dumps(
        {
            "evidence_domain": "simulation",
            "battery_percent": 87,
            "drive": "online",
            "localization": "online",
            "uptime_sec": int(time.time() - _state["boot_time"]),
            "timestamp": _ts(),
        },
        ensure_ascii=False,
    )


@server.tool(
    name="limo.speaker.play_tone",
    description="播放提示音（物理动作；SIM 下为仿真执行）。",
)
def play_tone(frequency_hz: int = 660, duration_sec: float = 0.25, volume_percent: int = 18) -> str:
    if not (20 <= frequency_hz <= 20_000):
        raise ValueError(f"frequency {frequency_hz} out of range")
    if not (0.01 <= duration_sec <= 5.0):
        raise ValueError(f"duration {duration_sec} out of range")
    if not (0 <= volume_percent <= 100):
        raise ValueError(f"volume {volume_percent} out of range")
    record = {
        "frequency_hz": frequency_hz,
        "duration_sec": duration_sec,
        "volume_percent": volume_percent,
        "executed_at": _ts(),
        "evidence_domain": "simulation",
        "acoustic_observation": None,  # 无麦克风：只证明驱动执行
    }
    _state["tones"].append(record)
    return json.dumps({"driver": "completed", **record}, ensure_ascii=False)


@server.tool(
    name="limo.localization.set_initial_pose",
    description="设置地图初始位姿（物理动作；SIM 下为仿真执行）。",
)
def set_initial_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> str:
    for value in (x, y, yaw):
        if not math.isfinite(value):
            raise ValueError("pose values must be finite")
    _state["pose"] = {"x": float(x), "y": float(y), "theta": float(yaw)}
    return json.dumps(
        {
            "driver": "completed",
            "evidence_domain": "simulation",
            "pose": dict(_state["pose"]),
            "executed_at": _ts(),
        },
        ensure_ascii=False,
    )


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
