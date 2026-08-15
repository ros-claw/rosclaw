"""Built-in agentd tools (P0).

SIM-only, honest tools: they never claim physical truth. ``sim_state``
reports a deterministic simulated body state explicitly marked as
simulation; real daemon-backed tools (get_robot_state via rosclawd) are
wired in a later PR through the daemon client — the registry is the seam.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rosclaw.agentd.models.gateway import StrictTool
from rosclaw.contracts.common import ValidationError

SIM_STATE_TOOL = "sim_get_state"
SIM_BODY_TOOL = "sim_body_profile"
SIM_REACH_TOOL = "sim_reach"
# 十四审 PR-14.6：SIM 动力学闭环正式能力（COMPUTE，SIMULATION 限定）。
TRAJ_PLAN_TOOL = "trajectory_generate_planar_path"
TRAJ_SIMULATE_TOOL = "ur5e_simulate_cartesian_trajectory"
TRAJ_RENDER_TOOL = "simulation_render_trace"
TRAJ_VERIFY_TOOL = "simulation_verify_tracking"

_TOOL_SCHEMAS: dict[str, StrictTool] = {
    SIM_STATE_TOOL: StrictTool(
        name=SIM_STATE_TOOL,
        description=(
            "Read the SIMULATED body state (joints, health). "
            "Evidence class: simulated — never usable as REAL proof."
        ),
        parameters={
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
            "required": ["verbose"],
            "additionalProperties": False,
        },
    ),
    SIM_BODY_TOOL: StrictTool(
        name=SIM_BODY_TOOL,
        description="Read the bound body's static profile summary.",
        parameters={
            "type": "object",
            "properties": {"detail": {"type": "boolean"}},
            "required": ["detail"],
            "additionalProperties": False,
        },
    ),
    SIM_REACH_TOOL: StrictTool(
        name=SIM_REACH_TOOL,
        description=(
            "Execute one MuJoCo physics reach with the SIMULATED UR5e to the "
            "given tabletop target (meters) and return the verified receipt: "
            "policy check, resource lease, physics steps, collision check and "
            "the TASK_VERIFIED predicate. Evidence class: simulated."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "target x in meters"},
                "y": {"type": "number", "description": "target y in meters"},
                "z": {"type": "number", "description": "target z in meters"},
            },
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
    ),
}


class BuiltinToolRegistry:
    """Allowlisted executor for the agentd's own P0 tools."""

    def __init__(self, *, body_id: str, body_summary: str, home=None) -> None:
        self._body_id = body_id
        self._body_summary = body_summary
        self._home = home

    def strict_tools(self, names: list[str]) -> list[StrictTool]:
        return [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in _TOOL_SCHEMAS:
            raise ValidationError(f"tool {name!r} not allowlisted")
        if name == SIM_STATE_TOOL:
            return json.dumps(
                {
                    "evidence_class": "simulated",
                    "mode": "SIMULATION",
                    "body_id": self._body_id,
                    "joints_rad": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                    "health": "OK",
                    "fresh": True,
                },
                ensure_ascii=False,
            )
        if name == SIM_REACH_TOOL:
            return await asyncio.to_thread(self._execute_reach, arguments)
        if name in (
            TRAJ_PLAN_TOOL, TRAJ_SIMULATE_TOOL, TRAJ_RENDER_TOOL, TRAJ_VERIFY_TOOL,
        ):
            return await asyncio.to_thread(self._execute_trajectory_tool, name, arguments)
        return json.dumps(
            {
                "evidence_class": "configured",
                "body_id": self._body_id,
                "summary": self._body_summary,
            },
            ensure_ascii=False,
        )

    def _execute_trajectory_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """十四审 PR-14.6：SIM 动力学闭环四能力（COMPUTE——纯计算/
        渲染/验证，无物理副作用；证据 SIM_DYN_ROLLOUT=simulated）。"""
        from pathlib import Path as _Path

        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        if self._home is None:
            raise ValidationError("sim trajectory tools require rosclaw home")
        svc = SimTrajectoryService(_Path(self._home))
        try:
            if name == TRAJ_PLAN_TOOL:
                result = svc.generate_planar_path(
                    shape=str(arguments.get("shape", "star5")),
                    center_m=arguments.get("center_m") or [0.35, 0.25, 0.30],
                    scale_m=float(arguments.get("scale_m", arguments.get("radius_m", 0.10))),
                    plane=str(arguments.get("plane", "xy")),
                    max_segment_m=float(arguments.get("max_segment_m", 0.02)),
                )
                # 模型视图不带完整 points（载荷留文件——句柄+摘要）。
                result = {k: v for k, v in result.items() if k != "points"}
            elif name == TRAJ_SIMULATE_TOOL:
                result = svc.simulate_cartesian_trajectory(
                    str(arguments["plan_id"])
                )
            elif name == TRAJ_RENDER_TOOL:
                result = svc.render_trace(
                    str(arguments["trace_id"]),
                    format=str(arguments.get("format", "gif")),
                )
            else:
                result = svc.verify_tracking(
                    str(arguments["trace_id"]),
                    max_tracking_error_m=float(arguments["max_tracking_error_m"]),
                )
        except (ValueError, KeyError) as exc:
            raise ValidationError(f"{name} 参数错误: {exc}") from exc
        result["evidence_class"] = "simulated"
        return json.dumps(result, ensure_ascii=False)

    def _execute_reach(self, arguments: dict[str, Any]) -> str:
        from rosclaw.product.demo import DemoConfigurationError, run_demo

        try:
            target = (
                float(arguments["x"]),
                float(arguments["y"]),
                float(arguments["z"]),
            )
            receipt, receipt_path = run_demo(
                "ur5e-reach",
                target=target,
                actor_id="rosclaw-agentd",
                agent_framework="agentd",
            )
        except (DemoConfigurationError, OverflowError) as exc:
            raise ValidationError(f"sim_reach target invalid: {exc}") from exc
        simulation = receipt.simulation_result or {}
        verification = receipt.verification_result or {}
        return json.dumps(
            {
                "evidence_class": "simulated",
                "run_id": receipt.action_id,
                "target_m": list(target),
                "policy": receipt.policy_decision.get("reason"),
                "physics_steps": simulation.get("steps"),
                "collision_check": (
                    "PASS" if not simulation.get("collision") else "FAIL"
                ),
                "final_distance_m": verification.get("final_error_m"),
                "final_state": receipt.final_state.value,
                "task_success": bool(verification.get("success")),
                "evidence_verified": receipt.verified,
                "verification": verification,
                "receipt_path": str(receipt_path),
            },
            ensure_ascii=False,
        )
