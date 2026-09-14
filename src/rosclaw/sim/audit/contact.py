"""接触类 audit（PR-MH4，规格 §17.1）：A05/A06。

初始穿透必须在 mj_forward 后、mj_step 前检查（warmup 会隐藏
初始穿模——Text2Mujoco 的关键经验）；序列穿透逐步检查，不能只
查首尾。
"""

from __future__ import annotations

from typing import Any

from rosclaw.sim.audit.context import AuditContext


def _deepest_contact(data) -> tuple[float, int]:  # noqa: ANN001
    deepest, index = 0.0, -1
    for i in range(data.ncon):
        dist = float(data.contact[i].dist)
        if dist < deepest:
            deepest, index = dist, i
    return deepest, index


def a05_initial_penetration(ctx: AuditContext) -> dict[str, Any]:
    """A05：qpos0 编译态与 reset 后两态各查一次初始穿透。"""
    worst = 0.0
    for _ in range(2):  # 编译态 + reset 后（Text2Mujoco：warmup 可能隐藏）
        data = ctx.fresh_data()
        deepest, _ = _deepest_contact(data)
        worst = min(worst, deepest)
    status = "FAIL" if worst < ctx.policy.static_penetration_m else "PASS"
    return {
        "status": status,
        "min_distance_m": worst,
        "violations": [] if status == "PASS" else [{"min_distance_m": worst}],
    }


def a06_sequence_penetration(ctx: AuditContext) -> dict[str, Any]:
    """A06：整个轨迹逐步检查穿透（有 trace 按其 controller 重放，
    否则 hold 扫描 policy.sweep_s 秒）。"""
    worst = 0.0
    worst_step = -1

    def visit(data, step) -> None:  # noqa: ANN001
        nonlocal worst, worst_step
        deepest, _ = _deepest_contact(data)
        if deepest < worst:
            worst, worst_step = deepest, step

    ctrl = None
    seconds = ctx.policy.sweep_s
    if ctx.trace_record is not None:
        controller = ctx.trace_record.get("controller", {})
        if "ctrl_series" in controller:
            ctrl = {"kind": "ctrl_series", "rows": controller["ctrl_series"]}
        elif "position_targets" in controller:
            ctrl = {"kind": "ctrl", "values": controller["position_targets"]}
        seconds = ctx.trace_record.get("steps", 1) * float(ctx.model.opt.timestep)

    ctx.sweep(seconds, visit, ctrl=ctrl)
    status = "FAIL" if worst < ctx.policy.run_penetration_m else "PASS"
    violations = [] if status == "PASS" else [{"min_distance_m": worst, "step": worst_step}]
    return {
        "status": status,
        "min_distance_m": worst,
        "step": worst_step,
        "violations": violations,
    }
