"""Rollout 引擎（PR-MH3，ADR-0014，规格 §15）。

Maturity: experimental（ADR-0000 §4）。

有界物理 rollout：controller（hold / ctrl_series / position_targets）
+ 强制预算（max_steps / max_duration / max_wall_time / max_record_points /
max_trace_bytes）。逐步发散哨兵：状态出现 NaN/Inf → ``SIM_DIVERGED``。
trace 与 final state 全部不可变落盘，同输入内容寻址幂等。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from rosclaw.contracts.common import canonical_json

DEFAULT_BUDGETS: dict[str, Any] = {
    "max_steps": 200_000,
    "max_duration_s": 600.0,
    "max_wall_time_s": 60.0,
    "max_record_points": 480,
    "max_trace_bytes": 32 * 1024 * 1024,
    "max_branch_count": 64,
}


def resolve_steps(
    controller: dict[str, Any],
    *,
    steps: int | None,
    duration_s: float | None,
    timestep: float,
) -> int:
    """确定 rollout 步数：显式 steps > duration_s > ctrl_series 长度。"""
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError(f"ROLLOUT_STEPS_INVALID: {steps!r}")
        return steps
    if duration_s is not None:
        if (
            not isinstance(duration_s, (int, float))
            or not np.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError(f"ROLLOUT_DURATION_INVALID: {duration_s!r}")
        return max(1, round(float(duration_s) / timestep))
    series = controller.get("ctrl_series")
    if series is not None:
        return len(series)
    raise ValueError("ROLLOUT_STEPS_REQUIRED: hold/position_targets 需要 steps 或 duration_s")


def validate_controller(controller: Any, nu: int) -> dict[str, Any]:
    """校验 controller 形状（fail closed）。"""
    if not isinstance(controller, dict):
        raise ValueError(f"CONTROLLER_INVALID: controller must be a mapping, got {controller!r}")
    if controller.get("hold") is True:
        return {"kind": "hold"}
    series = controller.get("ctrl_series")
    if series is not None:
        if not isinstance(series, list) or not series:
            raise ValueError("CONTROLLER_INVALID: ctrl_series must be a non-empty list")
        rows: list[list[float]] = []
        for row in series:
            if not isinstance(row, list) or len(row) != nu:
                raise ValueError(f"CONTROLLER_INVALID: ctrl_series row must have {nu} entries")
            values = []
            for v in row:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
                    raise ValueError(f"CONTROLLER_INVALID: non-finite ctrl value {v!r}")
                values.append(float(v))
            rows.append(values)
        return {"kind": "ctrl_series", "rows": rows}
    targets = controller.get("position_targets")
    if targets is not None:
        if not isinstance(targets, list) or len(targets) != nu:
            raise ValueError(f"CONTROLLER_INVALID: position_targets must have {nu} entries")
        values = []
        for v in targets:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
                raise ValueError(f"CONTROLLER_INVALID: non-finite target {v!r}")
            values.append(float(v))
        return {"kind": "position_targets", "values": values}
    raise ValueError(f"CONTROLLER_INVALID: unsupported controller {controller!r}")


def run_rollout(
    model,
    data,
    *,
    plan: dict[str, Any],
    steps: int,
    budgets: dict[str, Any] | None = None,
    visit=None,
) -> tuple[list[dict[str, Any]], int]:
    """执行 rollout，返回（采样状态序列, 实际步数）。发散即 SIM_DIVERGED。

    visit: 可选回调 visit(data, step)，每步有限性哨兵通过后调用
    （指标采集用，如 experiment.metrics）。
    """
    import mujoco

    merged = {**DEFAULT_BUDGETS, **(budgets or {})}
    if steps > merged["max_steps"]:
        raise ValueError(f"SIM_BUDGET_EXCEEDED: steps {steps} > {merged['max_steps']}")
    duration_s = steps * float(model.opt.timestep)
    if duration_s > merged["max_duration_s"]:
        raise ValueError(
            f"SIM_BUDGET_EXCEEDED: duration {duration_s}s > {merged['max_duration_s']}s"
        )

    if plan["kind"] == "hold":
        hold_ctrl = [float(v) for v in data.ctrl]
    elif plan["kind"] == "position_targets":
        for i, v in enumerate(plan["values"]):
            data.ctrl[i] = v

    stride = max(1, -(-steps // merged["max_record_points"]))  # 有界采样
    states: list[dict[str, Any]] = []

    def _record() -> None:
        states.append(
            {
                "t": float(data.time),
                "qpos": [float(v) for v in data.qpos],
                "qvel": [float(v) for v in data.qvel],
                "ctrl": [float(v) for v in data.ctrl],
            }
        )

    _record()
    started = time.monotonic()
    for step in range(steps):
        if plan["kind"] == "ctrl_series" and step < len(plan["rows"]):
            for i, v in enumerate(plan["rows"][step]):
                data.ctrl[i] = v
        elif plan["kind"] == "hold":
            for i, v in enumerate(hold_ctrl):
                data.ctrl[i] = v
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise ValueError(f"SIM_DIVERGED: non-finite state at step {step + 1}")
        if visit is not None:
            visit(data, step)
        if (step + 1) % stride == 0 or step == steps - 1:
            _record()
        if step % 4096 == 4095 and time.monotonic() - started > merged["max_wall_time_s"]:
            raise ValueError(f"SIM_BUDGET_EXCEEDED: wall time > {merged['max_wall_time_s']}s")
    return states, steps


def states_digest(states: list[dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(states).encode("utf-8")).hexdigest()
