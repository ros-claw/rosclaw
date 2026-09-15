"""Audit 上下文（PR-MH4）。

把 backend 的模型/状态/轨迹能力收敛成 check 函数的简单工作面：
编译后 MjModel、来源 MjSpec、MJCF 文本、新 MjData 工厂、
直接步进扫描（不经 store，快速）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.sim.audit.policy import STRICT_POLICY, AuditPolicy


@dataclass
class AuditContext:
    """一次审计的工作面。"""

    model: Any  # mujoco.MjModel
    spec: Any  # mujoco.MjSpec
    xml_text: str
    policy: AuditPolicy = STRICT_POLICY
    trace_record: dict[str, Any] | None = None
    state_ref: str | None = None
    restore_fn: Any = None  # (state_ref) -> (model, data)，A20 用
    extra: dict[str, Any] = field(default_factory=dict)

    def fresh_data(self):  # noqa: ANN202
        """新 MjData + mj_forward（qpos0 状态）。"""
        import mujoco

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return data

    def sweep(self, seconds: float, visit, *, ctrl: dict[str, Any] | None = None) -> dict[str, Any]:
        """直接步进扫描：每步调 visit(data, step) 收集结果。

        ctrl: {"kind": "hold"} 或 {"kind": "ctrl", "values": [...]} 或
        {"kind": "ctrl_series", "rows": [...]}。
        """
        import math

        import mujoco
        import numpy as np

        data = self.fresh_data()
        if ctrl and ctrl.get("kind") == "ctrl":
            for i, v in enumerate(ctrl["values"]):
                data.ctrl[i] = v
        steps = max(1, math.ceil(seconds / float(self.model.opt.timestep)))
        out: dict[str, Any] = {"diverged": False, "step_count": steps}
        for step in range(steps):
            if ctrl and ctrl.get("kind") == "ctrl_series" and step < len(ctrl["rows"]):
                for i, v in enumerate(ctrl["rows"][step]):
                    data.ctrl[i] = v
            mujoco.mj_step(self.model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                out["diverged"] = True
                out["diverged_step"] = step + 1
                break
            visit(data, step)
        out["final"] = data
        return out
