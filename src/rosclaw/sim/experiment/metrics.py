"""实验指标采集（PR-MH5，规格 §20/§45）。

指标在 rollout 过程中逐步采集（不经采样 trace——采样会漏掉峰值），
由 ``run_rollout`` 的 visit 回调驱动。全部机器计算，不靠 LLM 读数。
"""

from __future__ import annotations

import math
from typing import Any


class MetricCollector:
    """逐步采集跟踪/速度/能量/接触指标。"""

    def __init__(
        self,
        model,  # noqa: ANN001
        data,  # noqa: ANN001
        tracked: list[tuple[int, int]],
        series_rows: list[list[float]] | None = None,
    ) -> None:
        """tracked: [(ctrl_index, qpos_addr)]——被跟踪的 position 伺服关节。

        series_rows: ctrl_series 控制器的逐行目标（逐步目标）；
        否则目标取构造时 data.ctrl（hold/position_targets 需先应用）。
        """
        self._model = model
        self._tracked = tracked
        self._series_rows = series_rows
        self.targets = [float(data.ctrl[ctrl_i]) for ctrl_i, _ in tracked]
        self.initials = [float(data.qpos[addr]) for _, addr in tracked]
        self._err_sq = [0.0] * len(tracked)
        self._overshoot = [0.0] * len(tracked)
        self._series: list[list[float]] = []  # 每步 tracked 值（settling 用）
        self._steps = 0
        self.peak_qvel = 0.0
        self.energy0: float | None = None
        self.energy_end: float | None = None
        self.contact_max_penetration = 0.0

    def visit(self, data, step: int) -> None:  # noqa: ANN001
        import mujoco

        self._steps += 1
        row = []
        for k, (ctrl_i, addr) in enumerate(self._tracked):
            value = float(data.qpos[addr])
            row.append(value)
            if self._series_rows is not None:
                target = self._series_rows[min(step, len(self._series_rows) - 1)][ctrl_i]
            else:
                target = self.targets[k]
            err = value - target
            self._err_sq[k] += err * err
            direction = 1.0 if target - self.initials[k] >= 0 else -1.0
            self._overshoot[k] = max(self._overshoot[k], (value - target) * direction)
        self._series.append(row)
        if data.qvel.size:
            self.peak_qvel = max(self.peak_qvel, float(max(abs(v) for v in data.qvel)))
        mujoco.mj_energyPos(self._model, data)
        potential = float(data.energy[0])
        mujoco.mj_energyVel(self._model, data)
        total = potential + float(data.energy[1])
        if self.energy0 is None:
            self.energy0 = total
        self.energy_end = total
        for i in range(data.ncon):
            self.contact_max_penetration = min(
                self.contact_max_penetration, float(data.contact[i].dist)
            )

    def finalize(self, *, timestep: float, duration_s: float) -> dict[str, Any]:
        count = max(1, self._steps)
        tracking_rmse = (
            math.sqrt(sum(self._err_sq) / (count * max(1, len(self._tracked))))
            if self._tracked
            else 0.0
        )
        overshoot = max(self._overshoot, default=0.0)
        overshoot = max(0.0, overshoot)
        settling = 0.0
        if self._tracked and self._series:
            k = 0  # 主跟踪关节
            target = self._series_rows[-1][k] if self._series_rows is not None else self.targets[k]
            band = 0.02 * max(abs(target - self.initials[k]), 1e-9)
            last_outside = -1
            for index, row in enumerate(self._series):
                if abs(row[k] - target) > band:
                    last_outside = index
            settling = (last_outside + 1) * timestep if last_outside >= 0 else 0.0
        return {
            "tracking_rmse": tracking_rmse,
            "overshoot": overshoot,
            "settling_time_s": settling,
            "peak_qvel": self.peak_qvel,
            "energy0": self.energy0 or 0.0,
            "energy_end": self.energy_end or 0.0,
            "contact_max_penetration": self.contact_max_penetration,
            "collision": self.contact_max_penetration < -1e-3,
            "duration_s": duration_s,
        }
