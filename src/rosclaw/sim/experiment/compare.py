"""实验对比（PR-MH5，规格 §20/§21）：指标表 + best + Pareto 候选。"""

from __future__ import annotations

from typing import Any

#: Pareto 比较轴（全部越小越好）。
PARETO_METRICS = ("tracking_rmse", "energy_end", "peak_qvel")


def build_metric_table(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个实验一行：ref + success + 关键指标。"""
    rows = []
    for receipt in receipts:
        metrics = receipt.get("metrics", {})
        rows.append(
            {
                "receipt_ref": receipt["_ref"],
                "model_ref": receipt.get("model_ref", ""),
                "success": receipt.get("success"),
                "tracking_rmse": metrics.get("tracking_rmse", 0.0),
                "overshoot": metrics.get("overshoot", 0.0),
                "settling_time_s": metrics.get("settling_time_s", 0.0),
                "peak_qvel": metrics.get("peak_qvel", 0.0),
                "energy_end": metrics.get("energy_end", 0.0),
                "collision": metrics.get("collision", False),
            }
        )
    return rows


def pareto_front(
    rows: list[dict[str, Any]], metrics: tuple[str, ...] = PARETO_METRICS
) -> list[str]:
    """非支配解集（全部轴越小越好；success=False 不参与支配）。"""
    candidates = [row for row in rows if row.get("success") is not False]
    front = []
    for row in candidates:
        dominated = any(
            all(other[m] <= row[m] for m in metrics) and any(other[m] < row[m] for m in metrics)
            for other in candidates
            if other is not row
        )
        if not dominated:
            front.append(row["receipt_ref"])
    return front


def best_experiment(rows: list[dict[str, Any]], pareto: list[str]) -> str:
    """best：Pareto 内 tracking_rmse 最小（全失败时退化为误差最小）。"""
    if not rows:
        return ""
    pool = [row for row in rows if row["receipt_ref"] in pareto] or rows
    return min(pool, key=lambda row: row["tracking_rmse"])["receipt_ref"]
