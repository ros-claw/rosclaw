"""PlanRef 跨进程 conformance（0827 体验审计 P0-6）。

启动/快照期探针：生产者（ur5e.plan_cartesian_path 的 PlanStore）
写一条探针 plan，消费者（SimTrajectoryService._load_plan——生产
链用的解析器）必须能解析。不可解析 = 生产者/消费者不共享存储
（home 分裂/内存回落）——工具对必须排除出模型上下文（不兼容
工具在场 = 模型必然撞 REF_NOT_FOUND 后瞎试）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: PlanRef 生产者/消费者对（conformance 失败时整对排除——半个链
#: 在场比不在场更糟）。
_PLAN_REF_PAIR = (
    "ur5e.plan_cartesian_path",
    "ur5e.simulate_cartesian_trajectory",
)


def plan_ref_conformance(home: Path | str) -> list[dict[str, Any]]:
    """探针：生产者写入 → 消费者解析。返回排除清单（空 = 一致）。"""
    def _excluded(reason: str) -> list[dict[str, Any]]:
        return [
            {
                "capability_id": cid,
                "code": "REF_CONFORMANCE_FAILED",
                "reason": reason,
            }
            for cid in _PLAN_REF_PAIR
        ]

    producer_home = os.environ.get("ROSCLAW_HOME")
    if not producer_home:
        return _excluded(
            "ROSCLAW_HOME 未设置——生产者回落内存 PlanStore，"
            "跨进程不可解析"
        )
    if Path(producer_home).resolve() != Path(home).resolve():
        return _excluded(
            f"生产者/消费者 home 分裂：{producer_home} ≠ {home}"
        )
    # 探针：真实生产者 store 写入 → 真实消费者解析（不留探针文件）。
    import rosclaw.sim.ur5e_mcp as ur5e_mcp
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    store = ur5e_mcp._plan_store()
    record = store.put(
        {"hash": "conformance", "points": [], "shape": "probe"},
        "plan-ref conformance probe",
    )
    try:
        SimTrajectoryService(Path(home))._load_plan(str(record["plan_id"]))
    except Exception as exc:  # noqa: BLE001 - 不可解析即不一致
        return _excluded(f"探针 plan 消费者不可解析：{exc}"[:200])
    finally:
        path = getattr(store, "_path", None)
        if callable(path):
            path(str(record["plan_id"])).unlink(missing_ok=True)
    return []


__all__ = ["plan_ref_conformance"]
