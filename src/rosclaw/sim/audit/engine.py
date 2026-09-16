"""Audit 引擎（PR-MH4，规格 §19）：注册表 + 运行器 + 机器可读报告。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rosclaw.sim.audit import contact, determinism, dynamics, geometry
from rosclaw.sim.audit.context import AuditContext

#: check 注册表（规格 §17/§18 第一批 + ROSClaw 基础扩展）。
CHECKS: dict[str, Callable[[AuditContext], dict[str, Any]]] = {
    "A01_collision_coverage": geometry.a01_collision_coverage,
    "A02_explicit_mass": geometry.a02_explicit_mass,
    "A03_servo_hold": dynamics.a03_servo_hold,
    "A04_link_continuity": dynamics.a04_link_continuity,
    "A05_initial_penetration": contact.a05_initial_penetration,
    "A06_sequence_penetration": contact.a06_sequence_penetration,
    "A07_undeclared_self_overlap": geometry.a07_undeclared_self_overlap,
    "A08_marker_grounding": geometry.a08_marker_grounding,
    "A15_nan_inf": dynamics.a15_nan_inf,
    "A16_physics_divergence": dynamics.a16_physics_divergence,
    "A17_energy_explosion": dynamics.a17_energy_explosion,
    "A18_reset_determinism": determinism.a18_reset_determinism,
    "A19_replay_determinism": determinism.a19_replay_determinism,
    "A20_state_model_mismatch": determinism.a20_state_model_mismatch,
}


def run_checks(ctx: AuditContext, checks: list[str] | None = None) -> dict[str, Any]:
    """按注册表顺序运行 check，聚合机器可读结果。

    总状态：任一 FAIL → FAIL；否则任一 WARN → WARN；否则 PASS。
    单项 check 异常不外泄——记 ERROR 状态（fail closed 语义：
    审计器自身故障不能伪装成 PASS）。
    """
    names = checks if checks is not None else list(CHECKS)
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        raise ValueError(f"AUDIT_CHECK_UNKNOWN: {unknown}")

    results: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for name in names:
        try:
            outcome = CHECKS[name](ctx)
        except Exception as exc:  # noqa: BLE001 —— 审计器故障 ≠ 模型通过
            outcome = {
                "status": "ERROR",
                "violations": [{"reason": "audit_check_crashed", "error": str(exc)}],
            }
        results[name] = outcome
        for violation in outcome.get("violations", []):
            violations.append({"check": name, **violation})
        for warning in outcome.get("warnings", []):
            warnings.append({"check": name, **warning})

    statuses = {outcome["status"] for outcome in results.values()}
    if "FAIL" in statuses or "ERROR" in statuses:
        status = "FAIL"
    elif "WARN" in statuses:
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": results, "violations": violations, "warnings": warnings}
