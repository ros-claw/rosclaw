"""Interaction Contract（PR-MH7，规格 §23/§24）。

每个 interaction 是 typed target + affordance + action schema +
preconditions/success/effects + DAG 依赖——不是文字描述。

**不允许"假的 Affordance"**（§24）：Body 没有 gripper 时 grasp 不能
因为任务需要就偷偷 weld payload；interaction 受 Body capability +
World affordance + Current state 共同约束（W09 L08 的自然延伸）。
"""

from __future__ import annotations

from typing import Any

from rosclaw.sim.experiment.predicates import evaluate_predicates  # noqa: F401 兼容再导出

#: 需要夹爪能力的 affordance（无 gripper → CAPABILITY_UNAVAILABLE）。
GRASP_LIKE_AFFORDANCES = frozenset({"grasp"})


def requires_gripper(affordance: str) -> bool:
    return affordance in GRASP_LIKE_AFFORDANCES


def interaction_order(interactions: list[dict[str, Any]]) -> list[str]:
    """DAG 顺序（validation 已保证 depends_on 只引用前面的 id）。"""
    return [point["id"] for point in interactions]


def check_body_capability(
    interactions: list[dict[str, Any]],
    *,
    capabilities: dict[str, dict[str, Any]],
) -> None:
    """能力诚实检查（fail closed，0915 §五）。

    capabilities: body_ref id → resolve_grasp_capability 三态结果。
    grasp 类 affordance 要求至少一个被挂 body 为 AVAILABLE
    （声明+证明都成立）；UNDECLARED/UNPROVEN 一律拒绝——没有机器人
    的世界也不能假装能抓。
    """
    has_any_gripper = any(v.get("status") == "AVAILABLE" for v in capabilities.values())
    for point in interactions:
        if requires_gripper(point["affordance"]) and not has_any_gripper:
            summary = {key: v.get("status") for key, v in capabilities.items()} or "no_bodies"
            raise ValueError(
                f"CAPABILITY_UNAVAILABLE: affordance {point['affordance']!r} "
                f"requires a gripper, but no attached body provides one "
                f"(capabilities: {summary})"
            )
