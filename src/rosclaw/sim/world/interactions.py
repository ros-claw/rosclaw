"""Interaction Contract（PR-MH7，规格 §23/§24）。

每个 interaction 是 typed target + affordance + action schema +
preconditions/success/effects + DAG 依赖——不是文字描述。

**不允许"假的 Affordance"**（§24）：Body 没有 gripper 时 grasp 不能
因为任务需要就偷偷 weld payload；interaction 受 Body capability +
World affordance + Current state 共同约束（W09 L08 的自然延伸）。
"""

from __future__ import annotations

from typing import Any

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
    grippers: dict[str, bool],
) -> None:
    """能力诚实检查（fail closed）。

    grippers: body_ref id → 是否有夹爪执行器（从编译后模型推导，
    不从机器人名字猜）。grasp 类 affordance 要求至少一个被挂 body
    有真实夹爪——没有机器人的世界也不能假装能抓。
    """
    has_any_gripper = any(grippers.values())
    for point in interactions:
        if requires_gripper(point["affordance"]) and not has_any_gripper:
            raise ValueError(
                f"CAPABILITY_UNAVAILABLE: affordance {point['affordance']!r} "
                f"requires a gripper, but no attached body provides one"
            )


def evaluate_predicates(
    observations: dict[str, Any],
    predicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """机器谓词求值：输入 channel → 值映射，输出逐条 ok 判定。

    observations: {channel: {"pos": [...], "quat": [...]} 或 [value]}。
    """
    results = []
    for predicate in predicates:
        channel = predicate["channel"]
        field = predicate["field"]
        raw = observations.get(channel)
        if raw is None:
            results.append({"predicate": predicate, "ok": False, "reason": "channel_missing"})
            continue
        value = raw.get(field) if isinstance(raw, dict) else raw
        if "inside" in predicate:
            box = predicate["inside"]
            ok = all(lo <= v <= hi for v, lo, hi in zip(value, box["min"], box["max"], strict=True))
            results.append({"predicate": predicate, "ok": bool(ok), "value": value})
        else:
            near = predicate["near"]
            distance = sum((v - t) ** 2 for v, t in zip(value, near["target"], strict=True)) ** 0.5
            results.append(
                {
                    "predicate": predicate,
                    "ok": bool(distance <= near["tolerance"]),
                    "distance": distance,
                }
            )
    return results
