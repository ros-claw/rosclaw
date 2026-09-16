"""任务谓词机器求值（PR-MH9，0915 优化 §三）。

success/failure 只开放 inside / near 两种机器谓词（与
sim/world/validation.py 校验的形式一一对应）。求值器放在
experiment 层（WorldSpec 与 Receipt 共用），不从 Agent 文本判定。
"""

from __future__ import annotations

from typing import Any


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
        if value is None:
            results.append({"predicate": predicate, "ok": False, "reason": "channel_missing"})
            continue
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
