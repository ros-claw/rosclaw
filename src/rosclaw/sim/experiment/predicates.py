"""任务谓词机器求值（PR-MH9/MH12，0915 §三 + 0916 §十五）。

success/failure 谓词全部机器可执行（**不允许自然语言条件成为
verifier truth**）。每个谓词返回 {predicate, ok, measured, threshold}。
v1：inside / near；v2（MH12）：contact / joint_in_range / upright /
speed_below。
"""

from __future__ import annotations

from typing import Any

_PREDICATE_FORMS = ("inside", "near", "contact", "joint_in_range", "upright", "speed_below")

_ROLE_DEFAULT_FIELD = {
    "inside": "pos",
    "near": "pos",
    "upright": "quat",
}


def evaluate_predicates(
    observations: dict[str, Any],
    predicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """机器谓词求值：输入 channel → 值映射，输出逐条 ok 判定。

    observations: {channel: {"pos": [...], "quat": [...]} 或 [value]}，
    以及结构化通道：joint_positions/joint_velocities（list）、
    contact_pairs（list[dict]）。
    """
    results = []
    for predicate in predicates:
        results.append(_evaluate_one(observations, predicate))
    return results


def _evaluate_one(observations: dict[str, Any], predicate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(predicate, dict):
        return {"predicate": predicate, "ok": False, "reason": "predicate_not_mapping"}
    form = next((f for f in _PREDICATE_FORMS if f in predicate), None)
    if form is None:
        return {
            "predicate": predicate,
            "ok": False,
            "reason": f"predicate_form_unknown (allowed: {_PREDICATE_FORMS})",
        }

    channel = predicate.get("channel", "")
    raw = observations.get(channel)
    field = predicate.get("field") or _ROLE_DEFAULT_FIELD.get(form, "pos")
    value = raw.get(field) if isinstance(raw, dict) else raw

    if form == "inside":
        if value is None:
            return {"predicate": predicate, "ok": False, "reason": "channel_missing"}
        box = predicate["inside"]
        ok = all(lo <= v <= hi for v, lo, hi in zip(value, box["min"], box["max"], strict=True))
        measured = max(
            (max(box["min"][i] - v, v - box["max"][i]) for i, v in enumerate(value)),
            default=0.0,
        )
        return {
            "predicate": predicate,
            "ok": bool(ok),
            "measured": measured,
            "threshold": 0.0,
            "value": value,
        }

    if form == "near":
        if value is None:
            return {"predicate": predicate, "ok": False, "reason": "channel_missing"}
        near = predicate["near"]
        distance = sum((v - t) ** 2 for v, t in zip(value, near["target"], strict=True)) ** 0.5
        return {
            "predicate": predicate,
            "ok": bool(distance <= near["tolerance"]),
            "measured": distance,
            "threshold": near["tolerance"],
        }

    if form == "contact":
        spec = predicate["contact"]
        pairs = observations.get("contact_pairs") or []
        body_a, body_b = spec.get("body1"), spec.get("body2")
        best = None
        for pair in pairs:
            names = {pair.get("geom1"), pair.get("geom2")}
            bodies = {pair.get("body1"), pair.get("body2")} | names
            if (body_a is None or body_a in bodies) and (body_b is None or body_b in bodies):
                best = pair
                break
        if best is None:
            return {
                "predicate": predicate,
                "ok": False,
                "measured": None,
                "threshold": spec.get("max_dist", 0.0),
                "reason": "no_matching_contact",
            }
        dist = float(best.get("dist", 0.0))
        threshold = float(spec.get("max_dist", 0.0))
        return {
            "predicate": predicate,
            "ok": bool(dist <= threshold),
            "measured": dist,
            "threshold": threshold,
        }

    if form == "joint_in_range":
        positions = observations.get("joint_positions")
        if positions is None:
            return {"predicate": predicate, "ok": False, "reason": "channel_missing"}
        spec = predicate["joint_in_range"]
        index = int(spec.get("index", 0))
        measured = float(positions[index])
        lo, hi = float(spec["min"]), float(spec["max"])
        return {
            "predicate": predicate,
            "ok": bool(lo <= measured <= hi),
            "measured": measured,
            "threshold": [lo, hi],
        }

    if form == "upright":
        quat = value
        if quat is None:
            return {"predicate": predicate, "ok": False, "reason": "channel_missing"}
        import math

        x, y, z, w = quat  # 本 harness 约定 xyzw 序
        z_axis = [
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, z_axis[2]))))
        threshold = float(predicate["upright"].get("max_tilt_deg", 15.0))
        return {
            "predicate": predicate,
            "ok": bool(tilt <= threshold),
            "measured": tilt,
            "threshold": threshold,
        }

    if form == "speed_below":
        velocities = observations.get("joint_velocities")
        if velocities is None:
            return {"predicate": predicate, "ok": False, "reason": "channel_missing"}
        threshold = float(predicate["speed_below"]["max"])
        measured = max((abs(v) for v in velocities), default=0.0)
        return {
            "predicate": predicate,
            "ok": bool(measured <= threshold),
            "measured": measured,
            "threshold": threshold,
        }

    return {"predicate": predicate, "ok": False, "reason": "predicate_form_unknown"}
