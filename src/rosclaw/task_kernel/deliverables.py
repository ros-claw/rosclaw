"""交付物核验（R0-2，0826 体验审计 §5.R0-2）。

任务成功 ≠ 用户请求成功：spec 冻结的 required deliverables 必须
按 **kind** 出现在产物账本——preview_2d / scene_3d / robot_video
是不同产物 kind（lineage.kind 权威），2D 预览永远不能冒充场景
视频（0826 审计：GIF 交付冒充"仿真视频"是假绿）。
"""

from __future__ import annotations

import json
from typing import Any

from rosclaw.contracts.agent.task_spec import (
    DELIVERABLE_KIND_TO_ARTIFACT_KIND,
)


def artifact_delivery_kind(artifact: dict[str, Any]) -> str:
    """产物的交付 kind（lineage.kind 权威；无血缘为 data——
    不可用于媒体交付断言）。"""
    meta = artifact.get("metadata_json")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except ValueError:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    lineage = meta.get("lineage") or {}
    kind = str(lineage.get("kind", "")) if isinstance(lineage, dict) else ""
    return kind or "data"


def deliverable_verdict(
    deliverables: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """required 交付物按 kind 匹配产物账本。

    返回 {satisfied, missing, partial}：missing = required 但无对应
    kind 产物的 kind 列表；partial = 有 required 命中但也有缺失。
    """
    present_kinds = {artifact_delivery_kind(a) for a in artifacts}
    satisfied: list[str] = []
    missing: list[str] = []
    for item in deliverables:
        kind = str(item.get("kind", ""))
        if not kind:
            continue
        required = bool(item.get("required"))
        artifact_kind = DELIVERABLE_KIND_TO_ARTIFACT_KIND.get(kind, kind)
        media_type = str(item.get("media_type", ""))
        hit = False
        if artifact_kind in present_kinds:
            if not media_type:
                hit = True
            else:
                hit = any(
                    artifact_delivery_kind(a) == artifact_kind
                    and str(a.get("media_type", "")) == media_type
                    for a in artifacts
                )
        if hit:
            satisfied.append(kind)
        elif required:
            missing.append(kind)
    return {
        "satisfied": satisfied,
        "missing": missing,
        "partial": bool(satisfied and missing),
    }


__all__ = ["artifact_delivery_kind", "deliverable_verdict"]
