"""逐条 Requirement 验收（0902 审计 R0-3，§3.1/R0-4）——PASS 必须附带
逐条 RequirementCoverage；未覆盖/未满足条款自动阻止终态。

0902 实证（P0 事故）：receipt tool_ref=""/overlays=[] 而用户要求
"红色圆柱笔 + 3D 实际轨迹 + 不要 2D"——没有任何机制把条款和 receipt
对上，系统宣布 PASS/DELIVERED 假成功。

每条条款三态判定（诚实——不可证不冒充）：
- SATISFIED：有证据满足；
- VIOLATED：有证据证明未满足（receipt 在场但字段为空）；
- UNVERIFIABLE：当前没有证据通道（颜色/接触——能力缺口诚实暴露，
  对 must 条款同样阻止终态）。

证据面（全部来自当前 revision 的账本——R0-1）：
- receipt：产物 lineage 的 render_receipt_path（登记时 kernel 核验
  digest——不是调用方自述）；
- plan：trace.json 的 plan_hash → sim/plans/plan_<hash16>.json；
- 产物 kind 集合：artifact_delivery_kind。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
UNVERIFIABLE = "UNVERIFIABLE"

#: 竖直平面命名面集合（compile_recipe_inputs 的 xz/yz 语义）。
_VERTICAL_PLANES = frozenset({"xz", "yz"})


def _load_receipts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """产物血缘 → render receipt（只读已核验路径——登记时已算
    digest）。"""
    receipts: list[dict[str, Any]] = []
    for artifact in artifacts:
        try:
            meta = json.loads(str(artifact.get("metadata_json") or "{}"))
        except ValueError:
            continue
        lineage = meta.get("lineage") or {}
        path = str(lineage.get("render_receipt_path") or "")
        if not path:
            continue
        try:
            receipts.append(json.loads(Path(path).read_text(
                encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return receipts


def _load_plan(home: Path, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """trace → plan_hash → plan 记录（shape/plane 的权威）。

    两条路（都是 kernel 打戳的账本事实，不是调用方自述）：
    1. 任务产物里的 trace.json（evidence 产物在账）；
    2. 产物 lineage.trace_id → sim/traces/<id>/trace.json（受信
       管道的渲染产物只带 trace 引用，trace 本体不直接入账——
       wp4 形态）。
    """
    trace_ids: list[str] = []
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or ""))
        if path.name == "trace.json" and path.exists():
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            plan_hash = str(trace.get("plan_hash") or "")
            if plan_hash:
                plan_path = (
                    Path(home) / "sim" / "plans"
                    / f"plan_{plan_hash[:16]}.json"
                )
                if plan_path.exists():
                    try:
                        return json.loads(plan_path.read_text(
                            encoding="utf-8"))
                    except ValueError:
                        return {}
        try:
            meta = json.loads(str(artifact.get("metadata_json") or "{}"))
        except ValueError:
            continue
        lineage = meta.get("lineage") or {}
        tid = str(lineage.get("trace_id") or "")
        if tid:
            trace_ids.append(tid)
    for tid in trace_ids:
        trace_path = Path(home) / "sim" / "traces" / tid / "trace.json"
        if not trace_path.exists():
            continue
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        plan_hash = str(trace.get("plan_hash") or "")
        if not plan_hash:
            continue
        plan_path = (
            Path(home) / "sim" / "plans" / f"plan_{plan_hash[:16]}.json"
        )
        if plan_path.exists():
            try:
                return json.loads(plan_path.read_text(encoding="utf-8"))
            except ValueError:
                return {}
    return {}


def _check_one(req: dict[str, Any], *, receipts: list[dict[str, Any]],
               plan: dict[str, Any], kinds: set[str],
               media: set[str]) -> dict[str, Any]:
    """单条条款判定。"""
    verifier = str(req.get("verifier") or "")
    claim = str(req.get("claim") or verifier)
    level = str(req.get("level") or "must")
    base = {"req_id": str(req.get("req_id") or ""), "claim": claim,
            "level": level, "verifier": verifier}

    def _done(status: str, evidence: str) -> dict[str, Any]:
        return {**base, "status": status, "evidence": evidence}

    if verifier.startswith("shape."):
        want = verifier.split(".", 1)[1]
        if not plan:
            return _done(UNVERIFIABLE, "无 plan 记录可核对形状")
        actual = str(plan.get("shape") or "")
        if actual == want:
            return _done(SATISFIED, f"plan.shape={actual}")
        return _done(VIOLATED, f"plan.shape={actual or '∅'} ≠ {want}")
    if verifier == "plan.plane.vertical":
        if not plan:
            return _done(UNVERIFIABLE, "无 plan 记录可核对平面")
        plane = str(plan.get("plane") or "")
        if plane in _VERTICAL_PLANES:
            return _done(SATISFIED, f"plan.plane={plane}")
        return _done(VIOLATED, f"plan.plane={plane or '∅'} 非竖直面")
    if verifier == "plan.plane.horizontal":
        if not plan:
            return _done(UNVERIFIABLE, "无 plan 记录可核对平面")
        plane = str(plan.get("plane") or "")
        if plane == "xy":
            return _done(SATISFIED, "plan.plane=xy")
        return _done(VIOLATED, f"plan.plane={plane or '∅'} 非水平面")
    if verifier == "receipt.tool_ref":
        if not receipts:
            return _done(UNVERIFIABLE, "无 render receipt 可核对工具挂载")
        mounted = [str(r.get("tool_ref") or "") for r in receipts]
        if any(mounted):
            return _done(SATISFIED, f"receipt.tool_ref={mounted}")
        return _done(VIOLATED, "receipt.tool_ref 全为空——未挂载工具")
    if verifier == "receipt.overlays.actual_eef_trace":
        if not receipts:
            return _done(UNVERIFIABLE, "无 render receipt 可核对轨迹叠加")
        for r in receipts:
            # R2-2 渲染器回写 overlays_applied（真实绘制的 overlay——
            # 宣称与画面一致）；overlays 为旧键兼容。
            overlays = r.get("overlays_applied")
            if not isinstance(overlays, list):
                overlays = r.get("overlays")
            if isinstance(overlays, list) and "actual_eef_trace" in overlays:
                return _done(SATISFIED, "overlays_applied 含 actual_eef_trace")
        if any(("overlays_applied" in r or "overlays" in r) for r in receipts):
            return _done(VIOLATED, "receipt.overlays_applied 不含实际轨迹")
        return _done(UNVERIFIABLE,
                     "receipt 无 overlays 字段——渲染器尚不支持轨迹叠加")
    if verifier == "render.tool_color":
        return _done(UNVERIFIABLE, "颜色无证据通道（渲染器不回写外观属性）")
    if verifier == "verification.contact":
        return _done(UNVERIFIABLE, "接触无证据通道（无接触验证器）")
    if verifier == "deliverable.scene_3d":
        if "scene_3d" in kinds:
            return _done(SATISFIED, "scene_3d 产物在账")
        return _done(VIOLATED, "无 scene_3d 产物")
    if verifier == "deliverable.video":
        if any(m.startswith("video/") for m in media):
            return _done(SATISFIED, "video/* 产物在账")
        return _done(VIOLATED, "无视频产物")
    if verifier == "deliverable.preview_2d":
        if "preview_2d" in kinds:
            return _done(SATISFIED, "preview_2d 产物在账")
        return _done(VIOLATED, "无 preview_2d 产物")
    if verifier == "delivery.not_2d_only":
        # forbidden：主交付不得只有 2D——有 scene_3d 或 video/* 即满足
        # 禁止项（"不违反"）。
        has_3d_or_video = "scene_3d" in kinds or any(
            m.startswith("video/") for m in media)
        if has_3d_or_video:
            return _done(SATISFIED, "存在 3D/视频交付——未违反禁止项")
        return _done(VIOLATED, "只有 2D 产物——违反禁止项")
    # 未知 verifier 键：不可证（不猜）。
    return _done(UNVERIFIABLE, f"未知验收键 {verifier}——不可证")


def check_requirements(
    *,
    home: Path,
    requirements: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    embodied: bool = True,
    receipts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """逐条验收。artifacts 必须是当前 revision 的产物集（R0-1）。

    embodied=False（无本体任务——写文件类）时跳过物理证据条款
    （shape/plane/tool/overlay/color/contact 的证据通道只在具身链
    存在——非具身任务上它们永远 UNVERIFIABLE，是噪声不是信号）；
    deliverable/delivery 条款（产物面）不受本体影响，照常核验。
    """
    if not requirements:
        return []
    from rosclaw.task_kernel.deliverables import artifact_delivery_kind

    receipts = receipts if receipts is not None else _load_receipts(artifacts)
    plan = _load_plan(home, artifacts)
    kinds = {artifact_delivery_kind(a) for a in artifacts}
    media = {str(a.get("media_type") or "") for a in artifacts}
    return [
        _check_one(r, receipts=receipts, plan=plan, kinds=kinds,
                   media=media)
        for r in requirements
        if embodied
        or not str(r.get("verifier") or "").startswith(
            ("shape.", "plan.", "receipt.", "render.", "verification.")
        )
    ]


def unmet_failures(verdicts: list[dict[str, Any]]) -> list[str]:
    """verdicts → 阻止终态的失败行（must 未满足/不可证 +
    forbidden 被违反）。"""
    failures: list[str] = []
    for v in verdicts:
        if v["status"] == SATISFIED:
            continue
        failures.append(
            f"REQUIREMENT_UNMET: {v['claim']}（{v['status']}——"
            f"{v['evidence']}）"
        )
    return failures


__all__ = [
    "SATISFIED", "UNVERIFIABLE", "VIOLATED",
    "check_requirements", "unmet_failures",
]
