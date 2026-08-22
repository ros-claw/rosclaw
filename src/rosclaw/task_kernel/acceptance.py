"""AcceptanceCompilerV2（PR-N8）——验收条件按优先级合并。

优先级（方案 §七）：
  安全策略最低标准 → Capability 内置模板 → 用户显式要求
  → 任务类型默认标准 → 模型建议的更严格标准

不变量（测试钉住）：
- required_artifacts / visual_requirements / evidence_classes 取并集
  ——任何来源都不能删别人的约束（模型的 drop_* 声明被忽略）；
- numeric_thresholds 同键取 min（更严格者胜——模型放宽不生效）；
- 来源贡献记录在 spec.sources（可归因回放）。
"""

from __future__ import annotations

from typing import Any

from rosclaw.contracts.agent.acceptance import AcceptanceSpecV2
from rosclaw.contracts.common import new_id

_SOURCE_ORDER = (
    "safety_floor",
    "capability_template",
    "user_explicit",
    "task_default",
    "model_suggested",
)


def compile_acceptance(
    *,
    task_id: str,
    revision: int,
    safety_floor: dict[str, Any] | None = None,
    capability_template: dict[str, Any] | None = None,
    user_explicit: dict[str, Any] | None = None,
    task_default: dict[str, Any] | None = None,
    model_suggested: dict[str, Any] | None = None,
) -> AcceptanceSpecV2:
    parts = {
        "safety_floor": safety_floor or {},
        "capability_template": capability_template or {},
        "user_explicit": user_explicit or {},
        "task_default": task_default or {},
        "model_suggested": model_suggested or {},
    }
    artifacts: list[str] = []
    evidence: list[str] = []
    visual: list[str] = []
    postconditions: list[str] = []
    thresholds: dict[str, float] = {}
    modes: list[str] = []
    receipt = ""
    verifier_refs: list[str] = []
    provenance = False

    def _add_unique(target: list[str], values: Any) -> None:
        for v in values or []:
            if v not in target:
                target.append(str(v))

    for name in _SOURCE_ORDER:
        part = parts[name]
        _add_unique(artifacts, part.get("required_artifacts"))
        _add_unique(evidence, part.get("evidence_classes"))
        _add_unique(visual, part.get("visual_requirements"))
        _add_unique(postconditions, part.get("postconditions"))
        _add_unique(verifier_refs, part.get("verifier_refs"))
        _add_unique(modes, part.get("allowed_execution_modes"))
        # drop_* 是模型删除意图——忽略（只能加不能减）。
        for key, value in (part.get("numeric_thresholds") or {}).items():
            value = float(value)
            if key in thresholds:
                thresholds[key] = min(thresholds[key], value)  # 更严者胜
            else:
                thresholds[key] = value
        if part.get("resource_provenance_required"):
            provenance = True
        if part.get("required_receipt"):
            receipt = str(part["required_receipt"])
    return AcceptanceSpecV2(
        spec_id=new_id("acc"),
        task_id=task_id,
        revision=revision,
        required_artifacts=artifacts,
        evidence_classes=evidence,
        resource_provenance_required=provenance,
        numeric_thresholds=thresholds,
        visual_requirements=visual,
        postconditions=postconditions,
        allowed_execution_modes=modes or ["SIMULATION"],
        required_receipt=receipt,
        verifier_refs=verifier_refs,
        sources={k: sorted(v.keys()) for k, v in parts.items() if v},
    )


__all__ = ["compile_acceptance"]
