"""CapabilitySnapshot 构建（PR-N5D）——Registry → 当前回合精确工具面。

规则（测试钉住）：
- direct 曝光：effect ∈ READ_ONLY / PURE_COMPUTE / SIMULATED_EFFECT /
  NETWORK_EFFECT；
- PHYSICAL_EFFECT → propose_only（工具名 propose_<slug>——原始
  executor 永不直接暴露）；
- 其他 effect（WORKSPACE_WRITE/HOST_PROCESS/SHADOW_PROPOSAL 等）→
  excluded EFFECT_NOT_EXPOSABLE（fail closed）；
- 隔离 → excluded CAPABILITY_QUARANTINED；
- 工具名：capability_id "." → "__"（wire 约定）；碰撞追加 digest
  后缀；生成后做合法性/唯一性断言。
"""

from __future__ import annotations

import re

from rosclaw.contracts.agent.capability import EffectClassV1
from rosclaw.contracts.agent.capability_snapshot import (
    CapabilitySnapshotV1,
    SnapshotActiveToolV1,
    SnapshotExcludedV1,
)

_DIRECT_EFFECTS = frozenset({
    EffectClassV1.READ_ONLY,
    EffectClassV1.PURE_COMPUTE,
    EffectClassV1.SIMULATED_EFFECT,
    EffectClassV1.NETWORK_EFFECT,
})

_TOOL_NAME_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


def _slug(capability_id: str) -> str:
    slug = capability_id.replace(".", "__")
    if not _TOOL_NAME_RE.fullmatch(slug):
        # 特殊字符 → 合法化（非常规字符替换为 -）
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)
        if not slug[0].isalpha():
            slug = "cap-" + slug
    return slug


def build_capability_snapshot(
    catalog, *, body_id: str, mode: str, generation: int | None = None
) -> CapabilitySnapshotV1:
    """当前 registry 状态 → 快照（digest 覆盖全部可见内容）。"""
    active: list[SnapshotActiveToolV1] = []
    excluded: list[SnapshotExcludedV1] = []
    used_names: dict[str, str] = {}  # tool_name -> capability_id
    for cap in catalog.list_capabilities():
        cid = cap.capability_id
        reason = catalog.quarantine_reason(cid)
        if reason is not None:
            # 已知机器原因码直传（如 QUARANTINED_UNCLASSIFIED /
            # REF_CONFORMANCE_FAILED），其余统一 CAPABILITY_QUARANTINED。
            code = reason.split(":", 1)[0].strip()
            if code not in ("QUARANTINED_UNCLASSIFIED", "REF_CONFORMANCE_FAILED"):
                code = "CAPABILITY_QUARANTINED"
            excluded.append(SnapshotExcludedV1(
                capability_id=cid, reason=code,
            ))
            continue
        effect = cap.effect.class_
        if effect in _DIRECT_EFFECTS:
            exposure, prefix = "direct", ""
        elif effect is EffectClassV1.PHYSICAL_EFFECT:
            exposure, prefix = "propose_only", "propose_"
        else:
            excluded.append(SnapshotExcludedV1(
                capability_id=cid, reason="EFFECT_NOT_EXPOSABLE",
            ))
            continue
        name = prefix + _slug(cid)
        if name in used_names and used_names[name] != cid:
            # 碰撞：追加 capability digest 后缀（稳定）。
            name = f"{name}-{cap.canonical_hash().removeprefix('capability:')[:8]}"
        used_names[name] = cid
        active.append(SnapshotActiveToolV1(
            tool_name=name,
            capability_id=cid,
            exposure=exposure,
            effect_class=effect.value,
            description=cap.description[:400],
            input_schema=dict(cap.input_schema),
            output_schema=dict(cap.output_schema),
        ))
    # WP-2：引用连通性——消费者 accepts_refs 的每个 kind 必须在
    # active 里有 produces_refs 生产者；否则 excluded（模型看不到
    # 天然连不上的工具）。
    produced_kinds = set()
    for entry in active:
        for ref in _capability_produces(catalog, entry.capability_id):
            produced_kinds.add(ref.get("kind", ""))
    still_active: list[SnapshotActiveToolV1] = []
    for entry in active:
        needs = _capability_accepts(catalog, entry.capability_id)
        missing = [r.get("kind", "") for r in needs
                   if r.get("kind", "") not in produced_kinds]
        if missing:
            excluded.append(SnapshotExcludedV1(
                capability_id=entry.capability_id,
                reason="REF_NOT_CONNECTABLE",
            ))
        else:
            still_active.append(entry)
    active = still_active
    snap = CapabilitySnapshotV1(
        generation=generation if generation is not None else catalog.generation,
        body_id=body_id,
        mode=mode,
        active=active,
        excluded=excluded,
    )
    import hashlib

    from rosclaw.contracts.common import canonical_json

    snap.digest = "sha256:" + hashlib.sha256(
        canonical_json(snap.hash_payload()).encode("utf-8")
    ).hexdigest()
    return snap


def _capability_accepts(catalog, capability_id: str) -> list[dict]:
    cap = catalog.capability(capability_id)
    return list(cap.accepts_refs) if cap is not None else []


def _capability_produces(catalog, capability_id: str) -> list[dict]:
    cap = catalog.capability(capability_id)
    return list(cap.produces_refs) if cap is not None else []


__all__ = ["build_capability_snapshot"]
