"""仿真对象引用（PR-MH1，ADR-0014，规格 §6/§11）。

ref 是**内容寻址令牌，不是路径**：``sim<kind>_<sha256[:16]>``。
路径分隔符、``..``、URI scheme 在词法层即被拒（REF_INVALID）——
路径逃逸在构造上不可能。

legacy 前缀（``model_`` / ``obs_`` / ``op_``，sim/api.py 现行体系）
只做只读兼容，供渐进迁移（规格 §11）。
"""

from __future__ import annotations

import re

REF_RE = re.compile(r"^sim(mdl|sta|trc|obs|adt|rnd|exp|art)_[0-9a-f]{16}$")
LEGACY_RE = re.compile(r"^(model|obs|op)_[0-9a-f]{16}$")

KINDS: tuple[str, ...] = (
    "simmdl",
    "simsta",
    "simtrc",
    "simobs",
    "simadt",
    "simrnd",
    "simexp",
    "simart",
)

PREFIX_TO_PARTITION: dict[str, str] = {
    "simmdl": "models",
    "simsta": "states",
    "simtrc": "traces",
    "simobs": "states",
    "simadt": "audits",
    "simrnd": "renders",
    "simexp": "experiments",
    "simart": "experiments",
}

_DIGEST64_RE = re.compile(r"^[0-9a-f]{64}$")


def make_ref(kind: str, digest_hex: str) -> str:
    """由 kind 与完整 sha256 hex 构造 ref（取前 16 hex）。"""
    if kind not in PREFIX_TO_PARTITION:
        raise ValueError(f"REF_INVALID: unknown kind {kind!r}")
    if not _DIGEST64_RE.match(digest_hex or ""):
        raise ValueError(f"REF_INVALID: digest must be 64 lowercase hex, got {digest_hex!r}")
    return f"{kind}_{digest_hex[:16]}"


def parse_ref(ref: str) -> tuple[str, str]:
    """解析 ref 为 (kind, hex16)；任何词法违规 fail closed。"""
    match = REF_RE.match(ref or "")
    if not match:
        raise ValueError(f"REF_INVALID: {ref!r}")
    return f"sim{match.group(1)}", ref.rsplit("_", 1)[1]


def partition_for(ref: str) -> str:
    """ref 前缀 ↔ 分区一致性映射。"""
    kind, _ = parse_ref(ref)
    return PREFIX_TO_PARTITION[kind]


def is_legacy_ref(ref: str) -> bool:
    return bool(LEGACY_RE.match(ref or ""))


def classify(ref: str) -> str:
    """分流：'sim' | 'legacy' | 'invalid'。"""
    if REF_RE.match(ref or ""):
        return "sim"
    if LEGACY_RE.match(ref or ""):
        return "legacy"
    return "invalid"
