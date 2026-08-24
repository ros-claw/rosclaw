"""Canonical alias（P0-G，0824 总纲 §11.4/§19.P0-G）。

body_id（`sim/ur5e`）↔ resource_id（`robot:ur5e`）的唯一权威
映射——setup/header/inspect/verifier 全部经此换算，不再各处
手写 removeprefix/字符串拼接（漂移即不一致）。

规则：body 的实例前缀（sim/、real/ 等）剥离后加 `robot:` 资源
命名空间；未知 body 同样确定换算（不猜、不例外——调用方决定
是否 fail closed）。
"""

from __future__ import annotations

_INSTANCE_PREFIXES = ("sim/", "real/", "shadow/")


def canonical_resource_id(body_id: str) -> str:
    """body_id → 资源命名空间 id（sim/ur5e → robot:ur5e）。"""
    body = str(body_id).strip()
    for prefix in _INSTANCE_PREFIXES:
        if body.startswith(prefix):
            return f"robot:{body[len(prefix):]}"
    return f"robot:{body}"


def body_id_for_resource(resource_id: str, *, instance: str = "sim") -> str:
    """资源 id → body_id（robot:ur5e → sim/ur5e）。"""
    rid = str(resource_id).strip()
    name = rid.removeprefix("robot:")
    return f"{instance}/{name}"


__all__ = ["body_id_for_resource", "canonical_resource_id"]
