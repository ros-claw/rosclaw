"""inspect CLI（PR-N3，N 总纲 §4.3）：用户与 Agent 都能用的统一自检。

rosclaw inspect self/robot/capability/asset [--json]
Agent 侧映射为 rosclaw_inspect 工具（bridge → 本模块查询函数）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rosclaw.cognition.index.builder import build_index
from rosclaw.cognition.index.query import health, robot_chain, search


def _product_root() -> Path:
    """产品根：本文件上溯 4 级（src/rosclaw/cognition/inspect_cli.py → 根）。"""
    return Path(__file__).resolve().parents[3]


def _index_path(rosclaw_home: Path | None) -> Path:
    home = rosclaw_home or Path.home() / ".rosclaw"
    return home / "index" / "ecosystem.db"


def ensure_index(rosclaw_home: Path | None = None) -> Path:
    """索引存在且新鲜则直接用；否则（缺/损坏/陈旧）重建。"""
    index_path = _index_path(rosclaw_home)
    h = health(index_path)
    if h["ok"] and not h["stale"]:
        return index_path
    return build_index(index_path.parent, _product_root())


def inspect_self(rosclaw_home: Path | None = None,
                 index_path: Path | None = None) -> dict:
    """统一自检：版本/包根/根目录/索引健康/索引摘要。"""
    from rosclaw import __version__

    idx = index_path or ensure_index(rosclaw_home)
    h = health(idx)
    return {
        "schema_version": "rosclaw.inspect_self.v1",
        "version": str(__version__),
        "product_root": str(_product_root()),
        "package_root": str(_product_root() / "src" / "rosclaw"),
        "rosclaw_home": str(rosclaw_home or Path.home() / ".rosclaw"),
        "index": h,
    }


def dispatch_inspect_argv(argv: list[str]) -> int | None:
    """rosclaw inspect <kind> [query] [--json]；未命中返回 None。"""
    if not argv or argv[0] != "inspect":
        return None
    args = argv[1:]
    json_out = "--json" in args
    args = [a for a in args if a != "--json"]
    kind = args[0] if args else "self"
    query = args[1] if len(args) > 1 else ""

    if kind == "self":
        info = inspect_self()
        if json_out:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(f"ROSClaw {info['version']}")
            print(f"  product_root: {info['product_root']}")
            print(f"  index: {info['index']['entity_count']} 实体"
                  f"（{'健康' if info['index']['ok'] and not info['index']['stale'] else '需重建'}）")
        return 0

    if kind == "simulation-render":
        # WP-3：渲染后端探测诊断（EGL→OSMesa→Xvfb 子进程隔离）。
        from rosclaw.agentd.sim_render import probe_render_backend

        backend, detail = probe_render_backend()
        info = {
            "backend": backend,
            "probe": detail,
            "ok": backend is not None,
        }
        if json_out:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            if backend:
                print(f"渲染后端: {backend}")
            else:
                print("渲染后端不可用（EGL/OSMesa/Xvfb 全部失败）:")
            for name, msg in detail.items():
                print(f"  {name}: {msg}")
        return 0 if backend else 1

    idx = ensure_index()
    if kind == "robot":
        if not query:
            print("用法: rosclaw inspect robot <robot_id> [--json]", file=sys.stderr)
            return 2
        chain = robot_chain(idx, query, product_root=_product_root())
        if chain is None:
            print(f"未知机器人 {query!r}（索引无权威链）", file=sys.stderr)
            return 1
        if json_out:
            print(json.dumps(chain, ensure_ascii=False, indent=2))
        else:
            for key, value in chain.items():
                print(f"  {key}: {value}")
        return 0
    if kind == "asset":
        # PR-N4：ResourceManifestV1（production resolver——fixture 永不
        # 出现）。
        from rosclaw.cognition.resolver import resolve_resource

        manifest = resolve_resource(
            "robot", query, product_root=_product_root()
        )
        if manifest is None:
            print(f"无权威资源 {query!r}（production resolver 无结果——"
                  "不降级到 fixture/猜测）", file=sys.stderr)
            return 1
        if json_out:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            print(f"{manifest['resource_id']} [{manifest['quality']}] "
                  f"source={manifest['source']}")
            for key, path in manifest["paths"].items():
                print(f"  {key}: {path}")
        return 0
    if kind == "capability":
        hits = search(idx, query or kind, limit=20)
        if json_out:
            print(json.dumps({"hits": hits}, ensure_ascii=False, indent=2))
        else:
            for hit in hits:
                print(f"  [{hit['kind']}] {hit['name']}  {hit.get('path', '')}")
        return 0

    print(f"未知 inspect 主题 {kind!r}（self/robot/capability/asset）", file=sys.stderr)
    return 2
