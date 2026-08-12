"""Pi 侧产物定位（node runtime + 各包 dist 入口）。

从 cli.py 提取（十审 W1）：pi_managed worker adapter 也需要定位
rosclaw-agent dist 入口，不能经 cli（循环 import）。
"""

from __future__ import annotations

import os
from pathlib import Path


def _install_prefix_root() -> Path | None:
    """安装布局根（$PREFIX/current）：venv 位于 <root>/.venv 时返回 root。"""
    parts = Path(__file__).resolve().parts
    if ".venv" in parts:
        idx = parts.index(".venv")
        if idx > 0:
            return Path(*parts[:idx])
    return None


def _node_candidates() -> list[str]:
    """bundled node 优先（发布包免装 Node），其后系统 node。"""
    import shutil

    candidates: list[str] = []
    root = _install_prefix_root()
    if root is not None:
        bundled = root / "vendor" / "node-runtime" / "bin" / "node"
        if bundled.exists():
            candidates.append(str(bundled))
    candidates += [shutil.which("node") or "", "/usr/bin/node", "/usr/local/bin/node"]
    return candidates


def find_node() -> str | None:
    import subprocess as _sp

    for candidate in filter(None, _node_candidates()):
        try:
            out = _sp.check_output([candidate, "--version"], text=True, timeout=10).strip()
            parts = [int(p) for p in out.lstrip("v").split(".")]
            if parts >= [22, 19, 0]:
                return candidate
        except Exception:  # noqa: BLE001 - probe next candidate
            continue
    return None


def package_entry(pkg: str, env_var: str) -> str | None:
    """pkg dist 入口解析：env → 仓库布局 → 安装布局（<root>/packages/<pkg>）。"""
    entry_env = os.environ.get(env_var)
    if entry_env:
        return entry_env
    repo_entry = (
        Path(__file__).resolve().parents[3] / "packages" / pkg / "dist" / "src" / "main.js"
    )
    if repo_entry.exists():
        return str(repo_entry)
    root = _install_prefix_root()
    if root is not None:
        installed = root / "packages" / pkg / "dist" / "src" / "main.js"
        if installed.exists():
            return str(installed)
    return None


def find_pi_agent_entry() -> tuple[str, str] | None:
    """Locate (node ≥22.19, rosclaw-agent dist entry)。None = 不可用。"""
    node = find_node()
    if node is None:
        return None
    entry = package_entry("rosclaw-agent", "ROSCLAW_AGENT_ENTRY")
    if not entry or not Path(entry).exists():
        return None
    return node, entry


def find_tui_runtime() -> tuple[str, str] | None:
    """Locate (node ≥22.19, rosclaw-tui dist entry). None = unavailable."""
    node = find_node()
    if node is None:
        return None
    entry = package_entry("rosclaw-tui", "ROSCLAW_TUI_ENTRY")
    if not entry or not Path(entry).exists():
        return None
    return node, entry
