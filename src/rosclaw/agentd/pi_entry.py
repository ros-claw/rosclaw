"""Pi 侧产物定位（node runtime + 各包 dist 入口）。

从 cli.py 提取（十审 W1）：pi_managed worker adapter 也需要定位
rosclaw-agent dist 入口，不能经 cli（循环 import）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class JsRuntimeBootstrapError(RuntimeError):
    """JS 运行时 bootstrap 失败（JS_RUNTIME_BOOTSTRAP_FAILED）——
    带原因与手动命令，不静默降级。"""


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


def _wheel_package_root() -> Path:
    """已安装 rosclaw 包根（site-packages/rosclaw）。"""
    return Path(__file__).resolve().parents[1]


def _wheel_embedded_entry(pkg: str, *, package_root: Path | None = None) -> str | None:
    """W11 §15.1：wheel 内嵌 JS（rosclaw/js_stage/<pkg>/dist/src/
    main.js——共同 staging 产物，与离线 tar 同一构建输入）。"""
    root = package_root if package_root is not None else _wheel_package_root()
    entry = root / "js_stage" / pkg / "dist" / "src" / "main.js"
    return str(entry) if entry.exists() else None


# ---------------------------------------------------------------------------
# G-1a（0916 三审 B-1 冒烟实证）：wheel JS 运行时闭包。
#
# 实证：wheel 内嵌 js_stage 只有 dist .js 没有 node_modules（160MB
# 超 PyPI 上限，不能入 wheel）——干净 venv 安装后 chat 即死
# （Cannot find package '@earendil-works/pi-coding-agent'）。
# 修复：首跑把嵌入 package.json/lock/dist 复制到用户态运行时根，
# npm ci --omit=dev 装生产依赖（Node 模块解析沿 main.js 向上找
# node_modules——dist 与 node_modules 必须同根）。幂等靠 lock
# digest 标记；失败诚实报错带手动命令（不静默返回 broken entry）。


def _js_runtime_base() -> Path:
    override = os.environ.get("ROSCLAW_JS_RUNTIME_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".rosclaw" / "js-runtime"


def _embedded_lock_digest(pkg: str, package_root: Path) -> str:
    import hashlib

    lock = package_root / "js_stage" / pkg / "package-lock.json"
    if not lock.exists():
        return ""
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def js_runtime_state(pkg: str) -> dict:
    """诊断面（只读无副作用）：wheel 嵌入存在性 + bootstrap 完成态。

    doctor/status 用它如实报 needs_bootstrap——本地检查，不触发
    npm（联网装依赖只能发生在用户发起 chat 的时刻）。
    """
    package_root = _wheel_package_root()
    embedded = (package_root / "js_stage" / pkg / "dist" / "src" / "main.js").exists()
    root = _js_runtime_base() / pkg
    marker = root / ".lock-digest"
    bootstrapped = (
        embedded
        and marker.exists()
        and marker.read_text(encoding="utf-8").strip()
        == _embedded_lock_digest(pkg, package_root)
        and (root / "node_modules").is_dir()
        and (root / "dist" / "src" / "main.js").exists()
    )
    return {
        "embedded": embedded,
        "bootstrapped": bootstrapped,
        "needs_bootstrap": embedded and not bootstrapped,
        "runtime_root": str(root),
    }


def ensure_js_runtime(
    pkg: str,
    *,
    npm_runner=None,
    package_root: Path | None = None,
) -> Path:
    """保证 wheel 安装布局的 JS 依赖闭包可用，返回运行时根。

    幂等：lock digest + node_modules + entry 齐 → 直接复用。
    失败抛 JsRuntimeBootstrapError（原因 + 手动命令指引）。
    npm_runner：测试注入（cmd, cwd) -> CompletedProcess。
    """
    package_root = package_root or _wheel_package_root()
    stage = package_root / "js_stage" / pkg
    if not (stage / "dist" / "src" / "main.js").exists():
        raise JsRuntimeBootstrapError(
            f"JS_RUNTIME_BOOTSTRAP_FAILED: wheel 内无 {pkg} 嵌入 JS"
            f"（{stage} 不存在）——安装不完整，请重装 rosclaw"
        )
    root = _js_runtime_base() / pkg
    digest = _embedded_lock_digest(pkg, package_root)
    marker = root / ".lock-digest"
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8").strip() == digest
        and (root / "node_modules").is_dir()
        and (root / "dist" / "src" / "main.js").exists()
    ):
        return root

    import filelock

    root.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(root / ".bootstrap.lock"), timeout=300):
        # 拿锁后重查（并发 bootstrap 只跑一次）。
        if (
            marker.exists()
            and marker.read_text(encoding="utf-8").strip() == digest
            and (root / "node_modules").is_dir()
            and (root / "dist" / "src" / "main.js").exists()
        ):
            return root
        for name in ("package.json", "package-lock.json"):
            shutil.copy2(stage / name, root / name)
        # postinstall 补丁器是运行时依赖（package.json postinstall
        # 指向 patches/——缺它 npm ci 即死，G-1a 实证）。
        if (stage / "patches").is_dir():
            patches_target = root / "patches"
            if patches_target.exists():
                shutil.rmtree(patches_target)
            shutil.copytree(stage / "patches", patches_target)
        dist_target = root / "dist"
        if dist_target.exists():
            shutil.rmtree(dist_target)
        shutil.copytree(stage / "dist", dist_target)
        npm = shutil.which("npm")
        if not npm:
            raise JsRuntimeBootstrapError(
                "JS_RUNTIME_BOOTSTRAP_FAILED: 找不到 npm——安装 Node ≥22.19"
                " 后重试，或手动执行："
                f"cd {root} && npm ci --omit=dev"
            )
        runner = npm_runner or (
            lambda cmd, cwd: subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=900,
            )
        )
        result = runner(
            [npm, "ci", "--omit=dev", "--no-audit", "--no-fund"], root,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-300:]
            raise JsRuntimeBootstrapError(
                f"JS_RUNTIME_BOOTSTRAP_FAILED: npm ci 退出码 "
                f"{result.returncode}（{tail}）——可手动重试："
                f"cd {root} && npm ci --omit=dev"
            )
        marker.write_text(digest, encoding="utf-8")
    return root


def package_entry(
    pkg: str,
    env_var: str,
    *,
    bootstrap: bool = False,
    npm_runner=None,
) -> str | None:
    """pkg dist 入口解析：env → 仓库布局 → 安装布局（<root>/packages/<pkg>）
    → wheel 内嵌 js_stage（pip 安装布局，W11）。

    bootstrap=True（chat 启动路径）：wheel 分支先保证 node_modules
    闭包（首跑 npm ci——G-1a），返回 bootstrap 后的运行时入口；
    失败抛 JsRuntimeBootstrapError。默认 False（诊断面）：无副作用，
    wheel 分支直接返回嵌入入口（依赖闭包状态经 js_runtime_state 看）。
    """
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
    if bootstrap and _wheel_embedded_entry(pkg):
        runtime_root = ensure_js_runtime(pkg, npm_runner=npm_runner)
        return str(runtime_root / "dist" / "src" / "main.js")
    return _wheel_embedded_entry(pkg)


def find_pi_agent_entry(*, bootstrap: bool = False) -> tuple[str, str] | None:
    """Locate (node ≥22.19, rosclaw-agent dist entry)。None = 不可用。

    bootstrap=True（chat 启动）：wheel 布局先接通 node_modules
    闭包（首跑 npm ci；失败抛 JsRuntimeBootstrapError）。
    """
    node = find_node()
    if node is None:
        return None
    entry = package_entry("rosclaw-agent", "ROSCLAW_AGENT_ENTRY", bootstrap=bootstrap)
    if not entry or not Path(entry).exists():
        return None
    return node, entry


def find_tui_runtime(*, bootstrap: bool = False) -> tuple[str, str] | None:
    """Locate (node ≥22.19, rosclaw-tui dist entry). None = unavailable."""
    node = find_node()
    if node is None:
        return None
    entry = package_entry("rosclaw-tui", "ROSCLAW_TUI_ENTRY", bootstrap=bootstrap)
    if not entry or not Path(entry).exists():
        return None
    return node, entry
