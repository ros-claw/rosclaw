"""安装一致性诊断（P0-5，0823 审计 §三.P0-5）。

0823 事故：报告声明的能力在产品里 OUTPUT_SCHEMA_MISSING——实现
与部署漂移而不可察觉。`rosclaw version --diagnostic --json` 报告
安装的真实构成：

- rosclaw_version / wheel_commit（build stamp → git HEAD → unknown，
  逐级诚实降级）；
- ts_dist：解析到的 rosclaw-agent dist 入口 + build stamp commit +
  内容 digest（没解析到 → unavailable，不编造）；
- extension_digest：dist/src/extension 子树内容 digest；
- kit_digest：第一方 kits（sim/kits/*.json）内容 digest；
- migration_revision：bundled migrations 最大代数；
- eurdf_generation：e-urdf-zoo 内容 digest（无代数标记时内容
  寻址就是代数）；
- mixed_build：wheel 与 TS dist 的 commit 都已知且不同 →
  INSTALLATION_MIXED_BUILD（chat 启动阻断）；任一侧 unknown
  不谎报混合。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

_STAMP_NAME = "_build_stamp.json"


def _pkg_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    # src/rosclaw → parents[1] = 仓库根（src/ 的父级）。
    return _pkg_root().parents[1]


def _read_stamp(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _git_head(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def wheel_commit() -> str:
    """wheel 构建 commit：stamp（CI 写入）→ git HEAD → unknown。"""
    stamp = _read_stamp(_pkg_root() / _STAMP_NAME)
    if stamp.get("commit"):
        return str(stamp["commit"])
    head = _git_head(_repo_root())
    return head or "unknown"


def _tree_digest(root: Path, *, suffixes: tuple[str, ...] = ()) -> str:
    """目录内容寻址 digest（文件名+内容；不存在 → ""）。"""
    if not root.is_dir():
        return ""
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def _ts_dist() -> dict[str, Any]:
    """rosclaw-agent dist：entry（与 pi_entry 同一解析顺序）+
    stamp commit + dist 内容 digest。"""
    entry = os.environ.get("ROSCLAW_AGENT_ENTRY", "")
    dist_dir: Path | None = None
    if entry:
        dist_dir = Path(entry).resolve().parents[1]
    else:
        repo_entry = (
            _repo_root() / "packages" / "rosclaw-agent" / "dist" / "src" / "main.js"
        )
        if repo_entry.exists():
            entry = str(repo_entry)
            dist_dir = repo_entry.parents[1]
    if dist_dir is None or not dist_dir.is_dir():
        return {"entry": entry or "unavailable", "commit": "unknown",
                "dist_digest": "unavailable"}
    stamp = _read_stamp(dist_dir / "build-stamp.json")
    commit = str(stamp.get("commit") or "") or _git_head(_repo_root()) or "unknown"
    return {
        "entry": entry,
        "commit": commit,
        "dist_digest": _tree_digest(dist_dir, suffixes=(".js", ".json")),
    }


def extension_digest() -> str:
    """内联扩展子树 digest（dist/src/extension）。"""
    ts = _ts_dist()
    entry = str(ts.get("entry", ""))
    if entry in ("", "unavailable"):
        return "unavailable"
    ext_dir = Path(entry).resolve().parent / "extension"
    return _tree_digest(ext_dir, suffixes=(".js",)) or "unavailable"


def kit_digest() -> str:
    """第一方 kits（sim/kits/*.json）内容 digest。"""
    kits_dir = _pkg_root() / "sim" / "kits"
    return _tree_digest(kits_dir, suffixes=(".json",)) or "unavailable"


def migration_revision() -> int:
    """bundled migrations 最大代数（0 = 无）。"""
    mig_dir = _pkg_root() / "storage" / "migrations"
    revision = 0
    for path in mig_dir.glob("*.sql"):
        try:
            revision = max(revision, int(path.name.split("_", 1)[0]))
        except ValueError:
            continue
    return revision


def eurdf_generation() -> str:
    """e-URDF 内容代数：zoo 无显式代数标记——内容寻址即代数。"""
    zoo = _repo_root() / "e-urdf-zoo"
    if not zoo.is_dir():
        # 安装布局：pyproject 映射 e-urdf-zoo → rosclaw/eurdf_zoo_data
        zoo = _pkg_root() / "eurdf_zoo_data"
    return _tree_digest(zoo) or "unknown"


def collect_diagnostics(*, home: Path | None = None) -> dict[str, Any]:
    """安装一致性快照（home 保留给运行时项——当前全部为安装态）。"""
    del home  # 运行时项（capability snapshot digest）由 agentd 面提供
    from rosclaw import __version__

    ts = _ts_dist()
    diag: dict[str, Any] = {
        "rosclaw_version": __version__,
        "wheel_commit": wheel_commit(),
        "ts_dist": ts,
        "extension_digest": extension_digest(),
        "kit_digest": kit_digest(),
        "migration_revision": migration_revision(),
        "eurdf_generation": eurdf_generation(),
    }
    reason = mixed_build_reason(diag)
    diag["mixed_build"] = reason is not None
    if reason:
        diag["mixed_build_reason"] = reason
    return diag


def mixed_build_reason(diag: dict[str, Any]) -> str | None:
    """两侧 commit 都已知且不同 → 混合构建（诚实：unknown 不报）。"""
    wheel = str(diag.get("wheel_commit", "unknown"))
    ts = str((diag.get("ts_dist") or {}).get("commit", "unknown"))
    if wheel == "unknown" or ts == "unknown" or wheel == ts:
        return None
    return (
        f"INSTALLATION_MIXED_BUILD: wheel 构建自 {wheel[:12]} 而 TS dist "
        f"构建自 {ts[:12]}——实现与部署不一致，能力声明不可信"
    )


def is_stamped_install() -> bool:
    """是否带构建戳的安装（CI wheel）。源码 checkout 无戳——
    chat 门禁只约束安装产物（开发树 python/TS 各自演进是常态，
    mixed 状态在 version --diagnostic 可见但不阻断）。"""
    return (_pkg_root() / _STAMP_NAME).exists()


def assert_installation_coherent(diag: dict[str, Any]) -> None:
    """chat 启动门禁：混合构建直接阻断（SystemExit）。"""
    reason = mixed_build_reason(diag)
    if reason is not None:
        raise SystemExit(
            f"{reason}\n请重新安装一致构建（rosclaw version --diagnostic "
            "查看构成）。"
        )


def cmd_version(*, diagnostic: bool, as_json: bool,
                home: Path | None = None) -> int:
    """`rosclaw version [--diagnostic [--json]]` 入口。"""
    if not diagnostic:
        from rosclaw import __version__

        print(__version__ if not as_json else json.dumps(
            {"rosclaw_version": __version__}
        ))
        return 0
    diag = collect_diagnostics(home=home)
    if as_json:
        print(json.dumps(diag, ensure_ascii=False, indent=1))
    else:
        print(f"rosclaw {diag['rosclaw_version']}")
        print(f"  wheel_commit:      {diag['wheel_commit']}")
        ts = diag["ts_dist"]
        print(f"  ts_dist:           {ts['entry']}")
        print(f"  ts_dist_commit:    {ts['commit']}")
        print(f"  ts_dist_digest:    {ts['dist_digest']}")
        print(f"  extension_digest:  {diag['extension_digest']}")
        print(f"  kit_digest:        {diag['kit_digest']}")
        print(f"  migration:         r{diag['migration_revision']}")
        print(f"  eurdf_generation:  {diag['eurdf_generation']}")
        if diag["mixed_build"]:
            print(f"  ⚠ {diag['mixed_build_reason']}")
    return 0


__all__ = [
    "assert_installation_coherent",
    "is_stamped_install",
    "cmd_version",
    "collect_diagnostics",
    "eurdf_generation",
    "extension_digest",
    "kit_digest",
    "migration_revision",
    "mixed_build_reason",
    "wheel_commit",
]
