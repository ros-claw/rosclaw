"""MJCF 来源解析（PR-MH2，ADR-0014，规格 §12.1）。

Maturity: experimental（ADR-0000 §4）。

只允许两类来源：task-local 相对路径与 e-URDF zoo。外部 URI、
task_root/zoo 之外的路径一律 fail closed——Agent 不应依赖宿主
绝对路径，runtime 不偷偷拉网络资源。
"""

from __future__ import annotations

from pathlib import Path


def resolve_mjcf_source(asset_ref: str | Path, *, task_root: Path | str) -> Path:
    """解析 MJCF 来源为绝对路径（zoo 名 → zoo/<name>/robot.mjcf.xml）。"""
    raw = str(asset_ref)
    if "://" in raw:
        # 外部 URI 禁止（规格 §11/§39）：不报 PATH_ESCAPE，按"找不到"拒绝。
        raise ValueError(f"MODEL_NOT_FOUND: external URI not allowed: {raw!r}")

    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    zoo_root = _default_zoo_path().resolve()
    root = Path(task_root).resolve()

    candidate = Path(raw)
    if not candidate.is_absolute():
        zoo_file = zoo_root / raw / "robot.mjcf.xml"
        if zoo_file.is_file():
            return zoo_file
        resolved = (root / candidate).resolve()
    else:
        resolved = candidate.resolve()

    inside_task = resolved == root or root in resolved.parents
    inside_zoo = resolved == zoo_root or zoo_root in resolved.parents
    if not inside_task and not inside_zoo:
        raise ValueError(f"MODEL_PATH_ESCAPE: {raw!r} resolves outside task_root/e-urdf-zoo")
    if not resolved.is_file():
        raise ValueError(f"MODEL_NOT_FOUND: {raw!r}（已查 zoo 与 task_root）")
    return resolved


def source_kind_for(path: Path, *, task_root: Path | str) -> str:
    """判断已解析路径的来源类别（eurdf | task）。"""
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    zoo_root = _default_zoo_path().resolve()
    resolved = path.resolve()
    if resolved == zoo_root or zoo_root in resolved.parents:
        return "eurdf"
    return "task"
