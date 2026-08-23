"""任务运行目录（WP-8，0823 审计 §四.WP-8）。

项目源码不是任务垃圾场——0823 实测模型把交付 GIF 写进项目源码
树。每个 task/revision 有确定布局：

~/.rosclaw/runs/<task_id>/r<revision>/{scratch,outputs,evidence,logs}

- scratch：草稿/中间产物（不得登记为交付物）；
- outputs：交付物（register_artifact 记录 zone）；
- evidence：证据（receipt/trace 引用）；
- logs：运行日志。
"""

from __future__ import annotations

from pathlib import Path

ZONES = ("scratch", "outputs", "evidence", "logs")


def run_dir(home: Path, task_id: str, revision: int) -> Path:
    return Path(home) / "runs" / task_id / f"r{int(revision)}"


def ensure_run(home: Path, task_id: str, revision: int) -> dict[str, object]:
    """创建（幂等）并返回运行目录描述。"""
    base = run_dir(home, task_id, revision)
    zones: dict[str, str] = {}
    for zone in ZONES:
        path = base / zone
        path.mkdir(parents=True, exist_ok=True)
        zones[zone] = str(path)
    return {"run_dir": str(base), "zones": zones}


def zone_of(home: Path, task_id: str, revision: int, path: Path) -> str:
    """文件所在的运行区（不在本 task/revision 运行目录内 → ""）。"""
    base = run_dir(home, task_id, revision)
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts and rel.parts[0] in ZONES else ""


__all__ = ["ZONES", "ensure_run", "run_dir", "zone_of"]
