"""交付投影视图（0827 体验审计 P0-4）。

outputs/ 是 ArtifactStore 的**投影视图**，不是第二个交付真相：
登记产物由内核自动投影到运行 outputs/ 区（hardlink 优先、copy
兜底）——内核内部文件操作，不经模型 Shell、不依赖 bwrap。

纪律：
- 投影失败绝不翻转交付判定（DELIVERED + workspace_projection
  DEGRADED——账本/open_command 仍是权威）；
- 账本路径不被投影改写（内容寻址的 ArtifactStore 是唯一真相）；
- 幂等：同 sha256 投影不重复。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rosclaw.task_kernel.run_store import ensure_run, zone_of

if TYPE_CHECKING:
    from rosclaw.task_kernel.service import TaskKernel

_LOG = logging.getLogger("rosclaw.projection")


def project_deliverables(kernel: TaskKernel, task_id: str) -> str:
    """把任务登记产物投影到运行 outputs/ 区。返回 OK/DEGRADED。"""
    task = kernel.get_task(task_id)
    if task is None:
        return "DEGRADED"
    revision = int(task["active_revision"])
    try:
        run = ensure_run(kernel._home, task_id, revision)
        outputs = Path(str(run["zones"]["outputs"]))
        if not outputs.is_dir():
            raise OSError(f"outputs 区不是目录：{outputs}")
        degraded = False
        rows = kernel._conn.execute(
            "SELECT artifact_id, path, sha256 FROM artifacts "
            "WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        for row in rows:
            src = Path(str(row["path"]))
            if zone_of(kernel._home, task_id, revision, src) == "outputs":
                continue  # 已在交付区——无需投影
            if not src.exists():
                degraded = True
                _LOG.warning("projection source missing: %s", src)
                continue
            target = outputs / src.name
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() == str(
                    row["sha256"]
                ):
                    continue  # 幂等：同内容已投影
                target = outputs / f"{row['artifact_id']}_{src.name}"
            try:
                os.link(src, target)
            except OSError:
                try:
                    shutil.copy2(src, target)
                except OSError as exc:
                    degraded = True
                    _LOG.warning(
                        "projection failed: %s → %s: %s", src, target, exc,
                    )
        return "DEGRADED" if degraded else "OK"
    except OSError as exc:
        _LOG.warning("projection degraded for %s: %s", task_id, exc)
        return "DEGRADED"


__all__ = ["project_deliverables"]
