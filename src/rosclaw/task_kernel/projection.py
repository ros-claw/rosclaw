"""Non-authoritative outputs view: verified, independent copies, never hardlinks.

Projection failure degrades the view, not registration. Ledger paths remain
unchanged; only current-revision non-evidence payloads are projected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rosclaw.task_kernel.run_store import ensure_run, zone_of

if TYPE_CHECKING:
    from rosclaw.task_kernel.service import TaskKernel

_LOG = logging.getLogger("rosclaw.projection")


def project_deliverables(kernel: TaskKernel, task_id: str) -> str:
    """Project verified current-revision bytes; return OK or DEGRADED."""
    from rosclaw.task_kernel.service import read_artifact_payload

    task = kernel.get_task(task_id)
    if task is None:
        return "DEGRADED"
    revision = int(task["active_revision"])
    try:
        run = ensure_run(kernel._home, task_id, revision)
        outputs = Path(str(run["zones"]["outputs"]))
        if not outputs.is_dir():
            raise OSError(f"outputs is not a directory: {outputs}")
        degraded = False
        rows = kernel._conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ? AND revision = ?",
            (task_id, revision),
        ).fetchall()
        for row in rows:
            src = Path(str(row["path"]))
            meta = json.loads(str(row["metadata_json"]))
            zone = meta.get("zone") or zone_of(kernel._home, task_id, revision, src)
            if zone == "evidence":
                continue
            try:
                content = read_artifact_payload(src, str(row["sha256"]), int(row["size_bytes"]))
                name = Path(meta.get("source_path") or src).name
                target = outputs / name
                # Exclusive creation also rejects dangling symlinks. Existing
                # aliases are never overwritten, even when externally modified.
                if target.exists() or target.is_symlink():
                    if (not target.is_symlink() and target.is_file()
                            and target.stat().st_ino != src.stat().st_ino
                            and target.stat().st_size == len(content)
                            and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]):
                        continue
                    target = outputs / f"{row['artifact_id']}_{name}"
                if target.exists() or target.is_symlink():
                    if (not target.is_symlink() and target.is_file()
                            and target.stat().st_ino != src.stat().st_ino
                            and target.read_bytes() == content):
                        continue
                    raise ValueError("ARTIFACT_PROJECTION_CONFLICT")
                with target.open("xb") as handle:
                    handle.write(content)
            except (OSError, ValueError) as exc:
                degraded = True
                _LOG.warning("projection failed for %s: %s", src, exc)
        return "DEGRADED" if degraded else "OK"
    except OSError as exc:
        _LOG.warning("projection degraded for %s: %s", task_id, exc)
        return "DEGRADED"


__all__ = ["project_deliverables"]
