"""0901 体验探讨 P0-1 红测试：artifact CLI 必须经 `rosclaw` 入口可达。

0901 实证：`rosclaw artifact open art_xxx` 打顶层帮助——artifact
list/open/export 只在 rosclaw-agentd 可达，entrypoint dispatch 链
没有 artifact 分支。R0-4 的测试只测 handler 函数（假绿），从没在
`rosclaw` 入口跑过 subprocess。

硬 Gate B（文档 §十二）：真实入口 subprocess 下——
- `rosclaw artifact list` 不打顶层帮助（不回落 legacy parser）；
- `rosclaw artifact path <id>` 输出绝对路径（SSH/脚本可用）；
- `rosclaw artifact open <id>` 不打帮助（无显示环境打路径）；
- `rosclaw artifact export <id> <dest>` 复制且 sha256 一致；
- `rosclaw artifact show <id>` 显示类型/大小/任务/血缘。
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


def _make_artifact(home: Path) -> tuple[str, Path, str]:
    """在 tmp home 建一个真实 artifact（落盘账本——CLI subprocess
    读的是 home/agentd/missions.db）。"""
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    db_path = home / "agentd" / "missions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    kernel = TaskKernel(conn, home)
    kernel.persist_input(
        mission_id="mis_1", session_ref="s1",
        message_id="msg_1", text="画一个五角星",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        cwd=str(home), body_id="sim/ur5e",
    )
    task_id = str(bound["task_id"])
    payload = home / "demo.bin"
    payload.write_bytes(b"rosclaw-0901-artifact-payload" * 8)
    record = kernel.register_artifact(
        task_id=task_id, path=str(payload), media_type="application/octet-stream",
        producer="kernel:test",
    )
    conn.commit()
    conn.close()
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    return str(record["artifact_id"]), payload, digest


def _run(home: Path, *argv: str) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        [sys.executable, "-m", "rosclaw.entrypoint", *argv],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "ROSCLAW_HOME": str(home)},
    )


class TestArtifactCliReachable:
    def test_list_not_top_level_help(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path)
        result = _run(tmp_path, "artifact", "list")
        assert "usage: rosclaw" not in result.stdout, (
            f"artifact list 回落到顶层帮助（dispatch 缺失）：{result.stdout[:200]}"
        )
        assert "demo.bin" in result.stdout, result.stdout

    def test_path_prints_absolute_path(self, tmp_path: Path) -> None:
        artifact_id, payload, _ = _make_artifact(tmp_path)
        result = _run(tmp_path, "artifact", "path", artifact_id)
        assert result.returncode == 0, result.stderr
        assert str(payload) in result.stdout, result.stdout

    def test_open_never_prints_help(self, tmp_path: Path) -> None:
        artifact_id, payload, _ = _make_artifact(tmp_path)
        result = _run(tmp_path, "artifact", "open", artifact_id)
        assert "usage: rosclaw" not in result.stdout, result.stdout[:200]
        # 无显示环境：打路径（SSH 一等）。
        assert str(payload) in result.stdout, result.stdout

    def test_export_digest_consistent(self, tmp_path: Path) -> None:
        artifact_id, _payload, digest = _make_artifact(tmp_path)
        dest = tmp_path / "export" / "out.bin"
        result = _run(tmp_path, "artifact", "export", artifact_id, str(dest))
        assert result.returncode == 0, result.stderr
        assert dest.exists()
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest

    def test_show_details(self, tmp_path: Path) -> None:
        artifact_id, payload, digest = _make_artifact(tmp_path)
        result = _run(tmp_path, "artifact", "show", artifact_id)
        assert result.returncode == 0, result.stderr
        assert artifact_id in result.stdout
        assert f"sha256:{digest}" in result.stdout, result.stdout
        assert str(payload) in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
