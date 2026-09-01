"""0901 体验探讨 P0-2 红测试：artifact 生成即投影（不等 PASS）。

0901 实证：任务 FAIL+PARTIAL 时 Presenter 给了 3 个 Artifact ID，
但 Agent 查 outputs/ 目录是空的——P0-4 的投影只在 Coordinator
PASS 分支执行。正确逻辑（文档 §三）：每个有效 artifact 一生成立即
投影到任务 outputs/，最终 Outcome 只判 FULL/PARTIAL。

闭环断言：
1. register_artifact 成功即投影——不调 coordinator、不过 PASS，
   outputs/ 就有同内容文件；
2. 投影失败不翻转登记（登记成功 + 投影 DEGRADED 日志，不抛错）；
3. coordinator 的 PASS 分支投影保留为兜底（幂等——已投影不重复）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _kernel_on_disk(home: Path):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, home), conn


class TestProjectOnRegister:
    def test_register_projects_immediately(self, tmp_path: Path) -> None:
        """登记即投影：不等 coordinator PASS。"""
        from rosclaw.task_kernel.run_store import run_dir

        kernel, _conn = _kernel_on_disk(tmp_path)
        kernel.persist_input(
            mission_id="m1", session_ref="s1",
            message_id="msg_1", text="画一个五角星",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        payload = tmp_path / "traces" / "t1"
        payload.mkdir(parents=True)
        gif = payload / "scene.gif"
        gif.write_bytes(b"GIF89a" + b"\x00" * 100)
        record = kernel.register_artifact(
            task_id=task_id, path=str(gif), media_type="image/gif",
            producer="kernel:test",
        )
        assert record.get("artifact_id")
        projected = (
            run_dir(tmp_path, task_id, 1) / "outputs" / gif.name
        )
        assert projected.exists(), (
            f"登记后 outputs/ 无投影（0901 实证：PARTIAL 任务 outputs 空）："
            f"{projected}"
        )
        assert hashlib.sha256(projected.read_bytes()).hexdigest() == hashlib.sha256(
            gif.read_bytes()
        ).hexdigest()

    def test_projection_failure_does_not_break_registration(
        self, tmp_path: Path
    ) -> None:
        """投影失败（outputs 区被破坏）→ 登记仍成功（投影是视图，
        不是交付真相）。"""
        from rosclaw.task_kernel.run_store import run_dir

        kernel, _conn = _kernel_on_disk(tmp_path)
        kernel.persist_input(
            mission_id="m1", session_ref="s1",
            message_id="msg_1", text="画一个五角星",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        # 破坏 outputs 区（先删目录再占位为普通文件）。
        import shutil as _shutil

        outputs_parent = run_dir(tmp_path, task_id, 1)
        _shutil.rmtree(outputs_parent / "outputs", ignore_errors=True)
        (outputs_parent / "outputs").write_text("sabotaged", encoding="utf-8")
        payload = tmp_path / "x.bin"
        payload.write_bytes(b"x" * 16)
        record = kernel.register_artifact(
            task_id=task_id, path=str(payload), media_type="application/octet-stream",
            producer="kernel:test",
        )
        assert record.get("artifact_id"), "投影失败竟阻断登记"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
