"""PR-H4 红测试：Artifact/Acceptance/Verifier（总纲 v2 §12 + PR-H4）。

红测试先行——修复前必须红：
1. artifact 登记（sha256/尺寸/存在性实查）——登记才进交付列表；
2. task_finish 带登记 artifact → verifier 真跑 → SUCCEEDED +
   verifications 行 + accepted_at；
3. 零证据 finish → REPAIR_REQUIRED（不 SUCCEEDED）——无 artifact 且
   无总结文本证据不允许成功；
4. 伪造 artifact 路径（登记后删除/改写）→ 验收 FAIL（hash 失配）；
5. task_blocked → BLOCKED + 稳定原因码（恢复动作可查）；
6. 验收失败回同一 task（revision 不变、不新建 task）；
7. 终态不可逆（SUCCEEDED 后 finish 重放幂等）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(tmp_path / "k.db")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return conn, TaskKernel(conn, tmp_path)


def _task(kernel, tmp_path: Path) -> dict:
    result = kernel.bind_message(
        mission_id="m1", session_ref="s1", backend_native_id="n1",
        message_id="msg_1", text="做东西", cwd=str(tmp_path),
    )
    return result


class TestArtifactRegistry:
    def test_register_artifact_with_hash(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        produced = Path(task["workspace_path"]) / "out.txt"
        produced.write_text("real content\n")
        artifact = kernel.register_artifact(
            task_id=task["task_id"], path=str(produced),
            media_type="text/plain",
        )
        assert artifact["sha256"]
        assert artifact["size_bytes"] > 0
        row = conn.execute(
            "SELECT artifact_id, sha256 FROM artifacts WHERE task_id = ?",
            (task["task_id"],),
        ).fetchone()
        assert row is not None
        # 事件流有 artifact.created。
        events = conn.execute(
            "SELECT event_type FROM task_events WHERE task_id = ?",
            (task["task_id"],),
        ).fetchall()
        assert any(e["event_type"] == "artifact.created" for e in events)

    def test_register_missing_file_rejected(self, tmp_path: Path) -> None:
        """不存在的文件不得登记（ forged artifact 防线）。"""
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        import pytest

        with pytest.raises(ValueError, match="不存在|missing"):
            kernel.register_artifact(
                task_id=task["task_id"],
                path=str(tmp_path / "ghost.txt"),
                media_type="text/plain",
            )


class TestFinishVerifier:
    def test_finish_with_artifacts_succeeds(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        produced = Path(task["workspace_path"]) / "out.txt"
        produced.write_text("real\n")
        artifact = kernel.register_artifact(
            task_id=task["task_id"], path=str(produced),
            media_type="text/plain",
        )
        result = kernel.finish_task(
            task_id=task["task_id"], summary="完成",
            artifact_ids=[artifact["artifact_id"]],
        )
        assert result["status"] == "SUCCEEDED", result
        assert result["verification_id"]
        row = conn.execute(
            "SELECT status, checks_json FROM verifications WHERE task_id = ?",
            (task["task_id"],),
        ).fetchone()
        assert row["status"] == "PASS"
        final = kernel.get_task(task["task_id"])
        assert final["state"] == "SUCCEEDED"
        assert final["accepted_at"]

    def test_finish_zero_evidence_repair_required(self, tmp_path: Path) -> None:
        """无 artifact 且无实质总结 → REPAIR_REQUIRED（不 SUCCEEDED）。"""
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        result = kernel.finish_task(
            task_id=task["task_id"], summary="", artifact_ids=[],
        )
        assert result["status"] == "REPAIR_REQUIRED", result
        assert kernel.get_task(task["task_id"])["state"] == "RUNNING"

    def test_forged_artifact_fails_verification(self, tmp_path: Path) -> None:
        """登记后改写文件 → hash 失配 → REPAIR_REQUIRED。"""
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        produced = Path(task["workspace_path"]) / "out.txt"
        produced.write_text("real\n")
        artifact = kernel.register_artifact(
            task_id=task["task_id"], path=str(produced),
            media_type="text/plain",
        )
        produced.write_text("tampered\n")
        result = kernel.finish_task(
            task_id=task["task_id"], summary="完成",
            artifact_ids=[artifact["artifact_id"]],
        )
        assert result["status"] == "REPAIR_REQUIRED"
        assert any("hash" in f.lower() or "哈希" in f or "失配" in f
                   for f in result["failures"]), result

    def test_blocked_with_reason_code(self, tmp_path: Path) -> None:
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        kernel.block_task(
            task_id=task["task_id"], reason_code="MISSING_CAPABILITY",
            detail="需要网络访问（未授权）",
        )
        final = kernel.get_task(task["task_id"])
        assert final["state"] == "BLOCKED"
        assert "MISSING_CAPABILITY" in (final["terminal_reason"] or "")

    def test_finish_terminal_idempotent(self, tmp_path: Path) -> None:
        """SUCCEEDED 后重放 finish → 幂等（不重复验证、不覆盖）。"""
        conn, kernel = _kernel(tmp_path)
        task = _task(kernel, tmp_path)
        produced = Path(task["workspace_path"]) / "out.txt"
        produced.write_text("real\n")
        artifact = kernel.register_artifact(
            task_id=task["task_id"], path=str(produced),
            media_type="text/plain",
        )
        first = kernel.finish_task(
            task_id=task["task_id"], summary="完成",
            artifact_ids=[artifact["artifact_id"]],
        )
        assert first["status"] == "SUCCEEDED"
        replay = kernel.finish_task(
            task_id=task["task_id"], summary="篡改总结", artifact_ids=[],
        )
        assert replay["status"] == "SUCCEEDED"
        assert replay.get("already_terminal"), "终态后 finish 应幂等返回"
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM verifications WHERE task_id = ?",
            (task["task_id"],),
        ).fetchone()["c"]
        assert count == 1, "重放不得重复验证"
