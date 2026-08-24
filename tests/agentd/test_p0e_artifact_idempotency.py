"""P0-E 红测试（0824 总纲 §19.P0-E）：ArtifactStore 幂等与 kernel-owned evidence。

红测试先行——幂等登记/权限隔离不存在时必须红。

验收（文档原文）：
- 同文件登记 10 次仍只有一个 ArtifactRef；
- 模型写 evidence 被 OS/服务拒绝；
- 复制 trusted receipt 不会改变 producer/trust；
- 跨 revision 拼接失败（WP-4 已覆盖——本文件钉住不回归）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _kernel(home: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, home), conn


def _make_task(kernel, home: Path) -> str:
    kernel.persist_input(
        mission_id="mis_1", session_ref="s1",
        message_id="msg_1", text="生成交付物",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        cwd=str(home),
    )
    return str(bound["task_id"])


class TestArtifactIdempotency:
    def test_same_file_ten_times_one_ref(self, tmp_path: Path) -> None:
        """同文件登记 10 次仍只有一个 ArtifactRef（幂等 upsert——
        内容寻址）。"""
        kernel, conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        f = tmp_path / "report.txt"
        f.write_text("交付内容", encoding="utf-8")
        refs = {
            kernel.register_artifact(
                task_id=task_id, path=str(f), media_type="text/plain",
            )["artifact_id"]
            for _ in range(10)
        }
        assert len(refs) == 1, f"同文件登记 10 次产生 {len(refs)} 个 ArtifactRef"
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM artifacts WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert int(rows["n"]) == 1, "账本里出现重复 artifact 行"

    def test_changed_content_new_ref(self, tmp_path: Path) -> None:
        """内容变化 → 新 ArtifactRef（内容寻址——不是路径寻址）。"""
        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        f = tmp_path / "report.txt"
        f.write_text("v1", encoding="utf-8")
        first = kernel.register_artifact(
            task_id=task_id, path=str(f), media_type="text/plain",
        )["artifact_id"]
        f.write_text("v2-changed", encoding="utf-8")
        second = kernel.register_artifact(
            task_id=task_id, path=str(f), media_type="text/plain",
        )["artifact_id"]
        assert first != second, "内容变了引用没变——不是内容寻址"


class TestKernelOwnedEvidence:
    def test_model_register_under_evidence_rejected(
        self, tmp_path: Path
    ) -> None:
        """模型路径（producer=model:*）登记 evidence/ 区文件被拒——
        evidence 是 kernel-only（模型不能自己写 verify_*.json 冒充
        受信证据）。"""
        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        evidence = tmp_path / "runs" / task_id / "r1" / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        fake_receipt = evidence / "verify_fake.json"
        fake_receipt.write_text('{"verdict": "PASS"}', encoding="utf-8")
        with pytest.raises(ValueError, match="EVIDENCE_KERNEL_ONLY"):
            kernel.register_artifact(
                task_id=task_id, path=str(fake_receipt),
                media_type="application/json", producer="model:tool",
            )

    def test_kernel_pipeline_evidence_allowed(self, tmp_path: Path) -> None:
        """受信管道（producer=kernel:*）登记 evidence/ 正常——
        权限是对生产者身份的限制，不是对目录的封锁。"""
        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        evidence = tmp_path / "runs" / task_id / "r1" / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        receipt = evidence / "verify_rollout.json"
        receipt.write_text('{"verdict": "PASS"}', encoding="utf-8")
        result = kernel.register_artifact(
            task_id=task_id, path=str(receipt),
            media_type="application/json", producer="kernel:trajectory_pipeline",
        )
        assert result["artifact_id"]

    def test_copied_receipt_keeps_producer_identity(
        self, tmp_path: Path
    ) -> None:
        """复制 trusted receipt 再登记：producer/trust 不变（身份来自
        登记路径/进程，不接受文件内容自述）。"""
        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        outputs = tmp_path / "runs" / task_id / "r1" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        copied = outputs / "copied_receipt.json"
        copied.write_text(
            '{"producer": "kernel:trajectory_pipeline", "trust": "TRUSTED"}',
            encoding="utf-8",
        )
        result = kernel.register_artifact(
            task_id=task_id, path=str(copied),
            media_type="application/json", producer="model:tool",
        )
        import json

        row = _conn.execute(
            "SELECT producer, metadata_json FROM artifacts WHERE artifact_id = ?",
            (result["artifact_id"],),
        ).fetchone()
        meta = json.loads(str(row["metadata_json"]))
        assert row["producer"] == "model:tool", "文件内容自述的生产者被接受"
        assert meta.get("evidence_tier") == "EXPERIMENTAL", (
            "复制的 receipt 被升级为受信"
        )
