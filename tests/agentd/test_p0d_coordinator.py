"""P0-D 红测试（0824 总纲 §19.P0-D）：Coordinator 自动收尾与 TaskOutcomeV2。

红测试先行——coordinator/outcome 不存在时必须红。

验收（文档原文）：
- capability 完成后无需新模型调用即可登记、验证、出 outcome；
- motion pass + render fail 表达 execution success/delivery repair
  （不是整个任务 BLOCKED）；
- final summary 与 outcome 完全一致。

§8.3：同一错误指纹再次出现进入 WAITING_INPUT，不继续烧 token。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


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
        message_id="msg_1", text="生成 report.txt 并交付",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        cwd=str(home),
    )
    return str(bound["task_id"])


def _register_file(kernel, home: Path, task_id: str, name: str = "report.txt") -> dict:
    f = home / name
    f.write_text("交付内容", encoding="utf-8")
    return kernel.register_artifact(
        task_id=task_id, path=str(f), media_type="text/plain",
    )


class TestAutoFinalize:
    def test_finalize_without_model_call(self, tmp_path: Path) -> None:
        """交付物就位后 Coordinator 直接验收+终态——零模型调用。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        _register_file(kernel, tmp_path, task_id)
        coordinator = TaskCoordinator(kernel)
        outcome = coordinator.consider(task_id)
        assert outcome is not None, "Coordinator 未产出 outcome"
        assert outcome["lifecycle"] == "COMPLETED"
        assert outcome["verification"] == "PASS"
        assert outcome["delivery"] == "DELIVERED"
        assert outcome["execution"] == "SUCCEEDED"
        assert outcome["user_acceptance"] == "UNSEEN"
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", "终态未由 Coordinator 写入"

    def test_outcome_idempotent_replay(self, tmp_path: Path) -> None:
        """重复 consider 返回同一 outcome（deterministic replay——
        final summary 与 outcome 完全一致）。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        _register_file(kernel, tmp_path, task_id)
        coordinator = TaskCoordinator(kernel)
        first = coordinator.consider(task_id)
        second = coordinator.consider(task_id)
        assert json.dumps(first, sort_keys=True) == json.dumps(
            second, sort_keys=True
        ), "同一 task 的 outcome 不确定"

    def test_delivery_repair_not_blocked(self, tmp_path: Path) -> None:
        """执行成功但媒体/交付失败 = execution SUCCEEDED +
        delivery NEEDS_REPAIR——lifecycle 不关闭（不是 BLOCKED）。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        _register_file(kernel, tmp_path, task_id)

        def fake_verify(_task, _artifacts, _frozen):
            return {
                "status": "REPAIR_REQUIRED",
                "checks": 3,
                "failures": ["RENDER_RECEIPT_MISSING: 场景渲染缺 receipt"],
            }

        coordinator = TaskCoordinator(kernel, verify_runner=fake_verify)
        outcome = coordinator.consider(task_id)
        assert outcome["lifecycle"] == "ACTIVE", "媒体失败关闭了 lifecycle"
        assert outcome["execution"] == "SUCCEEDED"
        assert outcome["delivery"] == "NEEDS_REPAIR"
        assert outcome["verification"] == "FAIL"
        task = kernel.get_task(task_id)
        assert task["state"] not in ("BLOCKED", "FAILED"), (
            "delivery 失败被错误终态化"
        )

    def test_repair_fingerprint_twice_waits_input(
        self, tmp_path: Path
    ) -> None:
        """同一错误指纹再次出现 → WAITING_INPUT（不继续烧 token）。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, _conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        # 任务产出了部分交付物，但跟踪验收失败。
        _register_file(kernel, tmp_path, task_id)

        def fake_verify(_task, _artifacts, _frozen):
            return {
                "status": "REPAIR_REQUIRED",
                "checks": 1,
                "failures": ["TRACKING_EXCEEDED: 最大误差 0.4m > 阈值 0.05m"],
            }

        coordinator = TaskCoordinator(kernel, verify_runner=fake_verify)
        first = coordinator.consider(task_id)
        assert first["verification"] == "FAIL"
        directive = first.get("repair_directive") or {}
        assert directive.get("criterion"), "缺 RepairDirective.criterion"
        assert directive.get("fingerprint"), "缺错误指纹"
        second = coordinator.consider(task_id)
        task = kernel.get_task(task_id)
        assert task["state"] == "WAITING_INPUT", (
            "同指纹重复失败未进 WAITING_INPUT（继续烧 token）"
        )
        assert second["lifecycle"] == "ACTIVE"


class TestOutcomeContract:
    def test_outcome_persisted_and_six_dimensions(self, tmp_path: Path) -> None:
        """outcome 落库（task_outcomes）且六维齐全。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, conn = _kernel(tmp_path)
        task_id = _make_task(kernel, tmp_path)
        _register_file(kernel, tmp_path, task_id)
        coordinator = TaskCoordinator(kernel)
        coordinator.consider(task_id)
        row = conn.execute(
            "SELECT outcome_json FROM task_outcomes WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert row is not None, "task_outcomes 未落库"
        outcome = json.loads(str(row["outcome_json"]))
        for key in (
            "lifecycle", "execution", "verification", "delivery",
            "user_acceptance", "evidence",
        ):
            assert key in outcome, f"outcome 缺维度 {key}"
        assert outcome["evidence"]["trust"] in (
            "EXPERIMENTAL", "QUALIFIED", "TRUSTED",
        )
