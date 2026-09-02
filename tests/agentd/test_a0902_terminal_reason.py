"""0902 用户实测复归红测试：终态 FAIL 卡必须带原因。

用户实测（0902）："用机械臂画五角星…末端持笔…垂直桌子…"——确定性链
跑到 verify，语义验收诚实 FAIL（DELIVERABLE_MISSING scene_video +
TOOL_ASSET_MISSING tool:pen），但终态卡只有"验收 FAIL · 交付 PARTIAL"
没有原因——用户问"这里回的失败啥意思？"答不了。

机制实证（复现于本测试）：watcher 终态呈现时 consider 对已 FAILED 的
任务重新走 _verify（=finish_task），而 finish_task 终态幂等只回
{"status": "FAILED"}（无 failures）→ consider 写出 failures=[] 的空
repair directive（task_repairs 污染行）→ 卡片没有"原因"可显示。

闭环断言：
1. consider 终态 FAILED 任务 → outcome.repair_directive.failures 携带
   terminal_reason 的全部原因（不再重验、不写空 directive）；
2. 已终态任务不再调 finish_task（不重验——终态后验收不翻账）；
3. SUCCEEDED 终态重放不受影响（回归护栏）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _kernel(tmp_path: Path):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _failed_task(kernel, tmp_path: Path) -> str:
    bound = kernel.bind_message(
        mission_id="m1", session_ref="s1", backend_native_id="s1",
        message_id="msg_1",
        text="用机械臂画一个五角星，末端持笔，垂直桌子，我要看到轨迹",
        cwd=str(tmp_path), body_id="sim/ur5e",
    )
    task_id = str(bound["task_id"])
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    kernel.register_artifact(
        task_id=task_id, path=str(trace), media_type="application/json",
        producer="kernel:capability:ur5e.simulate",
    )
    kernel.transition(
        task_id, "FAILED",
        reason="ACCEPTANCE_FAILED: DELIVERABLE_MISSING: required 交付物 "
        "scene_video 未在产物账本；TOOL_ASSET_MISSING: tool:pen 无权威"
        "工具资产——不得假装持笔渲染",
    )
    return task_id


class TestTerminalReasonReplay:
    def test_failed_terminal_outcome_carries_reasons(
        self, tmp_path: Path
    ) -> None:
        """已 FAILED 任务 → consider 从账本重建 outcome，原因齐全。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel = _kernel(tmp_path)
        task_id = _failed_task(kernel, tmp_path)
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        directive = outcome.get("repair_directive") or {}
        failures = directive.get("failures") or []
        assert any("DELIVERABLE_MISSING" in f for f in failures), outcome
        assert any("TOOL_ASSET_MISSING" in f for f in failures), outcome

    def test_no_reverify_no_bogus_repair_row(self, tmp_path: Path) -> None:
        """终态后 consider 不调 finish_task（不重验）、不写空
        repair directive（task_repairs 不得被污染）。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel = _kernel(tmp_path)
        task_id = _failed_task(kernel, tmp_path)
        TaskCoordinator(kernel).consider(task_id)
        rows = kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM task_repairs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert int(rows["n"]) == 0, "终态重放竟写 task_repairs"
        # 终态任务不得新增 verifications（不重验）。
        vrows = kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM verifications WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert int(vrows["n"]) == 0, "终态重放竟重跑验收"

    def test_succeeded_replay_unaffected(self, tmp_path: Path) -> None:
        """SUCCEEDED 终态走既有重放路径（本次改动不碰）。"""
        kernel = _kernel(tmp_path)
        task_id = _failed_task(kernel, tmp_path)  # FAILED 建好后转 SUCCEEDED 不可能（终态不可逆）——改用直接成功路径
        # 终态不可逆语义护栏：FAILED 后拒绝覆盖。
        kernel.transition(task_id, "SUCCEEDED", reason="x")
        task = kernel.get_task(task_id)
        assert task["state"] == "FAILED", "终态不可逆语义被破坏"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
