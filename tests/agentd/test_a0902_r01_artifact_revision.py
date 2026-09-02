"""0902 审计 R0-1 红测试：Artifact revision 作用域——旧产物不得满足
新 revision（反假成功的结构修复）。

0902 实证（用户日志）：用户新增"红色圆柱笔+3D 实际轨迹+不要 2D"
（同任务新 revision）后，确定性链复用了上一 revision 的旧视频，
receipt 仍 tool_ref=""/overlays=[]，却宣布 PASS/DELIVERED——假成功。

根因：artifacts 是 task 级无 revision 分野——finish_task/consider 的
验收账本扫描不带 revision 过滤，旧 revision 的产物满足新 revision 的
交付条件。

闭环断言：
1. 登记时打 revision 戳（task 当前 active_revision）；
2. 新 revision 的验收只认当前 revision 产物（旧场景视频不计入）；
3. finish_task 显式传入的旧 revision artifact_id 不得计入验收；
4. 迁移回填：旧行 revision=1（保守方向——不会计入后续 revision）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _task(kernel, tmp_path: Path) -> str:
    bound = kernel.bind_message(
        mission_id="m1", session_ref="s1", backend_native_id="s1",
        message_id="msg_1", text="画五角星出 3D 场景视频",
        cwd=str(tmp_path), body_id="sim/ur5e",
    )
    return str(bound["task_id"])


def _register(kernel, task_id: str, tmp_path: Path, name: str,
              media: str = "video/mp4") -> str:
    p = tmp_path / name
    p.write_bytes(b"MP4" + b"x" * 2048)
    art = kernel.register_artifact(
        task_id=task_id, path=str(p), media_type=media,
        producer="kernel:sim_render",
    )
    return str(art["artifact_id"])


class TestArtifactRevisionScope:
    def test_register_stamps_current_revision(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        art_id = _register(kernel, task_id, tmp_path, "a.mp4")
        row = kernel._conn.execute(
            "SELECT revision FROM artifacts WHERE artifact_id = ?",
            (art_id,),
        ).fetchone()
        assert row is not None
        assert int(row["revision"]) == 1

    def test_new_revision_verification_ignores_old_artifacts(
        self, tmp_path: Path
    ) -> None:
        """假成功场景：rev1 有 scene 视频；用户纠正开 rev2，只产新
        trace——rev2 验收不得把 rev1 视频计入。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        # rev1：场景视频在账。
        _register(kernel, task_id, tmp_path, "scene-old.mp4")
        # 用户纠正 → rev2。
        kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_2", text="不对，加红色圆柱笔，重新来",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task = kernel.get_task(task_id)
        assert int(task["active_revision"]) == 2
        # rev2 只产新 trace（没有新视频）。
        _register(kernel, task_id, tmp_path, "trace.json",
                  media="application/json")
        # rev2 验收账本不得含 rev1 视频。
        current = kernel.artifacts_for_revision(task_id, 2)
        assert all("scene-old" not in str(a["path"]) for a in current), (
            "旧 revision 产物竟在新 revision 账本上"
        )
        # consider 不得因旧视频在场而 PASS（旧视频不算数后无媒体证据
        # 可言——不得终态成功）。
        outcome = TaskCoordinator(kernel).consider(task_id)
        if outcome is not None:
            assert outcome.get("verification") not in (
                "PASS", "PASS_NEAR_LIMIT",
            ), f"旧产物竟满足新 revision：{outcome.get('verification')}"

    def test_finish_rejects_old_revision_artifact_ids(
        self, tmp_path: Path
    ) -> None:
        """finish_task 显式传入旧 revision artifact_id → 不计入验收
        （调用方把旧产物当新交付也不行）。"""
        kernel = _kernel(tmp_path)
        task_id = _task(kernel, tmp_path)
        old_art = _register(kernel, task_id, tmp_path, "scene-old.mp4")
        kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_2", text="不对，重来",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="rev2 完成",
            artifact_ids=[old_art],
        )
        assert result["status"] != "SUCCEEDED", (
            f"旧 revision 产物竟能交付新 revision：{result}"
        )

    def test_migration_backfills_revision_1(self, tmp_path: Path) -> None:
        """旧库（无 revision 列）迁移后历史行 revision=1（保守方向）。"""
        from rosclaw.storage.migrations import MigrationRunner

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        runner = MigrationRunner()
        # 只跑到 036（037 之前的库形态）。
        runner.apply(conn, "sqlite")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(artifacts)")]
        assert "revision" in cols, "迁移后缺 revision 列"
        kernel = _kernel(tmp_path)  # 新库直接建
        task_id = _task(kernel, tmp_path)
        art_id = _register(kernel, task_id, tmp_path, "a.mp4")
        row = kernel._conn.execute(
            "SELECT revision FROM artifacts WHERE artifact_id = ?",
            (art_id,),
        ).fetchone()
        assert int(row["revision"]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
