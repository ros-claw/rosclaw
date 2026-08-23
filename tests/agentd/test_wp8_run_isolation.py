"""WP-8 红测试（0823 审计 §四.WP-8）：项目源码与任务产物隔离。

红测试先行——runs 布局不存在时必须红。

0823 实测：模型把 star_ur5e_sim.gif 写进项目源码目录并登记成交付
物——项目树成了任务垃圾场。布局：

~/.rosclaw/runs/<task_id>/r<revision>/{scratch,outputs,evidence,logs}

1. 新 task 绑定即建 r1 四区；revision+1 建 r<N> 四区；
2. task 详情暴露 run_dir（模型/界面可查）；
3. scratch 区文件不得登记为交付物（SCRATCH_NOT_DELIVERABLE——
   草稿不是交付）；
4. outputs/evidence 区登记记录 zone（交付/证据可归因）；
5. pi.context 响应带 active_run（模型每轮知道写哪里）。
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
    return TaskKernel(conn, home)


def _bind(kernel, home: Path, text: str = "画五角星", msg: str = "m1") -> dict:
    return kernel.bind_message(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        message_id=msg, text=text, cwd=str(home),
    )


class TestRunLayout:
    def test_new_task_creates_run_zones(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        run = tmp_path / "runs" / task_id / "r1"
        for zone in ("scratch", "outputs", "evidence", "logs"):
            assert (run / zone).is_dir(), f"缺运行区: {run / zone}"

    def test_revision_bump_creates_new_run_dir(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        # 同 session 第二条消息 → revision+1。
        _bind(kernel, tmp_path, text="改大一点儿", msg="m2")
        assert (tmp_path / "runs" / task_id / "r2" / "outputs").is_dir()

    def test_task_detail_exposes_run_dir(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task = kernel.get_task(str(bound["task_id"]))
        assert task is not None
        run_dir = Path(str(task.get("run_dir", "")))
        assert run_dir.name == "r1"
        assert run_dir.parent.name == str(bound["task_id"])
        assert (run_dir / "outputs").is_dir()


class TestArtifactZoneDiscipline:
    def test_scratch_rejected_as_deliverable(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        scratch = tmp_path / "runs" / task_id / "r1" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        draft = scratch / "draft.py"
        draft.write_text("print('draft')", encoding="utf-8")
        with pytest.raises(ValueError, match="SCRATCH_NOT_DELIVERABLE"):
            kernel.register_artifact(
                task_id=task_id, path=str(draft), media_type="text/x-python",
            )

    def test_outputs_zone_recorded(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        outputs = tmp_path / "runs" / task_id / "r1" / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        gif = outputs / "star.gif"
        gif.write_bytes(b"GIF89a" + b"\x00" * 64)
        result = kernel.register_artifact(
            task_id=task_id, path=str(gif), media_type="image/gif",
        )
        artifact = result.get("artifact") or result
        assert artifact.get("zone") == "outputs", artifact

    def test_evidence_zone_recorded(self, tmp_path: Path) -> None:
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        evidence = tmp_path / "runs" / task_id / "r1" / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        receipt = evidence / "receipt.json"
        receipt.write_text("{}", encoding="utf-8")
        result = kernel.register_artifact(
            task_id=task_id, path=str(receipt), media_type="application/json",
        )
        artifact = result.get("artifact") or result
        assert artifact.get("zone") == "evidence", artifact


class TestContextCarriesActiveRun:
    def test_pi_context_response_has_active_run(self, tmp_path: Path) -> None:
        """pi.context 响应（带 session）必须带 active_run——模型每轮
        知道当前任务的运行目录与四区。"""
        kernel = _kernel(tmp_path)
        bound = _bind(kernel, tmp_path)
        task_id = str(bound["task_id"])
        # 直接测内核查询助手（server 接线薄壳）。
        active = kernel.active_task_for_session("mis_1", "s1")
        assert active is not None, "缺 active_task_for_session 查询"
        assert active["task_id"] == task_id
        assert active["run_dir"].endswith("r1")
        zones = active.get("zones") or {}
        assert set(zones) >= {"scratch", "outputs", "evidence", "logs"}
