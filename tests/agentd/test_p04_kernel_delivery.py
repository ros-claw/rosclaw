"""0827 体验审计 P0-4 红测试：Kernel 原生产物交付投影。

0827 实证：Renderer 生成视频 → 模型用 Shell 复制 → bwrap 拒绝 →
模型手动 deliver → Artifact 登记成功 → Coordinator 又因 outputs
为空判交付 MISSING（两个交付真相互相矛盾）。

闭环断言：
1. 登记产物由**内核**自动投影到运行 outputs/ 区（hardlink/copy—
   —内核内部操作，不经模型 Shell、不依赖 bwrap）；ArtifactStore
   路径不变（权威仍在账本）；
2. outcome 带 workspace_projection 字段（OK/DEGRADED）；
3. 投影失败 → delivery 仍 DELIVERED + workspace_projection
   DEGRADED（绝不整体 MISSING）；
4. DEGRADED 时 artifact open_command 仍可用（账本权威不受投影
   影响）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _run_draw_chain(kernel, conn, tmp_path: Path) -> str:
    from rosclaw.agentd.task_execution import TaskExecutionService
    from tests.agentd.test_r02_task_spec_deliverables import _draw_task

    task_id = _draw_task(kernel, tmp_path, "画一个五角星")
    kernel.note_tool_use(task_id, "rosclaw_task")
    TaskExecutionService(kernel=kernel, conn=conn, home=tmp_path).execute(
        task_id,
        recipe_inputs={"shape": "star5",
                       "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
    )
    return task_id


class TestKernelDeliveryProjection:
    def test_artifacts_projected_to_outputs_zone(self, tmp_path: Path) -> None:
        """生产链跑通后：每个登记产物在 outputs/ 区有同内容投影
        （hardlink/copy 等价——sha256 一致），账本路径不变。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from rosclaw.task_kernel.run_store import run_dir
        from tests.agentd.test_r01_production_chain import _kernel

        kernel, conn = _kernel(tmp_path)
        task_id = _run_draw_chain(kernel, conn, tmp_path)
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        assert outcome.get("delivery") == "DELIVERED", outcome
        assert outcome.get("workspace_projection") == "OK", outcome
        outputs = run_dir(tmp_path, task_id, 1) / "outputs"
        assert outputs.is_dir(), f"outputs 区不存在：{outputs}"
        rows = conn.execute(
            "SELECT path, sha256 FROM artifacts WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        media = [r for r in rows if r["path"].endswith((".gif", ".mp4"))]
        assert media, "生产链应登记 GIF/MP4 产物"
        for row in media:
            projected = outputs / Path(str(row["path"])).name
            assert projected.exists(), (
                f"产物未投影到 outputs/：{row['path']}"
            )
            digest = hashlib.sha256(projected.read_bytes()).hexdigest()
            assert digest == str(row["sha256"]), (
                f"投影内容与账本不一致：{projected}"
            )

    def test_projection_failure_degraded_not_missing(
        self, tmp_path: Path
    ) -> None:
        """投影失败（outputs 区被破坏）→ delivery 仍 DELIVERED +
        workspace_projection DEGRADED——绝不翻转成 MISSING。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from rosclaw.task_kernel.run_store import run_dir
        from tests.agentd.test_r01_production_chain import _kernel

        kernel, conn = _kernel(tmp_path)
        task_id = _run_draw_chain(kernel, conn, tmp_path)
        # 破坏 outputs 区：目录替换成普通文件 → mkdir/link 必败。
        outputs = run_dir(tmp_path, task_id, 1) / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.rmtree(outputs)
        outputs.write_text("sabotaged", encoding="utf-8")
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        assert outcome.get("delivery") == "DELIVERED", (
            f"投影失败竟翻转交付判定：{outcome}"
        )
        assert outcome.get("workspace_projection") == "DEGRADED", outcome
        # 账本权威不受投影影响：artifact_refs 仍可 open。
        refs = outcome.get("artifact_refs") or []
        assert refs, "DEGRADED 时 artifact_refs 不应消失"
        assert all(
            str(r.get("open_command", "")).startswith("rosclaw artifact open ")
            for r in refs
        ), refs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
