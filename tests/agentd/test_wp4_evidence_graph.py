"""WP-4 红测试（0823 审计 §三.P0-2/§四.WP-4）：Artifact Evidence Graph。

红测试先行——血缘图不存在时必须红。

审计规则：
- 每个交付物必须保存完整血缘：Artifact → RenderReceipt →
  SimulationTrace → TrajectoryPlan → ResourceManifest；
- task_finish 必须遍历整个图：task/revision 一致、digest 一致、
  renderer 输入确实是该 trace、trace 确实来自 canonical 资源；
- 不允许跨 revision 拼接；不允许模型手工登记把 EXPERIMENTAL 升级
  TRUSTED；
- matplotlib 自写 GIF（无 render receipt 血缘）不能因同任务有受信
  trace 就升级为受信产物。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _kernel(tmp_path: Path):
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return TaskKernel(conn, tmp_path)


def _star_task(kernel, tmp_path: Path) -> str:
    bound = kernel.bind_message(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        message_id="m1", text="画五角星", cwd=str(tmp_path),
        body_id="sim/ur5e",
    )
    task_id = str(bound["task_id"])
    kernel.note_tool_use(task_id, "rosclaw_task")  # 行为任务标记
    return task_id


def _run_trusted_pipeline(home: Path) -> dict:
    """受信管道：plan → rollout → scene render（带血缘）。"""
    from rosclaw.agentd.sim_render import render_scene_trace
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    run = sim.simulate_cartesian_trajectory(plan["plan_id"])
    render = render_scene_trace(home, run["trace_id"])
    return {"plan": plan, "run": run, "render": render}


class TestLineageRegistration:
    def test_render_artifact_carries_lineage(self, tmp_path: Path) -> None:
        """渲染 GIF 登记时必须带血缘：render receipt digest + input
        trace digest（可追溯 RenderReceipt→Trace）。"""
        kernel = _kernel(tmp_path)
        task_id = _star_task(kernel, tmp_path)
        ctx = _run_trusted_pipeline(tmp_path)
        gif = ctx["render"]["artifact"]["path"]
        receipt_path = (
            tmp_path / "sim" / "traces" / ctx["run"]["trace_id"]
            / "render_receipt.json"
        )
        art = kernel.register_artifact(
            task_id=task_id, path=gif, media_type="image/gif",
            producer="kernel:sim_render",
            metadata={"lineage": {"render_receipt_path": str(receipt_path)}},
        )
        meta = json.loads(str(art.get("metadata_json", "{}")) or "{}")
        lineage = meta.get("lineage") or {}
        assert lineage.get("render_receipt_digest", "").startswith("sha256:")
        assert lineage.get("input_trace_digest", "").startswith("sha256:")
        assert lineage.get("trace_id") == ctx["run"]["trace_id"]


class TestFinishWalksGraph:
    def test_genuine_chain_succeeds(self, tmp_path: Path) -> None:
        """完整血缘（GIF←render receipt←trace←plan←canonical）→
        验收通过。"""
        kernel = _kernel(tmp_path)
        task_id = _star_task(kernel, tmp_path)
        ctx = _run_trusted_pipeline(tmp_path)
        task = kernel.get_task(task_id)
        ws = Path(str(task["workspace_path"]))
        # 交付物复制进任务工作区（单根规则）。
        import shutil

        gif_dst = ws / "star-scene.gif"
        shutil.copy(ctx["render"]["artifact"]["path"], gif_dst)
        receipt_dst = ws / "render_receipt.json"
        shutil.copy(
            tmp_path / "sim" / "traces" / ctx["run"]["trace_id"]
            / "render_receipt.json",
            receipt_dst,
        )
        art = kernel.register_artifact(
            task_id=task_id, path=str(gif_dst), media_type="image/gif",
            producer="kernel:sim_render",
            metadata={
                "lineage": {
                    "render_receipt_path": str(receipt_dst),
                    "trace_id": ctx["run"]["trace_id"],
                },
                "resource": ctx["run"]["resource"],
            },
        )
        result = kernel.finish_task(
            task_id=task_id, summary="五角星仿真完成",
            artifact_ids=[str(art["artifact_id"])],
        )
        assert result["status"] == "SUCCEEDED", result.get("failures")

    def test_matplotlib_gif_cannot_borrow_trusted_trace(
        self, tmp_path: Path
    ) -> None:
        """审计 P0-2 核心：同任务有受信 trace，但登记的 GIF 没有
        render 血缘（模型自写 matplotlib 产物）→ 不得 SUCCEEDED。"""
        kernel = _kernel(tmp_path)
        task_id = _star_task(kernel, tmp_path)
        _run_trusted_pipeline(tmp_path)  # 同任务存在受信 trace（审计场景前提）
        task = kernel.get_task(task_id)
        ws = Path(str(task["workspace_path"]))
        # 模型自写 GIF（无 render receipt 血缘）。
        fake_gif = ws / "fake.gif"
        fake_gif.write_bytes(b"GIF89a" + b"x" * 2048)
        art = kernel.register_artifact(
            task_id=task_id, path=str(fake_gif), media_type="image/gif",
            producer="model:rosclaw_artifact_register",
        )
        meta = json.loads(str(art.get("metadata_json", "{}")) or "{}")
        assert meta.get("evidence_tier") == "EXPERIMENTAL"
        result = kernel.finish_task(
            task_id=task_id, summary="完成",
            artifact_ids=[str(art["artifact_id"])],
        )
        assert result["status"] != "SUCCEEDED"
        assert any(
            "LINEAGE" in str(f) or "TRUSTED_EVIDENCE" in str(f)
            for f in result.get("failures", [])
        ), result

    def test_cross_revision_splice_rejected(self, tmp_path: Path) -> None:
        """跨 revision 拼接：revision 1 的受信 trace 不能被
        revision 2 的交付物借用。"""
        kernel = _kernel(tmp_path)
        task_id = _star_task(kernel, tmp_path)
        ctx = _run_trusted_pipeline(tmp_path)
        # revision 2（用户否定后重开）。
        kernel.bind_message(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            message_id="m2", text="不对，重来", cwd=str(tmp_path),
            body_id="sim/ur5e",
        )
        task = kernel.get_task(task_id)
        assert int(task["active_revision"]) == 2
        ws = Path(str(task["workspace_path"]))
        import shutil

        gif_dst = ws / "star-scene.gif"
        shutil.copy(ctx["render"]["artifact"]["path"], gif_dst)
        receipt_dst = ws / "render_receipt.json"
        shutil.copy(
            tmp_path / "sim" / "traces" / ctx["run"]["trace_id"]
            / "render_receipt.json",
            receipt_dst,
        )
        art = kernel.register_artifact(
            task_id=task_id, path=str(gif_dst), media_type="image/gif",
            producer="kernel:sim_render",
            metadata={
                "lineage": {
                    "render_receipt_path": str(receipt_dst),
                    "trace_id": ctx["run"]["trace_id"],
                },
                "resource": ctx["run"]["resource"],
            },
        )
        result = kernel.finish_task(
            task_id=task_id, summary="借用 r1 证据",
            artifact_ids=[str(art["artifact_id"])],
        )
        assert result["status"] != "SUCCEEDED", (
            "跨 revision 拼接被接受"
        )
