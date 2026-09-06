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


async def _run_draw_chain(service, mission, tmp_path: Path) -> str:
    """大道至简 R0-2b：通用能力链（plan→simulate→render，模型面
    同款 rosclaw_compute 驱动）——capability 产物自动登记在同一
    任务，返回 task_id。"""
    import json as _json

    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
    from tests.agentd.test_pi_tool_bridge import _issue_lease, _request

    lease = await _issue_lease(service, mission)
    dispatcher = PiToolDispatcher(service)
    plan = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem="p04_plan", lease=lease,
            arguments={
                "capability_id": "trajectory_generate_planar_path",
                "arguments": {"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                              "scale_m": 0.10, "plane": "xy",
                              "max_segment_m": 0.02},
            },
        ),
    )
    assert plan.ok, plan.summary
    plan_id = _json.loads(plan.summary)["value"]["plan_id"]
    sim = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem="p04_sim", lease=lease,
            arguments={
                "capability_id": "ur5e_simulate_cartesian_trajectory",
                "arguments": {"plan_id": plan_id},
            },
        ),
    )
    assert sim.ok, sim.summary
    trace_id = _json.loads(sim.summary)["value"]["trace_id"]
    render = await dispatcher.execute(
        caller_pid=1, caller_uid=1000,
        request=_request(
            "rosclaw_compute", mission=mission.mission_id,
            idem="p04_render", lease=lease,
            arguments={
                "capability_id": "simulation_render_trace",
                "arguments": {"trace_id": trace_id, "format": "gif"},
            },
        ),
    )
    assert render.ok, render.summary
    row = service._store.connection.execute(
        "SELECT task_id FROM artifacts ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert row, "能力链无产物登记"
    return str(row["task_id"])


class TestKernelDeliveryProjection:
    async def test_artifacts_projected_to_outputs_zone(self, tmp_path: Path) -> None:
        """能力链跑通后：每个登记产物在 outputs/ 区有同内容投影
        （hardlink/copy 等价——sha256 一致），账本路径不变。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from rosclaw.task_kernel.run_store import run_dir
        from tests.agentd.test_pi_tool_bridge import _setup_ur5e

        service, mission = await _setup_ur5e(tmp_path)
        conn = service._store.connection
        kernel = service._task_kernel
        task_id = await _run_draw_chain(service, mission, tmp_path)
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
        await service.close()

    async def test_projection_failure_degraded_not_missing(
        self, tmp_path: Path
    ) -> None:
        """投影失败（outputs 区被破坏）→ delivery 仍 DELIVERED +
        workspace_projection DEGRADED——绝不翻转成 MISSING。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from rosclaw.task_kernel.run_store import run_dir
        from tests.agentd.test_pi_tool_bridge import _setup_ur5e

        service, mission = await _setup_ur5e(tmp_path)
        kernel = service._task_kernel
        task_id = await _run_draw_chain(service, mission, tmp_path)
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
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
