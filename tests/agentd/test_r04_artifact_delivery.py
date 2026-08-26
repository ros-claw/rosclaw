"""R0-4 红测试（0826 体验审计 §5.R0-4）：ArtifactRef 用户可达交付面。

真实事故（0826 体验旅程）：MP4 生成并登记了，但模型/用户看不到
——"数据库里有文件"被当成了"交付成功"。

断言：
1. TaskOutcome 带用户可见 artifact_refs（artifact_id/kind/
   media_type/size/digest/open_command）——不是只有内部路径；
2. capability 产物自动登记后，artifact_refs 回进 ToolResult
   投影（模型看得到登记结果，不再被要求手动 deliver）；
3. `rosclaw artifact list/open/export` CLI：纯终端/SSH 环境
   artifact 可达（export 复制文件、open 无 DISPLAY 时给路径
   不失败）；
4. Gate：rosclaw_task 成功路径的 payload.artifact_refs 每条都
   含 artifact_id + open_command（用户最终答案可引用可打开
   的交付物；数据库有但用户面不可达 = 失败）。
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request
from tests.agentd.test_r01_production_chain import _kernel, _setup_ur5e
from tests.agentd.test_r02_task_spec_deliverables import _draw_task


class TestOutcomeArtifactRefs:
    def test_outcome_includes_user_visible_refs(self, tmp_path: Path) -> None:
        """Coordinator outcome 必须带 artifact_refs（id/kind/media/
        size/digest/open_command）——用户可达交付的权威视图。"""
        from rosclaw.agentd.task_execution import TaskExecutionService
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画一个五角星")
        kernel.note_tool_use(task_id, "rosclaw_task")
        TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        ).execute(
            task_id,
            recipe_inputs={"shape": "star5",
                           "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
        )
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        refs = outcome.get("artifact_refs")
        assert refs, f"outcome 缺 artifact_refs：{outcome}"
        gif = [r for r in refs if r.get("media_type") == "image/gif"]
        mp4 = [r for r in refs if r.get("media_type") == "video/mp4"]
        assert gif and mp4, refs
        for ref in refs:
            assert ref.get("artifact_id"), ref
            assert ref.get("kind"), ref
            assert ref.get("size_bytes", 0) > 0, ref
            assert ref.get("digest", "").startswith("sha256:"), ref
            assert ref.get("open_command", "").startswith(
                "rosclaw artifact open "
            ), ref


class TestCapabilityArtifactRefs:
    async def test_auto_register_returns_refs_in_projection(
        self, tmp_path: Path
    ) -> None:
        """capability 产物自动登记 → artifact_refs（含 kernel
        artifact_id）回进 ToolResult 投影——模型不需要也不应该
        手动 deliver capability 产物。"""
        service, mission = await _setup_ur5e(tmp_path)
        await service._ensure_mcp_discovered()
        lease = await _issue_lease(service, mission)
        # 真实 trace（render capability 的输入）。
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.08,
        )
        rollout = sim.simulate_cartesian_trajectory(plan["plan_id"])
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="r04_cap", lease=lease,
                arguments={
                    "capability_id": "simulation_render_trace",
                    "arguments": {"trace_id": rollout["trace_id"]},
                },
            )
        )
        assert result.ok, result.summary
        projection = json.loads(result.summary)
        refs = projection.get("artifact_refs") or []
        assert refs, f"投影缺 artifact_refs：{projection}"
        assert any(str(r.get("artifact_id", "")).startswith("art_")
                   for r in refs), refs
        assert all(r.get("open_command") for r in refs), refs
        await service.close()


class TestArtifactCli:
    def _register_one(self, tmp_path: Path) -> tuple[str, Path]:
        """文件账本（CLI 直读 <home>/agentd/missions.db——与生产
        同一事实源，不是 :memory:）。"""
        import sqlite3

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        (tmp_path / "agentd").mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            tmp_path / "agentd" / "missions.db", check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="画一个五角星",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1", backend_native_id="s1",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        f = tmp_path / "star.gif"
        f.write_bytes(b"GIF89a" + b"\x00" * 256)
        record = kernel.register_artifact(
            task_id=str(bound["task_id"]), path=str(f),
            media_type="image/gif", producer="kernel:test",
        )
        conn.commit()
        conn.close()
        return str(record["artifact_id"]), f

    def test_artifact_list(self, tmp_path: Path, capsys) -> None:
        artifact_id, f = self._register_one(tmp_path)
        from rosclaw.agentd.cli import main as agentd_main

        rc = agentd_main([
            "--home", str(tmp_path), "artifact", "list", "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        listing = json.loads(out)
        assert any(
            a["artifact_id"] == artifact_id and a["path"] == str(f)
            for a in listing
        ), listing

    def test_artifact_export_copies_file(self, tmp_path: Path) -> None:
        artifact_id, f = self._register_one(tmp_path)
        from rosclaw.agentd.cli import main as agentd_main

        dest = tmp_path / "exported.gif"
        rc = agentd_main([
            "--home", str(tmp_path), "artifact", "export",
            artifact_id, str(dest),
        ])
        assert rc == 0
        assert dest.exists()
        assert dest.read_bytes() == f.read_bytes()

    def test_artifact_open_headless_prints_path(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        """SSH/纯终端（无 DISPLAY）→ open 不失败，给出可复制的
        路径与 export 提示（OSC 8/xdg-open 是有显示时的增强）。"""
        artifact_id, f = self._register_one(tmp_path)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        from rosclaw.agentd.cli import main as agentd_main

        rc = agentd_main([
            "--home", str(tmp_path), "artifact", "open", artifact_id,
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert str(f) in out, out

    def test_artifact_open_unknown_id_honest(self, tmp_path: Path) -> None:
        from rosclaw.agentd.cli import main as agentd_main

        rc = agentd_main([
            "--home", str(tmp_path), "artifact", "open", "art_nope",
        ])
        assert rc != 0


class TestGateUserVisibleDelivery:
    async def test_payload_refs_openable(self, tmp_path: Path) -> None:
        """Gate R0-4：成功 payload 的 artifact_refs 每条含
        artifact_id + open_command——用户最终答案可引用可打开的
        交付物（DB 有但用户面不可达 = 失败）。"""
        service, mission = await _setup_ur5e(tmp_path)
        lease = await _issue_lease(service, mission)
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_task", mission=mission.mission_id,
                idem="r04_gate", lease=lease,
                arguments={
                    "goal": "draw_shape",
                    "parameters": {"shape": "star5"},
                },
            )
        )
        assert result.ok, result.summary
        payload = json.loads(result.summary)
        refs = payload.get("artifact_refs") or []
        assert refs, payload
        for ref in refs:
            assert ref.get("artifact_id"), ref
            assert ref.get("open_command", "").startswith(
                "rosclaw artifact open "
            ), ref
        media = {r.get("media_type") for r in refs}
        assert "image/gif" in media and "video/mp4" in media, media
        await service.close()
