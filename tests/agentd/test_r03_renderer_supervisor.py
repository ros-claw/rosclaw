"""R0-3 红测试（0826 体验审计 §5.R0-3）：RendererSupervisor 结构化
进程协议 + 内核内一次降级 + 熔断范围修复。

真实事故（0826 体验旅程）：
- sim_render 直接 json.loads(子进程 stdout)——空输出/噪声/rc=0
  无结果泄漏成裸 JSONDecodeError 给模型；
- 模型换 camera 参数绕过 doom-loop 指纹，同一基础设施故障被
  重试三次；
- scene renderer 用 empty world 且不读 TaskSpec——"桌面/持笔"
  场景语义缺位（不得假装）。

断言：
1. 结构化 IPC：子进程写原子 result JSON 文件；stdout/stderr 只
   作诊断——噪声 stdout 不污染结果；
2. 稳定错误码：rc=0 无 result → RENDER_RESULT_MISSING（不是
   JSONDecodeError）；result 坏 → RENDER_RESULT_CORRUPT；缺字
   段 → RENDER_RESULT_INCOMPLETE；rc!=0 → RENDER_FAILED；
3. 后端降级内核内一次：首选后端渲染失败 → 自动降级一次；全
   部失败 → 一条有语义错误（不多次抛给模型）；
4. 熔断范围：换 camera 不绕过 renderer 失败熔断（指纹不含
   camera）；
5. TaskSpec 场景语义：world_ref=tabletop → 场景加载 tabletop；
   tool_ref=pen 无资产 → TOOL_ASSET_MISSING（不假装持笔）；
6. recipe 场景节点：spec 要求 scene_video → 场景 GIF+MP4 以
   scene_3d kind 登记（满足交付）；场景失败 → PARTIAL（2D 交付
   仍在）；无要求 → 不跑场景渲染；
7. RenderReceipt 含 world/tool/trace digest。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agentd.test_r01_production_chain import _kernel
from tests.agentd.test_r02_task_spec_deliverables import _draw_task


def _make_trace(home: Path) -> dict:
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    sim = SimTrajectoryService(home)
    plan = sim.generate_planar_path(
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
    )
    return sim.simulate_cartesian_trajectory(plan["plan_id"])


def _fake_run_factory(behaviors):
    """按 argv 行为表伪造 subprocess.run（结构化协议故障注入）。"""
    calls: list[list] = []

    class Proc:
        def __init__(self, rc, out=b"", err=b""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def fake_run(argv, **kwargs):
        calls.append(argv)
        idx = len(calls) - 1
        action = behaviors[min(idx, len(behaviors) - 1)]
        kind = action[0]
        if kind == "probe_ok":
            return Proc(0, b"OK")
        if kind == "probe_fail":
            return Proc(1, b"", b"no display")
        if kind == "render_ok":
            # 结构化协议：写 result 文件（argv 约定含 --result）。
            result_path = Path(argv[argv.index("--result") + 1])
            payload = action[1]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            noise = action[2] if len(action) > 2 else b""
            return Proc(0, noise, b"")
        if kind == "render_silent":
            return Proc(0, b"", b"")  # rc=0 但无 result（事故原型）
        if kind == "render_corrupt":
            result_path = Path(argv[argv.index("--result") + 1])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{not json", encoding="utf-8")
            return Proc(0, b"", b"")
        if kind == "render_rc_fail":
            return Proc(1, b"", b"GL crashed badly")
        raise AssertionError(f"unknown fake behavior {action}")

    return fake_run, calls


class TestStructuredIPC:
    def test_empty_stdout_stable_error_not_jsondecode(
        self, tmp_path: Path
    ) -> None:
        """rc=0 但 stdout 空（事故原型）→ RENDER_RESULT_MISSING
        稳定错误码，绝不泄漏裸 JSONDecodeError。"""
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        fake, _calls = _fake_run_factory([
            ("probe_ok",), ("render_silent",),
            ("probe_ok",), ("render_silent",),
            ("probe_ok",), ("render_silent",),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="RENDER_RESULT_MISSING"):
                sim_render.render_scene_trace(
                    tmp_path, trace["trace_id"],
                )
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]

    def test_noisy_stdout_does_not_corrupt_result(
        self, tmp_path: Path
    ) -> None:
        """stdout 带库警告/噪声 → 结果仍从 result 文件读取（不
        解析 stdout）。"""
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        payload = {
            "ok": True,
            "artifact": {"path": "/tmp/x.gif", "frames": 60,
                         "format": "gif", "bytes": 1000},
            "artifacts": {"gif": {"path": "/tmp/x.gif", "frames": 60},
                          "mp4": {"path": "/tmp/x.mp4", "frames": 60}},
            "receipt": {"backend": "egl"},
        }
        fake, _calls = _fake_run_factory([
            ("probe_ok",),
            ("render_ok", payload, b"WARNING: libGL noise\nOK"),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            result = sim_render.render_scene_trace(
                tmp_path, trace["trace_id"],
            )
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]
        assert result["ok"] is True
        assert result["artifact"]["frames"] == 60

    def test_corrupt_result_stable_error(self, tmp_path: Path) -> None:
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        fake, _calls = _fake_run_factory([
            ("probe_ok",), ("render_corrupt",),
            ("probe_ok",), ("render_corrupt",),
            ("probe_ok",), ("render_corrupt",),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="RENDER_RESULT_CORRUPT"):
                sim_render.render_scene_trace(tmp_path, trace["trace_id"])
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]

    def test_incomplete_result_stable_error(self, tmp_path: Path) -> None:
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        fake, _calls = _fake_run_factory([
            ("probe_ok",), ("render_ok", {"ok": True}),
            ("probe_ok",), ("render_ok", {"ok": True}),
            ("probe_ok",), ("render_ok", {"ok": True}),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            with pytest.raises(
                ValueError, match="RENDER_RESULT_INCOMPLETE"
            ):
                sim_render.render_scene_trace(tmp_path, trace["trace_id"])
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]

    def test_rc_nonzero_render_failed(self, tmp_path: Path) -> None:
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        fake, _calls = _fake_run_factory([
            ("probe_ok",), ("render_rc_fail",),
            ("probe_ok",), ("render_rc_fail",),
            ("probe_ok",), ("render_rc_fail",),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="RENDER_FAILED"):
                sim_render.render_scene_trace(tmp_path, trace["trace_id"])
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]


class TestBackendFallbackOnce:
    def test_fallback_internal_once_then_success(
        self, tmp_path: Path
    ) -> None:
        """首选后端渲染失败 → 内核内降级一次成功——调用方只看到
        一个成功结果（没有"模型试三次"）。"""
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        payload = {
            "ok": True,
            "artifact": {"path": "/tmp/x.gif", "frames": 60,
                         "format": "gif", "bytes": 1000},
            "artifacts": {"gif": {"path": "/tmp/x.gif", "frames": 60},
                          "mp4": {"path": "/tmp/x.mp4", "frames": 60}},
            "receipt": {"backend": "osmesa"},
        }
        fake, calls = _fake_run_factory([
            ("probe_ok",),              # egl probe
            ("render_silent",),         # egl render 失败（rc=0 无结果）
            ("probe_ok",),              # osmesa probe
            ("render_ok", payload),     # osmesa render 成功
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            result = sim_render.render_scene_trace(
                tmp_path, trace["trace_id"],
            )
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]
        assert result["ok"] is True
        renders = [c for c in calls if "--result" in c]
        assert len(renders) <= 2, f"降级超过一次: {len(renders)}"

    def test_all_backends_fail_single_semantic_error(
        self, tmp_path: Path
    ) -> None:
        """全部后端失败 → 一条有语义错误（RENDER_FAILED/UNAVAILABLE）
        ——不是三次裸异常。"""
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        fake, _calls = _fake_run_factory([
            ("probe_fail",), ("probe_fail",), ("probe_fail",),
        ])
        original = sim_render.subprocess.run
        sim_render.subprocess.run = fake  # type: ignore[assignment]
        try:
            with pytest.raises(
                ValueError, match="RENDER_BACKEND_UNAVAILABLE|RENDER_FAILED"
            ):
                sim_render.render_scene_trace(tmp_path, trace["trace_id"])
        finally:
            sim_render.subprocess.run = original  # type: ignore[assignment]


class TestSceneContentFromSpec:
    def test_tabletop_world_loaded(self, tmp_path: Path) -> None:
        """spec world_ref=world:tabletop → 场景渲染加载 tabletop
        （不是 empty 冒充桌面）。"""
        from rosclaw.agentd import sim_render

        created: list[str] = []

        class FakeSandbox:
            def __init__(self, world: str):
                self.world = world
                self.has_physics = True
                self.load_error = ""

            @classmethod
            def create(cls, robot, world, engine, publisher=None):
                created.append(world)
                return cls(world)

            def close(self):
                pass

        trace = _make_trace(tmp_path)
        import contextlib

        import rosclaw.sandbox.sandbox_api as sandbox_api

        original = sandbox_api.Sandbox.create
        sandbox_api.Sandbox.create = FakeSandbox.create  # type: ignore[assignment]
        try:
            # 子进程入口直测：world_id 必须流进 Sandbox.create
            # （进程边界外 spy 不到子进程）。
            with contextlib.suppress(Exception):
                sim_render._render_impl(
                    tmp_path, trace["trace_id"], camera="follow",
                    max_frames=4, width=64, height=64,
                    world_id="tabletop", tool_ref="",
                )
        finally:
            sandbox_api.Sandbox.create = original  # type: ignore[assignment]
        assert created and created[0] == "tabletop", created

    def test_tool_asset_missing_honest(self, tmp_path: Path) -> None:
        """spec tool_ref=tool:pen 但无笔资产 → TOOL_ASSET_MISSING
        （诚实失败，不假装持笔）。"""
        from rosclaw.agentd import sim_render

        trace = _make_trace(tmp_path)
        with pytest.raises(ValueError, match="TOOL_ASSET_MISSING"):
            sim_render.render_scene_trace(
                tmp_path, trace["trace_id"], tool_ref="tool:pen",
            )


class TestRecipeSceneNode:
    def test_scene_video_produced_when_required(
        self, tmp_path: Path
    ) -> None:
        """spec 要求 scene_video → recipe 场景节点产出 scene_3d
        kind 的 GIF+MP4 并满足交付（真实渲染链）。"""
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画五角星并做仿真视频")
        kernel.note_tool_use(task_id, "rosclaw_task")
        outcome = TaskExecutionService(
            kernel=kernel, conn=conn, home=tmp_path,
        ).execute(
            task_id,
            recipe_inputs={"shape": "star5",
                           "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
        )
        assert outcome.ok, outcome.failures
        kinds = set()
        for a in outcome.artifacts:
            row = kernel._conn.execute(
                "SELECT metadata_json FROM artifacts WHERE artifact_id = ?",
                (a["artifact_id"],),
            ).fetchone()
            meta = json.loads(str(row["metadata_json"] or "{}"))
            kinds.add((meta.get("lineage") or {}).get("kind", ""))
        assert "scene_3d" in kinds, kinds
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", task["state"]

    def test_scene_failure_partial_2d_survives(
        self, tmp_path: Path
    ) -> None:
        """场景渲染失败（注入）→ PARTIAL：DELIVERABLE_MISSING
        scene_video，但 2D 预览交付不受影响（不整体失败）。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画五角星并做仿真视频")
        kernel.note_tool_use(task_id, "rosclaw_task")
        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_RESULT_MISSING: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        try:
            outcome = TaskExecutionService(
                kernel=kernel, conn=conn, home=tmp_path,
            ).execute(
                task_id,
                recipe_inputs={"shape": "star5",
                               "center_m": [0.35, 0.25, 0.30],
                               "scale_m": 0.10},
            )
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
        assert not outcome.ok
        assert any(
            "scene_video" in f for f in outcome.failures
        ), outcome.failures
        media = {a["media_type"] for a in outcome.artifacts}
        assert "image/gif" in media, "2D 预览交付被场景故障拖死"

    def test_no_scene_required_no_scene_render(
        self, tmp_path: Path
    ) -> None:
        """无 scene_video 要求 → 不跑场景渲染（不浪费、不误伤）。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画一个五角星")
        kernel.note_tool_use(task_id, "rosclaw_task")
        called: list[bool] = []
        original = sim_render.render_scene_trace

        def spy(*a, **k):
            called.append(True)
            return original(*a, **k)

        sim_render.render_scene_trace = spy  # type: ignore[assignment]
        try:
            outcome = TaskExecutionService(
                kernel=kernel, conn=conn, home=tmp_path,
            ).execute(
                task_id,
                recipe_inputs={"shape": "star5",
                               "center_m": [0.35, 0.25, 0.30],
                               "scale_m": 0.10},
            )
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
        assert outcome.ok, outcome.failures
        assert not called, "无要求不应跑场景渲染"


class TestBreakerScope:
    async def test_camera_change_does_not_bypass_breaker(
        self, tmp_path: Path
    ) -> None:
        """熔断范围修复：renderer 失败一次后，换 camera 重试同一
        capability 仍被 DOOM_LOOP 拒绝（指纹不含 camera）。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )

        service, mission = await _setup(tmp_path)
        await service._ensure_mcp_discovered()
        lease = await _issue_lease(service, mission)
        dispatcher = PiToolDispatcher(service)
        args_a = {
            "capability_id": "simulation_render_scene",
            "arguments": {"trace_id": "trace_nope", "camera": "follow"},
        }
        first = await dispatcher.execute(
            _request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="r03_brk_a", lease=lease, arguments=args_a,
            )
        )
        assert not first.ok, "不存在 trace 的渲染必须失败"
        args_b = {
            "capability_id": "simulation_render_scene",
            "arguments": {"trace_id": "trace_nope", "camera": "top"},
        }
        second = await dispatcher.execute(
            _request(
                "rosclaw_compute", mission=mission.mission_id,
                idem="r03_brk_b", lease=lease, arguments=args_b,
            )
        )
        assert second.error_code == "DOOM_LOOP", (
            f"换 camera 绕过了熔断: {second.error_code} {second.summary}"
        )
        await service.close()
