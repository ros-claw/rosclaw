"""R0-2 红测试（0826 体验审计 §5.R0-2）：TaskSpec 交付与验收语义。

真实事故（0826 体验旅程）：
- 用户要"末端持笔、垂直桌面画五角星、做仿真视频"——实际只证明
  了 UR5e 末端轨迹跟踪 + 2D 预览 GIF；场景渲染器用 empty world，
  无笔/桌面/接触/场景视频证据，却被整体判 VERIFIED；
- preview_2d 与 scene_3d 不分 kind——2D GIF 能冒充用户要的
  scene video。

断言：
1. 契约：TaskSpecV2 冻结 deliverables（kind/media_type/required/
   min_frames/min_resolution）+ subjects.tool_ref +
   constraints.contact_required/tool_axis_aligned——未知 kind 拒绝；
2. 编译：自然语言目标 → deliverables/tool/world/contact（通用
   标记——无形状特例）；
3. 产物 kind 分野：preview_2d / scene_3d / robot_video 是不同
   kind——2D 预览不能满足 scene_video 交付；
4. 多维 outcome：运动执行 PASS 但 required scene_video 缺失 →
   verification PARTIAL + delivery MISSING/PARTIAL，不是整体
   VERIFIED（账本 SUCCEEDED）；
5. Gate：3D 渲染缺失（renderer 打坏）→ 任务只能 PARTIAL/
   NEEDS_REPAIR——最终 payload 不得宣称"任务完整完成"；
6. 回归：无媒体要求的 draw 任务仍完整 SUCCEEDED。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request
from tests.agentd.test_r01_production_chain import _kernel, _setup_ur5e


class TestDeliverableContract:
    def test_deliverable_fields_frozen(self) -> None:
        from rosclaw.contracts.agent.task_spec import (
            TaskDeliverableV2,
            TaskSpecV2,
        )

        spec = TaskSpecV2(
            spec_id="tspec_1", task_id="task_1", revision=1,
            goal={"natural_language": "画五角星并做仿真视频",
                  "intent": "manipulation.draw_path"},
            subjects={"body_ref": "robot:ur5e", "tool_ref": "tool:pen",
                      "world_ref": "world:tabletop"},
            constraints={"mode": "SIMULATION", "contact_required": True,
                         "tool_axis_aligned_with_plane_normal_deg": 3.0},
            deliverables=[
                TaskDeliverableV2(
                    kind="scene_video", media_type="video/mp4",
                    required=True, min_frames=60,
                    min_resolution=[640, 360],
                ),
                TaskDeliverableV2(
                    kind="preview_animation", media_type="image/gif",
                    required=False,
                ),
            ],
        )
        assert spec.deliverables[0].kind == "scene_video"
        assert spec.subjects.tool_ref == "tool:pen"
        assert spec.constraints.contact_required is True

    def test_unknown_deliverable_kind_rejected(self) -> None:
        from rosclaw.contracts.agent.task_spec import TaskDeliverableV2

        with pytest.raises(ValueError, match="kind|taxonomy|分类"):
            TaskDeliverableV2(kind="hologram", media_type="video/mp4")

    def test_kinds_are_distinct(self) -> None:
        """preview_2d / scene_3d / robot_video 必须是不同 artifact
        kind（2D 预览不能冒充场景视频）。"""
        from rosclaw.contracts.agent.task_spec import (
            DELIVERABLE_KIND_TO_ARTIFACT_KIND,
        )

        kinds = set(DELIVERABLE_KIND_TO_ARTIFACT_KIND.values())
        assert "preview_2d" in kinds
        assert "scene_3d" in kinds
        assert "robot_video" in kinds
        assert len(kinds) == len(DELIVERABLE_KIND_TO_ARTIFACT_KIND)


class TestSpecCompiler:
    def _compile(self, text: str):
        from rosclaw.task_kernel.task_spec import compile_task_spec

        return compile_task_spec(
            task_id="task_1", revision=1, goal_text=text,
            body_id="sim/ur5e", mode="SIMULATION", acceptance_spec_id="",
        )

    def test_video_goal_produces_scene_video_deliverable(self) -> None:
        spec = self._compile("机械臂末端持笔垂直桌面画五角星，并做仿真视频")
        assert spec.goal.intent == "manipulation.draw_path"
        scene = [
            d for d in spec.deliverables
            if d.kind == "scene_video" and d.required
        ]
        assert scene, f"缺 required scene_video：{spec.deliverables}"
        assert scene[0].media_type == "video/mp4"
        assert spec.subjects.tool_ref == "tool:pen"
        assert spec.subjects.world_ref == "world:tabletop"
        assert spec.constraints.contact_required is True

    def test_plain_draw_no_required_scene(self) -> None:
        spec = self._compile("画一个五角星")
        required = [d for d in spec.deliverables if d.required]
        assert not any(d.kind == "scene_video" for d in required), (
            f"无视频要求不得编造 required scene_video：{spec.deliverables}"
        )


def _draw_task(kernel, home: Path, text: str) -> str:
    kernel.persist_input(
        mission_id="mis_1", session_ref="s1",
        message_id="msg_1", text=text,
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1", backend_native_id="s1",
        cwd=str(home), body_id="sim/ur5e",
    )
    return str(bound["task_id"])


class TestDeliverableVerification:
    def test_missing_scene_video_not_full_pass(self, tmp_path: Path) -> None:
        """运动执行成功但 required scene_video 未交付 → 不得整体
        PASS（finish_task 必须含 DELIVERABLE_MISSING）。

        R0-3 后场景链存在——本测试显式注入渲染故障（这正是 Gate
        R0-2 要求的"把 3D renderer 故意打坏"）。
        """
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.task_execution import TaskExecutionService

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画五角星并做仿真视频")
        kernel.note_tool_use(task_id, "rosclaw_task")
        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_BACKEND_UNAVAILABLE: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        try:
            outcome = TaskExecutionService(
                kernel=kernel, conn=conn, home=tmp_path,
            ).execute(
                task_id,
                recipe_inputs={"shape": "star5",
                               "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
            )
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
        # 运动执行成功（refs 全产出）但交付不完整。
        assert "TraceRef" in outcome.refs
        assert not outcome.ok, "required scene_video 缺失不得整体 ok"
        assert any(
            "DELIVERABLE_MISSING" in f and "scene_video" in f
            for f in outcome.failures
        ), outcome.failures
        task = kernel.get_task(task_id)
        assert task["state"] != "SUCCEEDED", (
            f"交付不完整不得 SUCCEEDED：{task['state']}"
        )

    def test_preview_does_not_satisfy_scene_video(
        self, tmp_path: Path
    ) -> None:
        """2D 预览（preview_2d kind 的 GIF/MP4）不满足 scene_video
        交付——kind 分野是硬边界。"""
        from rosclaw.task_kernel.deliverables import deliverable_verdict

        kernel, _conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画五角星")
        f = tmp_path / "preview.gif"
        f.write_bytes(b"GIF89a" + b"\x00" * 128)
        kernel.register_artifact(
            task_id=task_id, path=str(f), media_type="image/gif",
            producer="kernel:test",
            metadata={"lineage": {"kind": "preview_2d",
                                  "trace_id": "t1"}},
        )
        artifacts = [
            dict(r) for r in kernel._conn.execute(
                "SELECT * FROM artifacts WHERE task_id = ?", (task_id,),
            ).fetchall()
        ]
        verdict = deliverable_verdict(
            [{"kind": "scene_video", "media_type": "video/mp4",
              "required": True}],
            artifacts,
        )
        assert verdict["missing"] == ["scene_video"], verdict


class TestCoordinatorOutcome:
    def test_partial_delivery_outcome(self, tmp_path: Path) -> None:
        """Coordinator：执行成功 + required scene_video 缺失（注入
        渲染故障）→ execution SUCCEEDED + verification PARTIAL +
        delivery MISSING/PARTIAL——不是整体 VERIFIED/DELIVERED。"""
        from rosclaw.agentd import sim_render
        from rosclaw.agentd.task_execution import TaskExecutionService
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        kernel, conn = _kernel(tmp_path)
        task_id = _draw_task(kernel, tmp_path, "画五角星并做仿真视频")
        kernel.note_tool_use(task_id, "rosclaw_task")
        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_BACKEND_UNAVAILABLE: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        try:
            TaskExecutionService(
                kernel=kernel, conn=conn, home=tmp_path,
            ).execute(
                task_id,
                recipe_inputs={"shape": "star5",
                               "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
            )
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        assert outcome["execution"] == "SUCCEEDED"
        assert outcome["verification"] == "PARTIAL", outcome
        assert outcome["delivery"] in ("MISSING", "PARTIAL"), outcome
        assert outcome["lifecycle"] != "COMPLETED", outcome


class TestGateRendererBroken:
    async def test_broken_scene_renderer_honest_partial(
        self, tmp_path: Path
    ) -> None:
        """Gate R0-2：3D renderer 故意打坏 → 任务只能 PARTIAL/
        NEEDS_REPAIR；最终 payload 不得宣称任务完整完成。"""
        service, mission = await _setup_ur5e(tmp_path)
        # 目标含"仿真视频"——spec 冻结 required scene_video；
        # 显式打坏 3D 场景渲染链（R0-3 后链存在，故障必须诚实
        # 表达为 PARTIAL，不是整体成功也不是整体失败）。
        from rosclaw.agentd import sim_render

        original = sim_render.render_scene_trace

        def broken(*a, **k):
            raise ValueError("RENDER_BACKEND_UNAVAILABLE: 注入故障")

        sim_render.render_scene_trace = broken  # type: ignore[assignment]
        kernel = service._task_kernel
        kernel.persist_input(
            mission_id=mission.mission_id, session_ref="pi_1",
            message_id="msg_video",
            text="画五角星并做仿真视频",
        )
        lease = await _issue_lease(service, mission)
        try:
            result = await PiToolDispatcher(service).execute(
                _request(
                    "rosclaw_task", mission=mission.mission_id,
                    idem="r02_gate", lease=lease,
                    arguments={
                        "goal": "draw_shape",
                        "parameters": {"shape": "star5",
                                       "center_m": [0.35, 0.25, 0.30],
                                       "scale_m": 0.10},
                    },
                )
            )
        finally:
            sim_render.render_scene_trace = original  # type: ignore[assignment]
        payload = json.loads(result.summary)
        assert payload["state"] != "VERIFIED", payload
        assert any(
            "scene_video" in str(f) for f in payload.get("failures", [])
        ), payload
        view = str(payload.get("user_view", ""))
        assert "完成" not in view or "部分" in view or "未" in view, view
        task = kernel.latest_task_for(mission.mission_id, "pi_1")
        assert task["state"] != "SUCCEEDED"
        await service.close()

    async def test_plain_draw_still_succeeds(self, tmp_path: Path) -> None:
        """回归：无媒体/场景要求的 draw 任务仍完整 SUCCEEDED。"""
        service, mission = await _setup_ur5e(tmp_path)
        lease = await _issue_lease(service, mission)
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_task", mission=mission.mission_id,
                idem="r02_plain", lease=lease,
                arguments={
                    "goal": "draw_shape",
                    "parameters": {"shape": "star5"},
                },
            )
        )
        assert result.ok, result.summary
        payload = json.loads(result.summary)
        assert payload["state"] == "VERIFIED", payload
        await service.close()
