"""金丝雀残留修复红测试（0824 金丝雀实测）：

1. 渲染能力描述诚实标明产出物（MP4 可发现——模型选了无 MP4 的
   2D 预览渲染的根因）；
2. 终态后交付调用的语义（W05 §9.2 起）：追加交付登记在既有
   revision 并带可行动引导——不再 TASK_ALREADY_COMPLETED 拒绝。
"""

from __future__ import annotations

from pathlib import Path


def _descriptor(tool_id: str):
    from rosclaw.agentd.tooling.catalog import ToolCatalog
    from rosclaw.agentd.tooling.native_tools import register_native_tools
    from rosclaw.agentd.tools import BuiltinToolRegistry

    catalog = ToolCatalog()
    register_native_tools(
        catalog, BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e")
    )
    return catalog.get(tool_id)


class TestRenderDiscoverability:
    def test_scene_render_description_mentions_mp4(self) -> None:
        d = _descriptor("simulation_render_scene")
        assert d is not None
        assert "MP4" in d.description.upper(), "场景渲染描述未提 MP4——不可发现"

    def test_2d_render_description_honest_no_mp4(self) -> None:
        d = _descriptor("simulation_render_trace")
        assert d is not None
        assert "MP4" in d.description.upper() and "no MP4" in d.description, (
            "2D 预览渲染描述未诚实区分（应指明无 MP4 并指向场景渲染）"
        )


class TestCompletedTaskGuidance:
    async def test_deliver_after_succeeded_appends_with_guidance(
        self, tmp_path: Path
    ) -> None:
        """W05 §9.2：终态后登记 = 追加交付（带引导），不是裸错误
        也不是 TASK_ALREADY_COMPLETED 拒绝。"""
        from tests.agentd.test_pi_tool_bridge import (
            _issue_lease,
            _request,
            _setup,
        )
        from tests.agentd.test_w05_delivery_lifecycle import _succeeded_task

        service, mission = await _setup(tmp_path)
        await _succeeded_task(service, mission, tmp_path)
        f2 = tmp_path / "b.txt"
        f2.write_text("appendix", encoding="utf-8")
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_artifact_register", mission=mission.mission_id,
                idem="canary_append",
                lease=await _issue_lease(service, mission),
                arguments={"path": str(f2)},
            )
        )
        assert result.ok, result.summary
        assert "追加交付" in result.summary
        assert "用户有新目标时会开始新任务" in result.summary
        await service.close()
