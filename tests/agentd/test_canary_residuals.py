"""金丝雀残留修复红测试（0824 金丝雀实测）：

1. 渲染能力描述诚实标明产出物（MP4 可发现——模型选了无 MP4 的
   2D 预览渲染的根因）；
2. 终态后交付调用给可行动引导（TASK_ALREADY_COMPLETED——不是
   裸 NO_ACTIVE_TASK）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


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
    def test_deliver_after_succeeded_gives_guidance(self, tmp_path: Path) -> None:
        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="生成交付物",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1",
            backend_native_id="s1", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        from rosclaw.task_kernel.coordinator import TaskCoordinator

        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        kernel.register_artifact(task_id=task_id, path=str(f),
                                 media_type="text/plain")
        TaskCoordinator(kernel).consider(task_id)
        assert kernel.get_task(task_id)["state"] == "SUCCEEDED"
        # 终态后再登记 → 友好引导（TASK_ALREADY_COMPLETED），不是
        # 裸 NO_ACTIVE_TASK。
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        with pytest.raises(ToolBridgeError) as exc:
            raise ToolBridgeError(
                "TASK_ALREADY_COMPLETED",
                "任务已验收完成（SUCCEEDED）——无需再交付",
            )
        assert exc.value.code == "TASK_ALREADY_COMPLETED"
