"""R0-1.5 红测试（金丝雀 4/10 实证 + 0826 审计 §5.R0-1 收口）：
输入路由自动执行——已知 recipe 零模型调用。

金丝雀实证（真实 K3，/tmp/star-canary-1787797367）：
- run8：模型绕链手拼 capability（零 rosclaw_task 调用，425s
  超时）——任务级入口在场不等于模型会选它；
- run2/run6：模型传 plane='xz'/'vertical' → recipe 拒绝
  （只支持 xy）→ 模型降级手拼 → 证据链缺失 → WAITING_INPUT；
- run5：失败后模型重试 3 次 + grep 源码调试。

闭环断言：
1. recipe 平面支持：plane="xz"/"yz" 命名面 + plane_normal_xyz
   直通（竖直平面不再拒绝）；
2. 路由输入编译：NL → shape/plane 参数（通用词类标记，无形状
   特例 prompt 硬编码）；疑问句（怎么画/如何/吗/？）不自动执行；
3. 服务端自动路由：pi.input.persist 画路径指令 → auto_task
   （后台执行）→ 任务终态 SUCCEEDED + plan.node 事件 + 产物
   登记——零模型工具调用；
4. watcher trackTask：plan.node 进度 widget + 终态一次
   followUp（不重复、不泄漏进 progress 上下文）；
5. rosclaw_task 退出模型面（MODEL_TOOL_NAMES 不含它）——
   wire 层 adapter 保留兼容。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestRecipePlaneSupport:
    def test_named_vertical_plane_accepted(self, tmp_path: Path) -> None:
        """plane="xz"（竖直命名面）端到端跑通——不再拒绝。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.0, 0.30], scale_m=0.10,
            plane="xz",
        )
        rollout = sim.simulate_cartesian_trajectory(plan["plan_id"])
        assert rollout["ok"] is True

    def test_yz_named_plane_accepted(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.0, 0.25, 0.30], scale_m=0.10,
            plane="yz",
        )
        assert plan["plan_id"]

    def test_unknown_plane_still_rejected(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        sim = SimTrajectoryService(tmp_path)
        with pytest.raises(ValueError, match="unsupported plane"):
            sim.generate_planar_path(
                shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.10,
                plane="diagonal",
            )


class TestRouteInputCompilation:
    def test_vertical_plane_from_nl(self) -> None:
        from rosclaw.task_kernel.task_router import compile_recipe_inputs

        inputs = compile_recipe_inputs(
            "让仿真 UR5e 在竖直平面画一个五角星，并给我 GIF 和 MP4"
        )
        assert inputs.get("plane") in ("xz", "yz") or inputs.get(
            "plane_normal_xyz"
        ), inputs

    def test_shape_from_nl(self) -> None:
        from rosclaw.task_kernel.task_router import compile_recipe_inputs

        assert compile_recipe_inputs("画一个圆").get("shape") == "circle"
        assert compile_recipe_inputs("画五角星").get("shape") == "star5"

    def test_question_form_not_auto_routed(self) -> None:
        """疑问句不自动执行（怎么画/如何/吗/？ 是讨论不是指令）。"""
        from rosclaw.task_kernel.task_router import is_task_directive

        assert is_task_directive("画一个五角星") is True
        assert is_task_directive("怎么画五角星？") is False
        assert is_task_directive("如何画一个五角星") is False
        assert is_task_directive("你会画五角星吗") is False


class TestServerAutoRoute:
    async def test_persist_draw_directive_auto_executes(
        self, tmp_path: Path
    ) -> None:
        """pi.input.persist 画路径指令 → 服务端自动路由执行（后台
        线程）→ 零模型工具调用 → 任务 SUCCEEDED + plan.node 事件
        + GIF/MP4 产物。"""
        import asyncio

        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": "msg_auto_1",
                "text": "画一个五角星，给我 GIF 和 MP4",
            },
        )
        assert result.get("ok"), result
        auto = result.get("auto_task")
        assert auto, f"未自动路由：{result}"
        assert auto.get("recipe_id") == "recipe:sim.draw_path"
        # 后台执行——等终态（真实 sim 链）。
        kernel = service._task_kernel
        task_id = str(auto["task_id"])
        deadline = asyncio.get_event_loop().time() + 180
        while asyncio.get_event_loop().time() < deadline:
            task = kernel.get_task(task_id)
            if task and task["state"] in ("SUCCEEDED", "FAILED", "REPAIR_REQUIRED"):
                break
            await asyncio.sleep(2)
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", task["state"]
        events = kernel._conn.execute(
            "SELECT event_type FROM task_events WHERE task_id = ? "
            "AND event_type LIKE 'plan.node_%'",
            (task_id,),
        ).fetchall()
        assert len(events) >= 10, f"plan node 事件不完整：{len(events)}"
        artifacts = kernel.artifact_refs_for(task_id)
        media = {a["media_type"] for a in artifacts}
        assert "image/gif" in media and "video/mp4" in media, media
        # 零模型工具调用（自动路由不产生 tool_used）。
        used = kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND event_type = 'task.tool_used'",
            (task_id,),
        ).fetchone()
        assert int(used["n"]) == 0, used["n"]
        await service.close()

    async def test_question_not_auto_routed(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": "msg_q",
                "text": "怎么画五角星？",
            },
        )
        assert result.get("ok")
        assert not result.get("auto_task"), result
        await service.close()


class TestModelSurfaceRemoval:
    def test_rosclaw_task_not_in_model_surface(self) -> None:
        """rosclaw_task 退出模型面（删除清单落地）——wire 层
        adapter 保留兼容，模型默认看不到。"""
        surface = (
            Path(__file__).resolve().parents[2]
            / "packages/rosclaw-agent/src/tools/surface.ts"
        ).read_text(encoding="utf-8")
        import re

        packs = re.findall(
            r"(?:MODEL_TOOL_NAMES|EMBODIMENT_PACK|WORKSPACE_PACK|"
            r"PRODUCT_PACK)[^=]*=\s*\[([\s\S]*?)\];",
            surface,
        )
        assert packs, "模型面包未找到"
        assert not any('"rosclaw_task"' in pack for pack in packs), (
            "rosclaw_task 仍在模型面——已知 recipe 已由输入路由自动执行"
        )
