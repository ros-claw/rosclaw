"""十二审自审轮 live 验收（真实 Kimi K3，无 key 诚实 skip）：

1. resume 闭环：真实 Worker 完成短任务 → /job resume 恢复同一 Pi
   会话（session_resumed 事件实证）；
2. artifact_build 闭环：Worker 用 workbench bash+PIL 真产 GIF →
   deliverable 真校验通过 → ACCEPTED + 媒体工件在册。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def _has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


async def _wait_terminal(service, wo_id: str, timeout: float = 420.0):
    for _ in range(int(timeout * 2)):
        current = service._worker_manager.order(wo_id)
        if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED", "EXPIRED"):
            return current
        await asyncio.sleep(0.5)
    return None


@pytest.mark.skipif(not _has_key(), reason="无真实 provider key——诚实 skip")
class TestLiveResumeAndMedia:
    async def test_resume_same_session_real(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_12r_live1",
                arguments={
                    "goal": "记住数字 42，然后只回答 OK",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "scout",
                    "budget": {"wall_time_sec": 180, "model_tokens": 20000},
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.status == "STARTED", started.summary
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        final = await _wait_terminal(service, order.work_order_id, 240)
        assert final is not None and final.status == "ACCEPTED", final.status if final else "timeout"
        resumed = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_12r_live2",
                arguments={"work_order_id": order.work_order_id},
            )
        )
        assert resumed.ok, resumed.summary
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        second = orders[1]
        assert second.inputs.get("_resume_session"), "resume 未携带 session 检查点"
        assert Path(second.inputs["_resume_session"]).exists(), "session 文件不存在"
        # resume 单跑到终态（Worker 原任务已完成——恢复会话后通常直接
        # 报告/确认；我们只验证链路跑通且不崩）。
        final2 = await _wait_terminal(service, second.work_order_id, 240)
        assert final2 is not None, "resume 单未收敛"
        # session_resumed 事件实证同一 Pi 会话被恢复。
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        events = WorkerEventStore(tmp_path).tail(second.work_order_id)
        kinds = {e["kind"] for e in events}
        assert "session_resumed" in kinds, f"未恢复同一 session: {kinds}"
        await service.close()

    async def test_real_gif_deliverable_accepted(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        ws = tmp_path / "ws"
        ws.mkdir()
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_12g_live",
                arguments={
                    "goal": "用 bash 运行 python3 写一个 PIL 脚本：生成一个 3 帧的"
                    " GIF 动画（每帧一个不同颜色的方块），保存为 anim.gif，"
                    "然后用 python3 验证它能被 PIL 打开并报告帧数",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "sim-builder",
                    "task_type": "artifact_build",
                    "deliverables": [{"media_types": ["image/gif"], "required": True}],
                    "workspace": str(ws),
                    "budget": {"wall_time_sec": 300, "model_tokens": 80000},
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.status == "STARTED", started.summary
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        final = await _wait_terminal(service, order.work_order_id, 360)
        assert final is not None, "6 分钟未到终态"
        assert final.status == "ACCEPTED", f"status={final.status}"
        # 真实 GIF 工件在册且可解码。
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        work_dir = WorkerEventStore(tmp_path).dir_of(order.work_order_id)
        gifs = list(work_dir.glob("workspace/**/*.gif"))
        assert gifs, "无 GIF 产出"
        from PIL import Image

        img = Image.open(gifs[0])
        assert img.n_frames >= 3
        img.close()
        await service.close()
