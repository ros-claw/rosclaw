"""十一审 PR-A 真实长任务验收（Gate B/D）：真实 Kimi K3、>5 分钟、
高 thinking 长静默不误杀。

无 provider key 时诚实 skip。运行：
ROSCLAW_KIMI_API_KEY=sk-kimi-... pytest tests/agentd/test_eleven_a_live.py -s
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def _has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


@pytest.mark.skipif(not _has_key(), reason="无真实 provider key——诚实 skip")
class TestLiveLongTaskNoFalseKill:
    async def test_long_repo_analysis_survives_past_60s(self, tmp_path: Path) -> None:
        """长任务（仓库级分析）必须活过旧的 60s 误杀点；liveness 事件
        持续；终态真实（ACCEPTED 或带明确 error_code 的 FAILED）。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        repo = Path(__file__).resolve().parents[2]
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_11a_live",
                arguments={
                    "goal": "通读 src/rosclaw/agentd 下的主要模块（service、"
                    "workers、pi_bridge、context），写一份详细的中文架构报告："
                    "每个模块的职责、关键类、模块间数据流。报告至少 2000 字。",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "scout",
                    "workspace": str(repo),
                    "budget": {"wall_time_sec": 900, "model_tokens": 300_000},
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.ok, started.summary
        assert started.status == "STARTED"
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        # 活过旧误杀点（60s）是硬断言；此后等到终态（最长 16 分钟）。
        await asyncio.sleep(65)
        mid = service._worker_manager.order(order.work_order_id)
        assert mid is not None and mid.status in ("RUNNING", "ACCEPTED"), (
            f"60s 处被误杀/异常终态: {mid.status if mid else None}"
        )
        final = None
        deadline = time.monotonic() + 960
        while time.monotonic() < deadline:
            current = service._worker_manager.order(order.work_order_id)
            if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED", "EXPIRED"):
                final = current
                break
            await asyncio.sleep(2)
        assert final is not None, "16 分钟未到终态"
        if final.status == "FAILED":
            row = service._store.connection.execute(
                "SELECT verify_report_json FROM work_orders WHERE work_order_id = ?",
                (order.work_order_id,),
            ).fetchone()
            pytest.fail(f"长任务 FAILED: {row['verify_report_json'] if row else '?'}")
        assert final.status == "ACCEPTED"
        await service.close()
