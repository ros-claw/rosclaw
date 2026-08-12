"""十审 W1 真实模型闭环：delegate → 内置 Pi Worker（Kimi K3）→ 验证采纳。

无 provider key 时诚实 skip（绝不假绿）。运行：
ROSCLAW_KIMI_API_KEY=sk-kimi-... pytest tests/agentd/test_ten_w1_live.py
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


@pytest.mark.skipif(not _has_key(), reason="无真实 provider key——诚实 skip")
class TestLiveBuiltinWorker:
    async def test_delegate_to_pi_worker_real_model(self, tmp_path: Path) -> None:
        """同一 Kimi 配置（env passthrough/auth.json，不配置第二份 key）：

        delegate STARTED → 后台驱动 → check_work ACCEPTED + 真实报告。
        """
        # 给 scout 一个真实可读的 workspace。
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "alpha.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "ws" / "beta.txt").write_text("beta", encoding="utf-8")
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w1_live",
                arguments={
                    "goal": f"列出 {tmp_path}/ws 目录里的文件名",
                    "instructions": "用 ls 工具列出目录，报告看到的文件名",
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "scout",
                    "budget": {"wall_time_sec": 180, "model_tokens": 30000},
                    "sync_grace_sec": 0,
                    # delegate 的 inputs 没有 workspace 字段——用 goal 带路径
                    # （W3 Workbench 才引入正式 workspace 注入）。
                },
            )
        )
        assert result_ok(started), started.summary
        assert started.status == "STARTED"
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        assert order.assigned_to == "worker:rosclaw:pi"
        # 后台驱动到终态（真实模型 2 分钟内）。
        final = None
        for _ in range(240):
            current = service._worker_manager.order(order.work_order_id)
            if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED", "EXPIRED"):
                final = current
                break
            await asyncio.sleep(0.5)
        assert final is not None, "Worker 两分钟内未到终态"
        assert final.status == "ACCEPTED", f"status={final.status}"
        # check_work 读到真实结果。
        check = await dispatcher.execute(
            _request(
                "rosclaw_check_work",
                mission=mission.mission_id,
                idem="idem_w1_live_chk",
                arguments={"work_order_id": order.work_order_id},
            )
        )
        assert check.ok
        assert check.status == "COMPLETED"
        assert check.summary.strip(), "空报告"
        await service.close()


def result_ok(result) -> bool:
    return result.ok
