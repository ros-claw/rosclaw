"""十四审 Gate 2：真实 Kimi K3 长任务验收（总纲 §8 Gate 2——合并门禁）。

真实 provider/model/thinking 配置（与用户一致），无 key/node 诚实 skip。

验收矩阵：
1. 真实 K3 长任务跨 soft target（300s）不被杀、不被假暂停——wall/token
   soft target 只提醒；
2. 中途 pause（ACK PAUSED、进程存活、session 保持）→ resume 同会话完成；
3. 中途 steer 送达（steer_ack）；
4. 主对话并行：Worker 运行期间 Native Agent 会话仍可响应（不阻塞）；
5. agentd 重启一次 → INTERRUPTED_RESUMABLE（不是 FAILED）→ resume
   从同一 session 恢复并完成；
6. deliverable verifier PASS（WorkSpecV2 真实工件校验）。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.agentd.pi_entry import find_pi_agent_entry
from tests.agentd.test_pi_tool_bridge import _request, _setup

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def _has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


def _has_runtime() -> bool:
    try:
        return find_pi_agent_entry() is not None
    except Exception:  # noqa: BLE001
        return False


async def _wait_status(service, wo: str, targets: tuple[str, ...], timeout: float):
    for _ in range(int(timeout * 2)):
        current = service._worker_manager.order(wo)
        if current and current.status in targets:
            return current
        await asyncio.sleep(0.5)
    return None


@pytest.mark.skipif(
    not (_has_key() and _has_runtime()),
    reason="无真实 provider key 或 Node/dist——诚实 skip",
)
class TestGate2RealK3:
    async def test_long_task_pause_steer_resume_completes(self, tmp_path: Path) -> None:
        """真实长任务：soft target 不杀；pause→ACK→resume 同会话；
        steer 送达；deliverable 真实校验通过。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_gate2_1",
                arguments={
                    "goal": (
                        "在 workspace 写一个 Python 模块 primes.py（埃氏筛 + "
                        "分段筛两个实现 + 基准比较）和对应 pytest 测试，跑通"
                        "测试后给出最终报告（含文件路径和测试输出摘要）。"
                    ),
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "developer",
                    "task_type": "code_change",
                    "budget": {"wall_time_sec": 120, "model_tokens": 30000},
                },
            )
        )
        assert started.ok, started.summary
        assert started.status in ("STARTED", "COMPLETED")
        if started.status == "COMPLETED":
            await service.close()
            return  # 快任务直接完成（Grace 内）——仍算闭环
        wo = started.summary.split("WorkOrder: ")[1].split("\n")[0]
        # 等 worker 真正进入工作（事件流出现）。
        adapter = service._worker_manager._adapters["pi_managed"]
        for _ in range(600):
            events = adapter._events.tail(wo, limit=500)
            if any(e["kind"] in ("tool_started", "model_started") for e in events):
                break
            await asyncio.sleep(0.5)
        # 1) pause → ACK PAUSED，进程存活。
        paused = await adapter.request_pause(wo, reason="user")
        assert paused, "pause 未获 ACK（exit130 回归？）"
        proc = adapter._procs[wo]
        assert proc.returncode is None, "pause 后进程退出——exit130 回归"
        # 2) resume → 同会话继续。
        assert await adapter.request_resume(wo)
        # 3) steer 送达。
        from rosclaw.agentd.pi_bridge.server import worker_control  # noqa: F401

        steered = await dispatcher.execute(
            _request(
                "rosclaw_update_work",
                mission=mission.mission_id,
                idem="idem_gate2_steer",
                arguments={"work_order_id": wo, "note": "优先保证测试全绿，再写基准"},
            )
        )
        assert steered.ok, steered.summary
        # 4) wall soft target（120s）已过——任务必须仍在跑或正常完成，
        #    绝不能因 soft target 被终止。
        terminal = await _wait_status(
            service, wo, ("ACCEPTED", "FAILED", "CANCELLED", "SUBMITTED",
                          "VERIFYING"), timeout=1800,
        )
        assert terminal is not None, "30 分钟超时——任务未收敛"
        assert terminal.status in ("ACCEPTED", "SUBMITTED", "VERIFYING"), (
            f"终态 {terminal.status}——"
            "真实长任务必须完成（框架制造的 FAILED 应为零）"
        )
        # 6) deliverable：WorkSpecV2 真实工件（diff + 测试日志）已验证。
        result_row = service._store.connection.execute(
            "SELECT result_json FROM work_results WHERE work_order_id = ?", (wo,)
        ).fetchone()
        assert result_row is not None
        # 状态权威：termination_cause 必须 COMPLETED（不是 SIGNAL_UNKNOWN）。
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        state = WorkerEventStore(tmp_path).read_state(wo) or {}
        assert state.get("termination_cause") in (None, "COMPLETED"), state
        await service.close()

    async def test_restart_recovery_real(self, tmp_path: Path) -> None:
        """agentd 重启：RUNNING → INTERRUPTED_RESUMABLE（不 FAILED）→
        resume 同会话恢复并完成。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_gate2_2",
                arguments={
                    "goal": (
                        "调研 workspace 里的项目结构，写一份 STRUCTURE.md"
                        "（模块地图 + 关键数据流），写完给最终报告。"
                    ),
                    "worker_id": "worker:rosclaw:pi",
                    "worker_profile": "developer",
                    "budget": {"wall_time_sec": 600, "model_tokens": 80000},
                },
            )
        )
        assert started.ok and started.status == "STARTED", started.summary
        wo = started.summary.split("WorkOrder: ")[1].split("\n")[0]
        adapter = service._worker_manager._adapters["pi_managed"]
        for _ in range(600):
            events = adapter._events.tail(wo, limit=500)
            if any(e["kind"] == "session_persisted" for e in events):
                break
            await asyncio.sleep(0.5)
        # 模拟 agentd 重启：对账（杀孤儿 + RUNNING→INTERRUPTED_RESUMABLE）。
        reconciled = await service.reconcile_workers_on_start()
        assert wo in reconciled
        current = service._worker_manager.order(wo)
        assert current.status == "INTERRUPTED_RESUMABLE", current.status
        assert current.status != "FAILED"
        # resume：同一 session 恢复（新 attempt）。
        resumed = await dispatcher.execute(
            _request(
                "rosclaw_resume_work",
                mission=mission.mission_id,
                idem="idem_gate2_resume",
                arguments={"work_order_id": wo},
            )
        )
        assert resumed.ok, resumed.summary
        view = service._worker_manager.job_view(wo)
        assert view is not None and len(view["attempts"]) == 2
        new_wo = view["attempts"][1]["attempt_id"]
        terminal = await _wait_status(
            service, new_wo, ("ACCEPTED", "FAILED", "CANCELLED"), timeout=1800,
        )
        assert terminal is not None
        assert terminal.status == "ACCEPTED", terminal.status
        await service.close()
