"""十一审 PR-E 红测试：恢复/partial handoff/证据语言。

红测试先行——修复前必须红：
1. 基础设施错误自动重试至多一次（fingerprint 匹配），带 lineage +
   _reuse_workspace；第二次同指纹不再重试；
2. 语义失败（验证拒绝）不自动重试；
3. WAITING_INPUT 全链：worker 提问 → BLOCKED → rosclaw_answer_work
   送达 → RUNNING → 答案进最终结果；
4. 非 ROSClaw work 目录的 _reuse_workspace 被安全闸拒绝；
5. 证据三层文案（COMMAND_REPLAY 只显示路径预演）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _fake(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _order(mission_id: str, tmp_path: Path) -> WorkOrderV1:
    return WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id=mission_id,
        issued_by="test",
        capability="analysis.text",
        goal="x",
        inputs={"instructions": "x"},
        budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )


class TestAutoRetry:
    async def test_infra_failure_auto_retries_once_with_lineage(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = _fake(tmp_path, "fake-crash", "#!/bin/sh\nexit 1\n")
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        card = service._registry.get("worker:rosclaw:pi")
        order = _order(mission.mission_id, tmp_path)
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        service.spawn_worker_driver(scheduled)
        # 等自动重试链收敛（原单 FAILED + retry 单也 FAILED——共 2 单）。
        for _ in range(400):
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            if len(orders) >= 2 and all(
                o.status in ("FAILED", "ACCEPTED", "CANCELLED", "EXPIRED") for o in orders
            ):
                break
            await asyncio.sleep(0.05)
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 2, f"自动重试次数错误: {len(orders)}"
        first, second = orders[0], orders[1]
        assert first.status == "FAILED"
        assert second.parent_work_order_id == first.work_order_id
        assert second.inputs.get("_auto_retried") is True
        assert "基础设施错误" in second.inputs["instructions"]
        # 第二单同指纹——不再产生第三单。
        await asyncio.sleep(0.5)
        assert len(service._worker_manager.orders_for_mission(mission.mission_id)) == 2
        await service.close()

    async def test_semantic_failure_not_auto_retried(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """验证拒绝（fabricated 完成）不是基础设施错误——不重试。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        # fabricated：COMPLETED 但无工件 → verifier 拒绝。
        fake = _fake(
            tmp_path,
            "fake-fabricate",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"attempt_finished","report":"leaked sk-1234567890abcdef"}\'\n',
        )
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        card = service._registry.get("worker:rosclaw:pi")
        order = _order(mission.mission_id, tmp_path)
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        service.spawn_worker_driver(scheduled)
        for _ in range(200):
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            if orders and orders[0].status in ("FAILED", "ACCEPTED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 1, f"语义失败不应自动重试: {len(orders)}"
        assert orders[0].status == "FAILED"  # secret scan 拒绝
        await service.close()


class TestWaitingInput:
    async def test_waiting_input_answer_roundtrip(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = _fake(
            tmp_path,
            "fake-ask",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            'echo \'{"kind":"waiting_input","question":"用哪个目录?"}\'\n'
            "while IFS= read -r line; do\n"
            "  case \"$line\" in\n"
            "    *answer*)\n"
            '      echo \'{"kind":"answer_received"}\'\n'
            '      echo \'{"kind":"attempt_finished","report":"已按回答继续"}\'\n'
            "      exit 0;;\n"
            "  esac\n"
            "done\n",
        )
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
            )
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_11e_wait",
                arguments={
                    "goal": "需要提问的任务",
                    "worker_id": "worker:rosclaw:pi",
                    "sync_grace_sec": 0,
                },
            )
        )
        assert started.status == "STARTED"
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        # 等 BLOCKED（waiting_input 驱动）。
        for _ in range(200):
            current = service._worker_manager.order(wo)
            if current and current.status == "BLOCKED":
                break
            await asyncio.sleep(0.05)
        current = service._worker_manager.order(wo)
        assert current is not None and current.status == "BLOCKED", current.status
        # 回答 → RUNNING → 完成。
        answered = await dispatcher.execute(
            _request(
                "rosclaw_answer_work",
                mission=mission.mission_id,
                idem="idem_11e_ans",
                arguments={"work_order_id": wo, "text": "用 src/ 目录"},
            )
        )
        assert answered.ok, answered.summary
        for _ in range(200):
            current = service._worker_manager.order(wo)
            if current and current.status in ("ACCEPTED", "FAILED"):
                break
            await asyncio.sleep(0.05)
        current = service._worker_manager.order(wo)
        assert current is not None and current.status == "ACCEPTED", current.status
        await service.close()


class TestReuseGuard:
    async def test_reuse_workspace_outside_rosclaw_work_refused(
        self, tmp_path: Path
    ) -> None:
        """_reuse_workspace 指向主仓库等外部路径 → 安全闸忽略（新建）。"""
        from rosclaw.agentd.workers.pi_managed import PiManagedAdapter

        adapter = PiManagedAdapter(rosclaw_home=tmp_path / "rh")
        external = tmp_path / "user-repo"
        external.mkdir()
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=external, check=True)
        (external / "f.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=external, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=external,
            check=True,
        )
        order = _order("mis_x", tmp_path)
        order = order.model_copy(
            update={
                "inputs": {
                    "worker_profile": "developer",
                    "workspace": str(external),
                    "_reuse_workspace": str(external),  # 恶意/错误复用
                }
            }
        )
        workspace, _ = await adapter._prepare_workspace(order)
        # 绝不能直接返回用户主仓库——必须是 ROSClaw work 下的新 worktree。
        assert workspace != str(external)
        assert str(Path(workspace)).startswith(str(tmp_path / "rh" / "work"))


class TestEvidenceLanguage:
    def test_three_tier_labels(self) -> None:
        from rosclaw.agentd.task_runner import evidence_user_label

        assert "路径预演" in evidence_user_label("COMMAND_REPLAY")
        assert "动力学仿真" in evidence_user_label("SIM_DYN_ROLLOUT")
        assert "真机" in evidence_user_label("REAL_RECEIPT")
        assert "仿真完成" not in evidence_user_label("COMMAND_REPLAY")
