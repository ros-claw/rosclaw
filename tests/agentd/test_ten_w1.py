"""十审 Gate W1 红测试：内置 Pi Worker 最小闭环。

红测试先行——以下在修复前必须红：

1. 内置 Worker 注册为 worker:rosclaw:pi（pi_managed adapter），默认
   ENABLED（node+dist 可用时）；probe 不依赖外部 CLI 安装。
2. envelope 无 secret（worker_cannot_serialize_credentials）：快照只含
   provider/model/thinking；携带凭据字段直接 AdapterError。
3. delegate 可通过 auto/worker:rosclaw:pi 调度到内置 Worker（registry
   有卡 + adapter 注册）。
4. Worker profile 不含任何 rosclaw_* 工具（worker_profile_excludes_
   rosclaw_action_tools）。
5. 端到端：内置 Worker 子进程真实跑通（需要本机 node+dist+模型配置，
   无则 skip——fake headless 脚本覆盖协议面）。
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestBuiltinWorkerRegistration:
    async def test_pi_worker_registered_and_schedulable(self, tmp_path: Path) -> None:
        service, _mission = await _setup(tmp_path)
        card = service._registry.get("worker:rosclaw:pi")
        assert card is not None, "内置 Pi Worker 未注册"
        assert card.adapter_type == "pi_managed"
        assert "pi_managed" in service._worker_manager._adapters
        names = {c.name for c in card.capabilities}
        assert "analysis.text" in names
        await service.close()

    async def test_pi_worker_declares_no_rosclaw_action_tools(self, tmp_path: Path) -> None:
        """内置 Worker 的工具面不得含 ROSClaw 物理/动作工具。"""
        service, _mission = await _setup(tmp_path)
        card = service._registry.get("worker:rosclaw:pi")
        assert card is not None
        for cap in card.capabilities:
            assert not cap.name.startswith("rosclaw."), cap.name
            assert cap.side_effect_class != "physical"
        await service.close()


class TestEnvelopeNoSecrets:
    async def test_envelope_carries_no_credentials(self, tmp_path: Path) -> None:
        from rosclaw.agentd.workers.pi_managed import PiManagedAdapter
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderLease,
            WorkOrderV1,
        )

        adapter = PiManagedAdapter(rosclaw_home=tmp_path)
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by="test",
            capability="analysis.text",
            goal="g",
            inputs={
                "instructions": "i",
                "model_snapshot": {
                    "provider": "kimi-for-coding",
                    "model": "kimi-for-coding",
                    "thinking": "high",
                },
            },
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
            lease=WorkOrderLease(lease_id="lease_1", issued_at="t", expires_at="t"),
        )
        path, _cwd = adapter._write_envelope(order)
        raw = path.read_text()
        assert "sk-" not in raw
        assert "api_key" not in raw.lower()
        envelope = json.loads(raw)
        assert envelope["model"]["provider"] == "kimi-for-coding"
        # 权限 0600
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, oct(mode)

    async def test_envelope_rejects_credential_fields(self, tmp_path: Path) -> None:
        from rosclaw.agentd.workers.adapter import AdapterError
        from rosclaw.agentd.workers.pi_managed import PiManagedAdapter
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderLease,
            WorkOrderV1,
        )

        adapter = PiManagedAdapter(rosclaw_home=tmp_path)
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by="test",
            capability="analysis.text",
            goal="g",
            inputs={
                "model_snapshot": {"provider": "p", "model": "m", "api_key": "sk-evil"},
            },
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
            lease=WorkOrderLease(lease_id="lease_1", issued_at="t", expires_at="t"),
        )
        with pytest.raises(AdapterError):
            adapter._write_envelope(order)


class TestHeadlessWorkerProtocol:
    """fake headless 脚本驱动协议面（不依赖真实模型）。"""

    def _fake_node(self, tmp_path: Path, script_body: str) -> tuple[str, str]:
        """返回 (node, entry)——node 是 sh 包装，entry 是脚本路径。"""
        script = tmp_path / "fake-entry.js"
        script.write_text(script_body)
        script.chmod(0o755)
        return "/bin/sh", str(script)

    async def test_streaming_events_drive_heartbeat_and_result(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """逐行 JSONL：attempt_started → tool_started → attempt_finished
        推进 progress_seq（真实心跳），结果进入 WorkResultV1。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        fake = tmp_path / "fake-entry"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "sleep 0.2\n"
            'echo \'{"kind":"tool_started","tool":"read"}\'\n'
            "sleep 0.2\n"
            'echo \'{"kind":"usage","input_tokens":100,"output_tokens":20}\'\n'
            'echo \'{"kind":"attempt_finished","report":"看到了三个文件"}\'\n'
        )
        fake.chmod(0o755)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        service._worker_manager._adapters["pi_managed"] = adapter

        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        card = service._registry.get("worker:rosclaw:pi")
        assert card is not None
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="code.repository_analysis",
            goal="看仓库",
            inputs={"instructions": "x", "workspace": str(tmp_path)},
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        result, report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED", result.summary
        assert "看到了三个文件" in result.summary
        assert report.accepted, report.reasons
        assert result.usage.prompt_tokens == 100
        await service.close()

    async def test_silent_worker_fails_startup_timeout(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """无 attempt_started 的挂死 Worker 必须诚实失败（不无限 Working）。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "STARTUP_TIMEOUT_SEC", 1.0)
        fake = tmp_path / "fake-silent"
        fake.write_text("#!/bin/sh\nsleep 30\n")
        fake.chmod(0o755)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        service._worker_manager._adapters["pi_managed"] = adapter

        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        card = service._registry.get("worker:rosclaw:pi")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="analysis.text",
            goal="x",
            inputs={"instructions": "x"},
            budgets=BudgetEnvelope(wall_time_sec=60, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        result, report = await service._worker_manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        assert "startup timeout" in result.summary or "exited" in result.summary
        assert not report.accepted
        await service.close()

    async def test_cancel_kills_pi_worker_process_group(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """cancel 整组杀（含 Worker fork 的孙进程）。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.workers import pi_managed

        pgid_file = tmp_path / "pgid.txt"
        fake = tmp_path / "fake-hang"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            f"echo $$ > {pgid_file}\n"
            "sleep 300 &\n"
            "sleep 300\n"
        )
        fake.chmod(0o755)
        monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
        adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
        service._worker_manager._adapters["pi_managed"] = adapter

        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        card = service._registry.get("worker:rosclaw:pi")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="analysis.text",
            goal="x",
            inputs={"instructions": "x"},
            budgets=BudgetEnvelope(wall_time_sec=300, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = service._worker_manager.hire(
            order,
            [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                           circuit_open=False)],
        )
        driver = asyncio.create_task(service._worker_manager.run_to_completion(scheduled))
        for _ in range(200):
            if pgid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pgid_file.exists(), "fake worker 未启动"
        pgid = int(pgid_file.read_text().strip())
        os.killpg(pgid, 0)
        await service._worker_manager.cancel_order(scheduled.work_order_id, reason="test")
        for _ in range(70):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                break
            await asyncio.sleep(0.1)
        with pytest.raises((ProcessLookupError, PermissionError)):
            os.killpg(pgid, 0)
        current = service._worker_manager.order(scheduled.work_order_id)
        assert current is not None and current.status == "CANCELLED"
        await asyncio.wait_for(driver, 10)
        await service.close()


class TestDelegateRoutesToBuiltinWorker:
    async def test_delegate_accepts_pi_worker_hint(self, tmp_path: Path) -> None:
        """worker_id=worker:rosclaw:pi 的调度行为必须诚实：
        node+dist 可用 → hire 成功（STARTED）；不可用 → 诚实拒绝
        （WORKER_UNAVAILABLE/SCHEDULING_FAILED），绝不假装能跑。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w1_route",
                arguments={
                    "goal": "协议验证",
                    "worker_id": "worker:rosclaw:pi",
                    "sync_grace_sec": 0,
                },
            )
        )
        enabled = service._registry.status_of("worker:rosclaw:pi") == "ENABLED"
        if enabled:
            assert result.ok, result.summary
            assert result.status == "STARTED"
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            assert orders and orders[0].assigned_to == "worker:rosclaw:pi"
            await dispatcher.execute(
                _request(
                    "rosclaw_cancel_work",
                    mission=mission.mission_id,
                    idem="idem_w1_route_c",
                    arguments={"work_order_id": orders[0].work_order_id},
                )
            )
        else:
            assert not result.ok
            assert result.error_code in {"WORKER_UNAVAILABLE", "SCHEDULING_FAILED"}
        await service.close()
