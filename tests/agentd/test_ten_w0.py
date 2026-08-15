"""十审 Gate W0 红测试（P0-WORKER-BLOCK / P0-CAPABILITY-LIE / cancel 闭环 /
P0-ORDER-CORRELATION / P1-PRODUCT-NOISE）。

红测试先行——以下缺陷修复前必须红：

1. rosclaw_delegate 不得同步阻塞父会话超过短阈值：慢 Worker 时必须在
   grace 内返回 STARTED + 精确 WorkOrder ID + worker + 预算 + deadline。
2. rosclaw_check_work 按精确 work_order_id 关联（不再取 mission 最后
   一单）；终态返回真实结果。
3. rosclaw_cancel_work → WorkOrder CANCELLED + adapter cancel + 外部
   进程整组 SIGTERM/SIGKILL（无孤儿进程）。
4. driver 崩溃不得让 WorkOrder 永久 RUNNING（标记 FAILED）。
5. 能力诚实：side_effect_class=none 的卡不得声明 write/edit/delete/
   execute 类能力名；官方 pack 不得含 docs.write。
6. SIM MCP server 默认不在 stderr 泄漏 INFO 请求日志。
"""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


def _slow_adapter_module():
    from rosclaw.agentd.workers.adapter import RunHandle

    class SlowStubAdapter:
        """poll 永不完成的 Worker（直到 release 事件置位）。"""

        def __init__(self) -> None:
            self.cancelled: list[tuple[str, str]] = []
            self.release = asyncio.Event()

        async def probe(self, worker_id=None):  # noqa: ARG002
            from rosclaw.agentd.workers.adapter import WorkerProbeResult

            return WorkerProbeResult(ready=True, detail="stub")

        async def start(self, order, credential_refs):  # noqa: ARG002
            handle = RunHandle(
                work_order_id=order.work_order_id,
                lease_id=order.lease.lease_id if order.lease else "",
                worker_id=order.assigned_to or "worker:stub:slow",
            )
            task = asyncio.create_task(self._run(order, handle))
            self._tasks = getattr(self, "_tasks", {})
            self._tasks[order.work_order_id] = (handle, task)
            return handle

        async def _run(self, order, handle):
            from rosclaw.contracts.worker.order import (
                ResultArtifact,
                ResultClaim,
                WorkResultV1,
            )

            await self.release.wait()
            ref = "artifact://text/sha256:stub"
            return WorkResultV1(
                work_order_id=order.work_order_id,
                worker_id=handle.worker_id,
                lease_id=handle.lease_id,
                status="COMPLETED",
                summary="slow stub finished",
                artifacts=[
                    ResultArtifact(ref=ref, media_type="text/plain", digest="sha256:stub")
                ],
                claims=[ResultClaim(claim="stub done", evidence_refs=[ref])],
            )

        async def poll(self, handle):
            entry = getattr(self, "_tasks", {}).get(handle.work_order_id)
            if entry is None:
                from rosclaw.agentd.workers.adapter import AdapterError

                raise AdapterError(f"unknown handle {handle.work_order_id}")
            stored, task = entry
            if not task.done():
                return stored
            return task.result()

        async def cancel(self, handle, reason: str) -> None:
            self.cancelled.append((handle.work_order_id, reason))
            entry = getattr(self, "_tasks", {}).pop(handle.work_order_id, None)
            if entry is not None:
                entry[1].cancel()

        async def reconcile(self, idempotency_key: str) -> str:  # noqa: ARG002
            return "not_found"

    return SlowStubAdapter


def _crash_adapter_class():
    from rosclaw.agentd.workers.adapter import AdapterError

    class CrashStubAdapter:
        async def probe(self, worker_id=None):  # noqa: ARG002
            from rosclaw.agentd.workers.adapter import WorkerProbeResult

            return WorkerProbeResult(ready=True, detail="stub")

        async def start(self, order, credential_refs):  # noqa: ARG002
            raise AdapterError("spawn blew up")

        async def poll(self, handle):  # pragma: no cover - never reached
            raise AdapterError("n/a")

        async def cancel(self, handle, reason: str) -> None:  # noqa: ARG002
            return None

        async def reconcile(self, idempotency_key: str) -> str:  # noqa: ARG002
            return "not_found"

    return CrashStubAdapter


def _register_stub(service, adapter, *, worker_id: str, adapter_type: str):
    from rosclaw.contracts.worker.card import (
        CapabilityDecl,
        WorkerCardV1,
        WorkerConstraints,
        WorkerHealth,
        WorkerImplementation,
        WorkerKind,
        WorkerProvenance,
        WorkerSecurity,
        WorkerTrust,
    )

    card = WorkerCardV1(
        worker_id=worker_id,
        display_name="Stub Worker",
        kind=WorkerKind.HARNESS,
        adapter_type=adapter_type,
        adapter_version="1.0.0",
        implementation=WorkerImplementation(
            product="stub", version="1.0.0", executable_ref="inproc:"
        ),
        capabilities=[
            CapabilityDecl(
                name="analysis.text",
                input_schema="rosclaw://schemas/text-task.v1",
                output_schema="rosclaw://schemas/text-result.v1",
                side_effect_class="none",
            )
        ],
        constraints=WorkerConstraints(supported_platforms=["linux"], max_concurrency=4),
        security=WorkerSecurity(isolation="process"),
        health=WorkerHealth(probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=3600),
        provenance=WorkerProvenance(source="test", license="MIT"),
        trust=WorkerTrust(initial_level="T3", evidence_count=0),
    )
    service._registry.register(card, actor_id="test")
    service._worker_manager._adapters[adapter_type] = adapter
    return card


class TestNonBlockingDelegate:
    async def test_delegate_slow_worker_returns_started_with_full_identity(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        stub = _slow_adapter_module()()
        _register_stub(
            service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio"
        )
        dispatcher = PiToolDispatcher(service)
        started = time.monotonic()
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_slow",
                arguments={"goal": "跑一个长任务", "worker_id": "worker:stub:slow"},
            )
        )
        elapsed = time.monotonic() - started
        # 不得同步等到 Worker 完成（stub 永不完成）——短阈值内返回。
        assert elapsed < 30, f"delegate 阻塞了 {elapsed:.1f}s"
        assert result.ok, result.summary
        assert result.status == "STARTED", f"慢 Worker 应返回 STARTED: {result.status}"
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        assert len(orders) == 1
        order = orders[0]
        # 第一屏：精确 WorkOrder ID + worker + 预算（十四审 PR-14.7：
        # wall 是 soft target 提醒阈值——不再显示误导性 "Deadline"）。
        assert order.work_order_id in result.summary
        assert "worker:stub:slow" in result.summary
        assert str(order.budgets.wall_time_sec) in result.summary
        assert "提醒阈值" in result.summary or "预计" in result.summary
        assert "Deadline" not in result.summary
        await service.close()

    async def test_fast_worker_still_completes_within_grace(self, tmp_path: Path) -> None:
        """快任务保持同步返回 COMPLETED（不在本轮回归）。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_fast",
                arguments={"goal": "总结这段日志", "worker_id": "auto"},
            )
        )
        assert result.ok and result.status == "COMPLETED"
        await service.close()

    async def test_background_driver_eventually_completes_order(self, tmp_path: Path) -> None:
        """STARTED 后后台 driver 必须真正驱动到终态（结果可经 check_work 读到）。"""
        service, mission = await _setup(tmp_path)
        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_bg",
                arguments={"goal": "长任务", "worker_id": "worker:stub:slow"},
            )
        )
        assert result.status == "STARTED"
        order = service._worker_manager.orders_for_mission(mission.mission_id)[0]
        stub.release.set()
        # 后台 driver 驱动到终态。
        for _ in range(200):
            current = service._worker_manager.order(order.work_order_id)
            if current and current.status in ("ACCEPTED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        current = service._worker_manager.order(order.work_order_id)
        assert current is not None and current.status == "ACCEPTED", current.status
        await service.close()


class TestCheckWork:
    async def test_check_work_correlates_exact_order(self, tmp_path: Path) -> None:
        """P0-ORDER-CORRELATION：查询绑定精确 ID，不取 mission 最后一单。"""
        service, mission = await _setup(tmp_path)
        stub1 = _slow_adapter_module()()
        _register_stub(service, stub1, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        first = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_corr1",
                arguments={"goal": "第一单", "worker_id": "worker:stub:slow"},
            )
        )
        second = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_corr2",
                arguments={"goal": "第二单", "worker_id": "worker:stub:slow"},
            )
        )
        assert first.status == "STARTED" and second.status == "STARTED"
        orders = service._worker_manager.orders_for_mission(mission.mission_id)
        wo1, wo2 = orders[0].work_order_id, orders[1].work_order_id
        check = await dispatcher.execute(
            _request(
                "rosclaw_check_work",
                mission=mission.mission_id,
                idem="idem_w0_chk1",
                arguments={"work_order_id": wo1},
            )
        )
        assert check.ok, check.summary
        assert wo1 in check.summary
        assert wo2 not in check.summary
        await service.close()

    async def test_check_work_unknown_id_honest(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_check_work",
                mission=mission.mission_id,
                idem="idem_w0_chk404",
                arguments={"work_order_id": "wo_nonexistent"},
            )
        )
        assert not result.ok
        assert result.error_code == "WORK_ORDER_NOT_FOUND"
        await service.close()

    async def test_check_work_cross_mission_refused(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        other = service.create_mission("别的 mission")
        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_xm",
                arguments={"goal": "x", "worker_id": "worker:stub:slow"},
            )
        )
        assert started.status == "STARTED"
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        # 用 other mission 的绑定请求查本 mission 的单——拒绝。
        # （_request 固定绑 pi_1→mission；直接换 mission_id 会先撞
        # MISSION_MISMATCH——这也算拒绝。）
        result = await dispatcher.execute(
            _request(
                "rosclaw_check_work",
                mission=other.mission_id,
                idem="idem_w0_xm2",
                arguments={"work_order_id": wo},
            )
        )
        assert not result.ok
        await service.close()


class TestCancelWork:
    async def test_cancel_transitions_order_and_stops_driver(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        stub = _slow_adapter_module()()
        _register_stub(service, stub, worker_id="worker:stub:slow", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        started = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_cancel",
                arguments={"goal": "长任务", "worker_id": "worker:stub:slow"},
            )
        )
        assert started.status == "STARTED"
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        cancelled = await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w0_cancel2",
                arguments={"work_order_id": wo, "reason": "user_abort"},
            )
        )
        assert cancelled.ok, cancelled.summary
        assert "CANCELLED" in cancelled.summary or "取消" in cancelled.summary
        # adapter cancel 被真实调用（handle 级）。
        assert stub.cancelled and stub.cancelled[0][0] == wo
        # DB 终态 CANCELLED；后台 driver 随之退出且不改写终态。
        for _ in range(100):
            current = service._worker_manager.order(wo)
            if current and current.status == "CANCELLED":
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        current = service._worker_manager.order(wo)
        assert current is not None and current.status == "CANCELLED", current.status
        await service.close()

    async def test_cancel_terminal_order_is_honest_noop(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        done = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_ct",
                arguments={"goal": "快任务", "worker_id": "auto"},
            )
        )
        assert done.status == "COMPLETED"
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        result = await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_w0_ct2",
                arguments={"work_order_id": wo},
            )
        )
        # 已终态：诚实返回当前状态，不报错也不翻转。
        assert result.ok
        current = service._worker_manager.order(wo)
        assert current is not None and current.status == "ACCEPTED"
        await service.close()

    async def test_cancel_kills_external_process_group(self, tmp_path: Path) -> None:
        """外部 harness cancel 必须杀整个进程组（含孙进程），无孤儿。"""
        from rosclaw.agentd.workers.external import ExternalHarnessAdapter
        from rosclaw.agentd.workers.packs import WorkerPackManifest

        script = tmp_path / "fake-harness"
        pgid_file = tmp_path / "pgid.txt"
        script.write_text(
            "#!/bin/sh\n"
            f"echo $$ > {pgid_file}\n"
            "sleep 300 &\n"  # 孙进程——cancel 必须连带杀掉
            "sleep 300\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        pack = WorkerPackManifest(
            pack_id="fake",
            worker_id="worker:fake:local",
            product="codex-cli",  # 复用其命令构造（参数被脚本忽略）
            display_name="Fake",
            executable=str(script),
            min_version="0.0.0",
            install_hint="",
            license="MIT",
            capabilities=(("analysis.text", "rosclaw://schemas/text-task.v1"),),
        )
        adapter = ExternalHarnessAdapter(cwd=tmp_path)
        adapter._packs = {pack.worker_id: pack}

        service, mission = await _setup(tmp_path)
        # 用测试 adapter 替换 service 的 external_cli 实现（指向 fake 脚本）。
        service._worker_manager._adapters["external_cli"] = adapter
        _register_stub_card = None  # 直接走 manager
        from rosclaw.contracts.common import new_id

        # 注册 fake 卡 + adapter。
        from rosclaw.contracts.worker.card import (
            CapabilityDecl,
            WorkerCardV1,
            WorkerConstraints,
            WorkerHealth,
            WorkerImplementation,
            WorkerKind,
            WorkerProvenance,
            WorkerSecurity,
            WorkerTrust,
        )
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        card = WorkerCardV1(
            worker_id=pack.worker_id,
            display_name="Fake",
            kind=WorkerKind.HARNESS,
            adapter_type="external_cli",
            adapter_version="1.0.0",
            implementation=WorkerImplementation(
                product="codex-cli", version="0", executable_ref=f"path:{script}"
            ),
            capabilities=[
                CapabilityDecl(name="analysis.text", side_effect_class="none")
            ],
            constraints=WorkerConstraints(supported_platforms=["linux"], max_concurrency=1),
            security=WorkerSecurity(isolation="process"),
            health=WorkerHealth(
                probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=3600
            ),
            provenance=WorkerProvenance(source="test", license="MIT"),
            trust=WorkerTrust(initial_level="T3", evidence_count=0),
        )
        service._registry.register(card, actor_id="test")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=mission.mission_id,
            issued_by="test",
            capability="analysis.text",
            goal="sleep forever",
            inputs={"instructions": "x"},
            budgets=BudgetEnvelope(wall_time_sec=300, model_tokens=1000),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        from rosclaw.agentd.workers.scheduler import CandidateView

        scheduled = service._worker_manager.hire(
            order,
            [
                CandidateView(
                    card=card, registry_status="ENABLED", running_orders=0, circuit_open=False
                )
            ],
        )
        driver = asyncio.create_task(service._worker_manager.run_to_completion(scheduled))
        # 等子进程起来。
        for _ in range(200):
            if pgid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pgid_file.exists(), "fake harness 未启动"
        pgid = int(pgid_file.read_text().strip())
        # 进程组活着（shell + 两个 sleep）。
        os.killpg(pgid, 0)
        t0 = time.monotonic()
        await service._worker_manager.cancel_order(
            scheduled.work_order_id, reason="test_abort"
        )
        # cancel 后进程组必须整体消失（2s 级，硬上限 7s）。
        deadline = time.monotonic() + 7
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                break
            await asyncio.sleep(0.1)
        elapsed = time.monotonic() - t0
        with pytest.raises((ProcessLookupError, PermissionError)):
            os.killpg(pgid, 0)
        assert elapsed < 7, f"进程组清理耗时 {elapsed:.1f}s"
        current = service._worker_manager.order(scheduled.work_order_id)
        assert current is not None and current.status == "CANCELLED", current.status
        await asyncio.wait_for(driver, 10)
        await service.close()


class TestDriverCrashSafety:
    async def test_driver_crash_marks_failed_not_running_forever(
        self, tmp_path: Path
    ) -> None:
        service, mission = await _setup(tmp_path)
        crash = _crash_adapter_class()()
        _register_stub(service, crash, worker_id="worker:stub:crash", adapter_type="process_stdio")
        dispatcher = PiToolDispatcher(service)
        await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_w0_crash",
                arguments={"goal": "x", "worker_id": "worker:stub:crash"},
            )
        )
        # 启动即崩：可能同步 FAILED，也可能 STARTED 后后台转 FAILED——
        # 关键是不得永久 RUNNING。
        wo = service._worker_manager.orders_for_mission(mission.mission_id)[0].work_order_id
        for _ in range(200):
            current = service._worker_manager.order(wo)
            if current and current.status in ("FAILED", "ACCEPTED", "CANCELLED", "EXPIRED"):
                break
            await asyncio.sleep(0.05)
        current = service._worker_manager.order(wo)
        assert current is not None and current.status == "FAILED", current.status
        await service.close()


class TestCapabilityHonesty:
    def test_registry_rejects_write_named_capability_with_none_side_effect(self) -> None:
        """capability_registry_rejects_docs_write_without_write_profile。"""
        from rosclaw.agentd.workers.registry import CardValidationError, validate_card
        from rosclaw.contracts.worker.card import (
            CapabilityDecl,
            WorkerCardV1,
            WorkerConstraints,
            WorkerHealth,
            WorkerImplementation,
            WorkerKind,
            WorkerProvenance,
            WorkerSecurity,
            WorkerTrust,
        )

        card = WorkerCardV1(
            worker_id="worker:fake:writer",
            display_name="Fake Writer",
            kind=WorkerKind.HARNESS,
            adapter_type="external_cli",
            adapter_version="1.0.0",
            implementation=WorkerImplementation(
                product="fake", version="1", executable_ref="path:fake"
            ),
            capabilities=[
                CapabilityDecl(name="docs.write", side_effect_class="none"),
            ],
            constraints=WorkerConstraints(supported_platforms=["linux"], max_concurrency=1),
            security=WorkerSecurity(isolation="process"),
            health=WorkerHealth(probe="adapter:ping", heartbeat_interval_sec=15, lease_ttl_sec=60),
            provenance=WorkerProvenance(source="test", license="MIT"),
            trust=WorkerTrust(initial_level="T1", evidence_count=0),
        )
        with pytest.raises(CardValidationError):
            validate_card(card)

    def test_official_packs_declare_no_fake_write_capability(self) -> None:
        from rosclaw.agentd.workers.packs import ALL_PACKS

        implying = {"write", "edit", "delete", "execute", "install", "promote"}
        for pack in ALL_PACKS:
            for name, _schema in pack.capabilities:
                tail = name.rsplit(".", 1)[-1]
                assert tail not in implying, (
                    f"{pack.pack_id} 仍声明副作用能力 {name}（side_effect_class=none 下虚假）"
                )


class TestStartupNoise:
    def test_sim_mcp_servers_default_quiet(self) -> None:
        """SIM MCP server 默认 log_level WARNING——INFO 请求日志（如
        Processing request of type ListToolsRequest）不得泄漏到终端。"""
        from rosclaw.limo.sim_mcp import server as limo_server
        from rosclaw.sim.ur5e_mcp import server as ur5e_server

        assert ur5e_server.settings.log_level.upper() in ("WARNING", "ERROR")
        assert limo_server.settings.log_level.upper() in ("WARNING", "ERROR")
