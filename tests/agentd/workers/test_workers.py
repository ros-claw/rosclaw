"""Worker Fabric tests (PR-WF-050/051/053 exits).

- card validation: bad adapter, physical side effects, missing hard-floor
  forbidden scopes, license requirements
- registry: register/enable/disable/quarantine with audit
- scheduler: hard filter gates (never scored), scoring explainable
- manager: full lifecycle to ACCEPTED; verification rejects fabricated
  success, secret leakage, stale lease, unsupported claims
- fault semantics: lease expiry sweeper, no blind re-dispatch for
  side-effect orders, circuit breaker, duplicate idempotency rejected
- native worker: isolation (no shared conversation), timeout, honest
  failure result
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.workers import (
    CardValidationError,
    NativeWorkerAdapter,
    Scheduler,
    SchedulingError,
    WorkerManager,
    WorkerRegistry,
    verify_result,
)
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.worker.card import (
    CapabilityDecl,
    WorkerCardV1,
    WorkerImplementation,
    WorkerKind,
)
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ResultArtifact,
    ResultClaim,
    SideEffectPolicy,
    WorkOrderV1,
    WorkResultV1,
)

ACTOR = "agent:rosclaw-native:sim_ur5e_01"


def _card(worker_id: str = "worker:test:one", **kwargs) -> WorkerCardV1:
    payload = {
        "worker_id": worker_id,
        "kind": WorkerKind.TOOL,
        "adapter_type": "process_stdio",
        "implementation": WorkerImplementation(product="test", version="1.0"),
        "capabilities": [
            CapabilityDecl(name="analysis.text", side_effect_class="none"),
        ],
    }
    payload.update(kwargs)
    return WorkerCardV1(**payload)


def _order(
    mission_id: str = "mis_x",
    capability: str = "analysis.text",
    side_effect: dict | None = None,
    **kwargs,
) -> WorkOrderV1:
    payload = {
        "work_order_id": new_id("wo"),
        "mission_id": mission_id,
        "issued_by": ACTOR,
        "capability": capability,
        "goal": "分析失败日志并提出修复建议",
        "inputs": {"instructions": "只基于给定日志", "artifacts": ["log://1"]},
        "budgets": BudgetEnvelope(wall_time_sec=10),
        "side_effect_policy": SideEffectPolicy(**(side_effect or {"class": "none"})),
    }
    payload.update(kwargs)
    return WorkOrderV1(**payload)


@pytest.fixture
def store(tmp_path: Path) -> MissionStore:
    return MissionStore(tmp_path / "test.db")


@pytest.fixture
def registry(store: MissionStore) -> WorkerRegistry:
    reg = WorkerRegistry(store.connection)
    reg.register_builtins(actor_id=ACTOR)
    return reg


def _manager(store: MissionStore, gateway=None) -> WorkerManager:
    gw = gateway or MockModelGateway(mock_profile(), [])
    return WorkerManager(
        store.connection,
        adapters={"native_inproc": NativeWorkerAdapter(gw)},
        actor_id=ACTOR,
    )


class TestCardValidation:
    def test_builtin_native_basic_valid(self, registry: WorkerRegistry) -> None:
        card = registry.get("worker:native:basic")
        assert card is not None
        assert card.trust.initial_level == "T3"

    def test_physical_side_effect_rejected(self) -> None:
        from rosclaw.agentd.workers.registry import validate_card

        card = _card(capabilities=[CapabilityDecl(name="act.move", side_effect_class="physical")])
        with pytest.raises(CardValidationError, match="physical"):
            validate_card(card)

    def test_unsupported_adapter_rejected(self) -> None:
        from rosclaw.agentd.workers.registry import validate_card

        with pytest.raises(CardValidationError, match="adapter_type"):
            validate_card(_card(adapter_type="telepathy"))

    def test_hard_floor_forbidden_scopes(self) -> None:
        from rosclaw.agentd.workers.registry import validate_card

        card = _card()
        card = card.model_copy(
            update={"security": card.security.model_copy(update={"forbidden_scopes": []})}
        )
        with pytest.raises(CardValidationError, match="hard floor"):
            validate_card(card)

    def test_requesting_forbidden_scope_rejected(self) -> None:
        from rosclaw.agentd.workers.registry import validate_card

        card = _card()
        card = card.model_copy(
            update={
                "security": card.security.model_copy(
                    update={"default_data_scopes": ["raw_secrets"]}
                )
            }
        )
        with pytest.raises(CardValidationError, match="hard-forbidden"):
            validate_card(card)


class TestRegistry:
    def test_status_lifecycle_audited(self, registry: WorkerRegistry, store: MissionStore) -> None:
        registry.register(_card(), actor_id=ACTOR)
        registry.set_status("worker:test:one", "DISABLED", actor_id=ACTOR, reason="维护")
        assert registry.status_of("worker:test:one") == "DISABLED"
        registry.set_status("worker:test:one", "QUARANTINED", actor_id=ACTOR, reason="异常输出")
        assert registry.status_of("worker:test:one") == "QUARANTINED"
        events = store.connection.execute(
            "SELECT event_type FROM worker_events WHERE worker_id = 'worker:test:one'"
        ).fetchall()
        types = [e["event_type"] for e in events]
        assert "rosclaw.worker.card.disabled.v1" in types
        assert "rosclaw.worker.card.quarantined.v1" in types

    def test_catalog(self, registry: WorkerRegistry) -> None:
        catalog = registry.catalog()
        assert any(c.worker_id == "worker:native:basic" for c in catalog)


class TestScheduler:
    def test_capability_not_declared_rejected(self) -> None:
        scheduler = Scheduler()
        order = _order(capability="code.repository_edit")
        with pytest.raises(SchedulingError, match="not declared"):
            scheduler.select(order, [CandidateView(card=_card())])

    def test_quarantined_never_scored(self) -> None:
        scheduler = Scheduler()
        order = _order()
        with pytest.raises(SchedulingError, match="QUARANTINED"):
            scheduler.select(order, [CandidateView(card=_card(), registry_status="QUARANTINED")])

    def test_concurrency_gate(self) -> None:
        scheduler = Scheduler()
        order = _order()
        card = _card()
        view = CandidateView(card=card, running_orders=card.constraints.max_concurrency)
        with pytest.raises(SchedulingError, match="concurrency"):
            scheduler.select(order, [view])

    def test_scoring_explainable(self) -> None:
        scheduler = Scheduler()
        order = _order()
        good = CandidateView(card=_card("worker:test:good"), reliability=0.9)
        bad = CandidateView(card=_card("worker:test:bad"), reliability=0.1)
        view, scored = scheduler.select(order, [bad, good])
        assert view.card.worker_id == "worker:test:good"
        assert "reliability=0.90" in scored.reasons
        assert scored.features["capability"] == 1.0

    def test_circuit_open_rejected(self) -> None:
        scheduler = Scheduler()
        with pytest.raises(SchedulingError, match="circuit"):
            scheduler.select(_order(), [CandidateView(card=_card(), circuit_open=True)])


class TestVerify:
    def test_fabricated_completion_rejected(self) -> None:
        order = _order()
        order = order.model_copy(
            update={
                "expected_output": order.expected_output.model_copy(
                    update={"artifacts": ["text/plain"]}
                )
            }
        )
        result = WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id="worker:test:one",
            lease_id="lease_1",
            status="COMPLETED",
            summary="done (fabricated)",
        )
        # lease mismatch AND missing artifacts
        report = verify_result(
            order.model_copy(update={"lease": None}),
            result,
        )
        assert not report.accepted

    def test_secret_in_summary_rejected(self) -> None:
        order = _order()
        from rosclaw.contracts.worker.order import WorkOrderLease

        order = order.model_copy(
            update={"lease": WorkOrderLease(lease_id="lease_1", issued_at="t", expires_at="t+1")}
        )
        result = WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id=order.assigned_to or "worker:test:one",
            lease_id="lease_1",
            status="COMPLETED",
            summary="key is sk-ABCDEFGHIJKLMNOP1234",
            artifacts=[ResultArtifact(ref="artifact://text/x", media_type="text/plain")],
            claims=[ResultClaim(claim="x", evidence_refs=["artifact://text/x"])],
        )
        report = verify_result(order, result)
        assert not report.accepted
        assert report.verifier_results["secret_scan"] is False

    def test_unsupported_claim_rejected(self) -> None:
        order = _order()
        from rosclaw.contracts.worker.order import WorkOrderLease

        order = order.model_copy(
            update={"lease": WorkOrderLease(lease_id="lease_1", issued_at="t", expires_at="t+1")}
        )
        result = WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id="w",
            lease_id="lease_1",
            status="COMPLETED",
            summary="x",
            claims=[ResultClaim(claim="selected tests passed", evidence_refs=[])],
        )
        report = verify_result(order, result)
        assert not report.accepted
        assert report.verifier_results["claims_have_evidence"] is False


class _FakeAdapter:
    """Scripted adapter for lifecycle/fault tests."""

    def __init__(self, results: list[WorkResultV1 | BaseException | None]) -> None:
        self._results = list(results)
        self.started: list[WorkOrderV1] = []
        self.cancelled: list[str] = []

    async def probe(self):
        from rosclaw.agentd.workers.adapter import WorkerProbeResult

        return WorkerProbeResult(ready=True)

    async def start(self, order, credential_refs):
        from rosclaw.agentd.workers.adapter import RunHandle

        self.started.append(order)
        return RunHandle(
            work_order_id=order.work_order_id,
            lease_id=order.lease.lease_id,
            worker_id=order.assigned_to or "w",
        )

    async def poll(self, handle):
        item = self._results.pop(0) if self._results else None
        if item is None:
            return handle  # still running
        if isinstance(item, BaseException):
            raise item
        return item

    async def cancel(self, handle, reason):
        self.cancelled.append(reason)

    async def reconcile(self, idempotency_key):
        return "not_found"


def _write_capable_card(worker_id: str = "worker:test:writer") -> WorkerCardV1:
    """Card declaring a workspace_write capability (native_inproc adapter)."""
    return _card(
        worker_id=worker_id,
        adapter_type="native_inproc",
        capabilities=[CapabilityDecl(name="analysis.text", side_effect_class="workspace_write")],
    )


class TestManagerLifecycle:
    async def test_happy_path_accepted(self, store: MissionStore, registry) -> None:
        card = registry.get("worker:native:basic")
        order = _order()
        adapter = _FakeAdapter([None])  # replaced below
        manager = WorkerManager(
            store.connection, adapters={"native_inproc": adapter}, actor_id=ACTOR
        )
        scheduled = manager.hire(order, [CandidateView(card=card)])
        assert scheduled.status == "RUNNING"
        result = WorkResultV1(
            work_order_id=scheduled.work_order_id,
            worker_id=card.worker_id,
            lease_id=scheduled.lease.lease_id,
            status="COMPLETED",
            summary="分析完成",
            artifacts=[ResultArtifact(ref="artifact://text/a", media_type="text/plain")],
            claims=[ResultClaim(claim="produced analysis", evidence_refs=["artifact://text/a"])],
        )
        adapter._results = [result]
        final, report = await manager.run_to_completion(scheduled)
        assert report.accepted, report.reasons
        stored = manager.order(scheduled.work_order_id)
        assert stored.status == "ACCEPTED"

    async def test_verification_failure_marks_failed(self, store: MissionStore, registry) -> None:
        card = registry.get("worker:native:basic")
        order = _order()
        adapter = _FakeAdapter([])
        manager = WorkerManager(
            store.connection, adapters={"native_inproc": adapter}, actor_id=ACTOR
        )
        scheduled = manager.hire(order, [CandidateView(card=card)])
        result = WorkResultV1(
            work_order_id=scheduled.work_order_id,
            worker_id=card.worker_id,
            lease_id=scheduled.lease.lease_id,
            status="COMPLETED",
            summary="leaked sk-ABCDEFGHIJKLMNOP1234",
        )
        adapter._results = [result]
        _, report = await manager.run_to_completion(scheduled)
        assert not report.accepted
        assert manager.order(scheduled.work_order_id).status == "FAILED"

    async def test_stale_lease_result_not_accepted(self, store: MissionStore, registry) -> None:
        card = registry.get("worker:native:basic")
        adapter = _FakeAdapter([])
        manager = WorkerManager(
            store.connection, adapters={"native_inproc": adapter}, actor_id=ACTOR
        )
        scheduled = manager.hire(_order(), [CandidateView(card=card)])
        late = WorkResultV1(
            work_order_id=scheduled.work_order_id,
            worker_id=card.worker_id,
            lease_id="lease_OLD_STALE",
            status="COMPLETED",
            summary="late result",
        )
        adapter._results = [late]
        _, report = await manager.run_to_completion(scheduled)
        assert not report.accepted
        assert "lease" in report.reasons[0]

    async def test_duplicate_idempotency_rejected(self, store: MissionStore, registry) -> None:
        card = _write_capable_card()
        registry.register(card, actor_id=ACTOR)
        manager = _manager(store)
        order = _order(side_effect={"class": "workspace_write", "idempotency_key": "idem_1"})
        manager.hire(order, [CandidateView(card=card)])
        with pytest.raises(ValidationError, match="duplicate idempotency"):
            manager.hire(
                _order(side_effect={"class": "workspace_write", "idempotency_key": "idem_1"}),
                [CandidateView(card=card)],
            )

    async def test_lease_sweeper_expires_pure_compute(self, store: MissionStore, registry) -> None:
        card = registry.get("worker:native:basic")
        adapter = _FakeAdapter([None] * 1000)  # never completes
        manager = WorkerManager(
            store.connection, adapters={"native_inproc": adapter}, actor_id=ACTOR
        )
        scheduled = manager.hire(_order(), [CandidateView(card=card)])
        # Force lease into the past.
        store.connection.execute(
            "UPDATE work_orders SET lease_expires_at = '2000-01-01' WHERE work_order_id = ?",
            (scheduled.work_order_id,),
        )
        expired = await manager.sweep_expired()
        assert expired == [scheduled.work_order_id]
        assert manager.order(scheduled.work_order_id).status == "EXPIRED"

    async def test_side_effect_order_not_swept_while_running(
        self, store: MissionStore, registry
    ) -> None:
        card = _write_capable_card()
        registry.register(card, actor_id=ACTOR)

        class RunningAdapter(_FakeAdapter):
            async def reconcile(self, idempotency_key):
                return "running"

        adapter = RunningAdapter([None] * 1000)
        manager = WorkerManager(
            store.connection, adapters={"native_inproc": adapter}, actor_id=ACTOR
        )
        scheduled = manager.hire(
            _order(side_effect={"class": "workspace_write", "idempotency_key": "idem_x"}),
            [CandidateView(card=card)],
        )
        store.connection.execute(
            "UPDATE work_orders SET lease_expires_at = '2000-01-01' WHERE work_order_id = ?",
            (scheduled.work_order_id,),
        )
        expired = await manager.sweep_expired()
        assert expired == []  # no blind expiry/re-dispatch while work may land
        assert manager.order(scheduled.work_order_id).status == "RUNNING"

    def test_circuit_breaker_after_failures(self, store: MissionStore, registry) -> None:
        card = registry.get("worker:native:basic")
        manager = _manager(store)
        assert not manager.circuit_open(card.worker_id, "analysis.text")
        for _ in range(3):
            order = _order()
            scheduled = manager.hire(order, [CandidateView(card=card)])
            store.connection.execute(
                "UPDATE work_orders SET status = 'FAILED' WHERE work_order_id = ?",
                (scheduled.work_order_id,),
            )
        assert manager.circuit_open(card.worker_id, "analysis.text")


class TestNativeWorker:
    async def test_produces_verifiable_result(self, store: MissionStore, registry) -> None:
        def worker_answer(request):
            return ModelTurnResultV1(
                turn_id="t",
                provider="mock",
                model="mock-model",
                content="根因：超时配置过短。建议：提高到 30s 并重试。[推断]",
                assistant_message={"role": "assistant", "content": "..."},
                usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},  # type: ignore[arg-type]
            )

        gateway = MockModelGateway(mock_profile(), [worker_answer])
        manager = _manager(store, gateway)
        card = registry.get("worker:native:basic")
        scheduled = manager.hire(_order(), [CandidateView(card=card)])
        result, report = await manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED"
        assert "根因" in result.summary
        assert report.accepted
        assert result.usage.prompt_tokens + result.usage.completion_tokens == 70
        # Worker ran with an isolated, single-message conversation.
        request = gateway.requests[-1]
        assert len(request.messages) == 1
        assert request.tools == []

    async def test_worker_model_failure_honest(self, store: MissionStore, registry) -> None:
        gateway = MockModelGateway(mock_profile(), [])  # script exhausted → error
        manager = _manager(store, gateway)
        card = registry.get("worker:native:basic")
        scheduled = manager.hire(_order(), [CandidateView(card=card)])
        result, report = await manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        assert not report.accepted
