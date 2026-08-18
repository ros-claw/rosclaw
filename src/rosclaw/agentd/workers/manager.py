"""WorkerManager — WorkOrder lifecycle, leases, reconciliation (PR-WF-051).

Dual-track lifecycle (ADR-0003):
run track  DRAFT → OFFERED → CLAIMED → RUNNING → SUBMITTED → VERIFYING → ACCEPTED
                                          ↘ BLOCKED / FAILED / EXPIRED / CANCELLED
lease track heartbeat → SUSPECT → EXPIRED.

Failure semantics:
- heartbeat/lease timeout never triggers blind re-dispatch of a side-effect
  order; reconciliation (adapter journal / idempotency record) comes first;
- results submitted under a stale lease are recorded late/stale, not accepted;
- consecutive failures open a per (worker, capability) circuit breaker;
- every transition is journaled to worker_events with actor + idempotency.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.agentd.workers.adapter import RunHandle, WorkerAdapter
from rosclaw.agentd.workers.scheduler import CandidateView, Scheduler, SchedulingError
from rosclaw.agentd.workers.verify import VerificationReport, verify_result
from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.worker.card import WorkerCardV1
from rosclaw.contracts.worker.order import (
    WorkOrderLease,
    WorkOrderV1,
    WorkResultV1,
)

_RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    "DRAFT": frozenset({"OFFERED", "CANCELLED"}),
    "OFFERED": frozenset({"CLAIMED", "CANCELLED", "EXPIRED"}),
    "CLAIMED": frozenset({"RUNNING", "CANCELLED", "EXPIRED"}),
    # 十三审 HOTFIX-13.2：UNREACHABLE（失联探测中）/INTERRUPTED_RESUMABLE
    # （进程死亡但可恢复）/BUDGET_PAUSED（预算暂停待 extend）不再是
    # FAILED 的替身。
    "RUNNING": frozenset({
        "SUBMITTED", "FAILED", "EXPIRED", "CANCELLED", "BLOCKED",
        "UNREACHABLE", "INTERRUPTED_RESUMABLE", "BUDGET_PAUSED",
        "PAUSE_REQUESTED", "PAUSED",
    }),
    "SUBMITTED": frozenset({"VERIFYING", "FAILED"}),
    "VERIFYING": frozenset({"ACCEPTED", "FAILED", "BLOCKED"}),
    "BLOCKED": frozenset(
        {"RUNNING", "CANCELLED", "FAILED", "INTERRUPTED_RESUMABLE"}
    ),
    "UNREACHABLE": frozenset(
        {"RUNNING", "INTERRUPTED_RESUMABLE", "FAILED", "CANCELLED"}
    ),
    "INTERRUPTED_RESUMABLE": frozenset({"RUNNING", "FAILED", "CANCELLED"}),
    "BUDGET_PAUSED": frozenset({"RUNNING", "CANCELLED", "FAILED"}),
    # 十四审：PAUSE_REQUESTED 只在 ACK 前存在——ACK 后 PAUSED/BUDGET_PAUSED，
    # ACK 失败回 RUNNING；用户取消任何时刻合法；agentd 重启可落
    # INTERRUPTED_RESUMABLE（PR-14.5 降级方案）。
    "PAUSE_REQUESTED": frozenset(
        {"PAUSED", "BUDGET_PAUSED", "RUNNING", "CANCELLED", "FAILED",
         "INTERRUPTED_RESUMABLE"}
    ),
    "PAUSED": frozenset({"RUNNING", "CANCELLED", "FAILED", "INTERRUPTED_RESUMABLE"}),
    "FAILED": frozenset(),
    "EXPIRED": frozenset(),
    "CANCELLED": frozenset(),
    "ACCEPTED": frozenset(),
}

CIRCUIT_FAILURE_THRESHOLD = 3


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class LeaseExpiredError(ValidationError):
    pass


class WorkerManager:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        adapters: dict[str, WorkerAdapter],
        actor_id: str,
        poll_interval_sec: float = 0.05,
        event_recorder=None,
    ) -> None:
        self._conn = conn
        self._adapters = adapters
        self._actor_id = actor_id
        self._poll_interval = poll_interval_sec
        self._event_recorder = event_recorder
        self._scheduler = Scheduler()
        # 十审 W0：活动 run 注册表——cancel_order 凭它找到 adapter+handle
        # 杀进程（此前 handle 只是 run_to_completion 的局部变量，cancel
        # 永远够不到进程）。
        self._runs: dict[str, tuple[WorkerAdapter, RunHandle]] = {}

    # ------------------------------------------------------------------
    # cancel（十审 W0：abort → WorkOrder CANCELLED → adapter cancel →
    # 进程组 kill 闭环）
    # ------------------------------------------------------------------
    async def cancel_order(self, work_order_id: str, *, reason: str = "user_cancel") -> WorkOrderV1:
        """取消未终态的 WorkOrder：adapter.cancel（杀进程树）+ CANCELLED。

        已终态返回当前状态（诚实 no-op）；未知 ID 抛 ValidationError。
        run_to_completion 的驱动循环会发现 CANCELLED 并退出（不改写终态）。
        """
        order = self.order(work_order_id)
        if order is None:
            raise ValidationError(f"unknown work order {work_order_id!r}")
        if order.status in ("ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"):
            return order
        import contextlib

        entry = self._runs.get(work_order_id)
        if entry is not None:
            adapter, handle = entry
            # cancel 尽力而为，状态机照常收尾。
            with contextlib.suppress(Exception):
                await adapter.cancel(handle, reason)
        if self.order(work_order_id).status not in (  # type: ignore[union-attr]
            "ACCEPTED",
            "FAILED",
            "EXPIRED",
            "CANCELLED",
        ):
            # 与驱动循环竞态（如已进 SUBMITTED）：保持诚实当前态。
            with contextlib.suppress(ValidationError):
                self._transition(work_order_id, "CANCELLED", reason)
        current = self.order(work_order_id)
        assert current is not None
        return current

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def hire(
        self,
        order: WorkOrderV1,
        candidates: list[CandidateView],
        *,
        adapter_for: dict[str, str] | None = None,
        attempt_actor: str | None = None,
        attempt_fingerprint: str = "",
    ) -> WorkOrderV1:
        """Schedule + persist + mark OFFERED. Returns the scheduled order."""
        if order.side_effect_policy.idempotency_key:
            dup = self._conn.execute(
                "SELECT work_order_id FROM work_orders WHERE idempotency_key = ?",
                (order.side_effect_policy.idempotency_key,),
            ).fetchone()
            if dup is not None:
                raise ValidationError(f"duplicate idempotency key (already {dup['work_order_id']})")
        view, scored = self._scheduler.select(order, candidates)
        adapter_key = (adapter_for or {}).get(view.card.worker_id, view.card.adapter_type)
        if adapter_key not in self._adapters:
            raise SchedulingError(f"no adapter registered for {adapter_key!r}")
        now = datetime.now(UTC)
        lease = WorkOrderLease(
            lease_id=new_id("lease"),
            issued_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=view.card.health.lease_ttl_sec)).isoformat(),
        )
        scheduled = order.model_copy(
            update={
                "assigned_to": view.card.worker_id,
                "lease": lease,
                "status": "OFFERED",
            }
        )
        self._insert(scheduled, scored)
        # 十四审 PR-14.2：稳定 Job + Attempt 账本——一个用户任务一张卡，
        # retry/resume 只是新 attempt（root = root_work_order_id）。
        self._record_attempt(
            scheduled,
            actor=attempt_actor or str(
                scheduled.inputs.get("_attempt_actor") or "native_agent"
            ),
            fingerprint=attempt_fingerprint,
        )
        self._transition(scheduled.work_order_id, "CLAIMED", "adapter_claimed")
        self._transition(scheduled.work_order_id, "RUNNING", "adapter_started")
        current = self.order(scheduled.work_order_id)
        assert current is not None
        return current

    def _record_attempt(
        self, order: WorkOrderV1, *, actor: str, fingerprint: str
    ) -> None:
        root = order.root_work_order_id or order.work_order_id
        self._conn.execute(
            "INSERT OR IGNORE INTO worker_jobs "
            "(root_job_id, mission_id, user_goal, created_at) VALUES (?, ?, ?, ?)",
            (root, order.mission_id, order.goal, _utcnow()),
        )
        seq = self._conn.execute(
            "SELECT COALESCE(MAX(attempt_seq), 0) + 1 AS s FROM worker_attempts "
            "WHERE root_job_id = ?",
            (root,),
        ).fetchone()["s"]
        self._conn.execute(
            "INSERT INTO worker_attempts (attempt_id, root_job_id, attempt_seq, "
            "actor, failure_fingerprint, state, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)",
            (order.work_order_id, root, seq, actor, fingerprint, _utcnow()),
        )

    def job_view(self, root_job_id: str) -> dict | None:
        """稳定 Job 视图（一张用户任务卡 + 全部 attempts）。"""
        job = self._conn.execute(
            "SELECT * FROM worker_jobs WHERE root_job_id = ?", (root_job_id,)
        ).fetchone()
        if job is None:
            return None
        attempts = self._conn.execute(
            "SELECT * FROM worker_attempts WHERE root_job_id = ? "
            "ORDER BY attempt_seq",
            (root_job_id,),
        ).fetchall()
        return {"job": dict(job), "attempts": [dict(a) for a in attempts]}

    async def run_to_completion(
        self, order: WorkOrderV1, *, timeout_sec: float | None = None
    ) -> tuple[WorkResultV1, VerificationReport]:
        """Drive CLAIMED→RUNNING→SUBMITTED→VERIFYING→ACCEPTED/FAILED.

        十审 W0：
        - handle 注册进 self._runs（cancel_order 由此杀进程树）；
        - 驱动循环发现外部 CANCELLED 即退出（不改写终态、不做验证）；
        - 驱动器自身异常不得让 WorkOrder 永久 RUNNING——标记 FAILED。
        """
        try:
            return await self._run_to_completion_inner(order, timeout_sec=timeout_sec)
        except Exception as exc:  # noqa: BLE001 - 后台驱动永不抛出；失败即数据
            try:
                current = self.order(order.work_order_id)
                if current is not None and current.status not in (
                    "ACCEPTED",
                    "FAILED",
                    "EXPIRED",
                    "CANCELLED",
                ):
                    self._transition(
                        order.work_order_id, "FAILED", f"driver_crash:{type(exc).__name__}"
                    )
            finally:
                self._runs.pop(order.work_order_id, None)
            result = WorkResultV1(
                work_order_id=order.work_order_id,
                worker_id=order.assigned_to or "",
                lease_id=order.lease.lease_id if order.lease else "",
                status="FAILED",
                summary=f"worker driver crashed: {type(exc).__name__}: {exc}",
                warnings=["driver_crash"],
            )
            report = VerificationReport(
                accepted=False,
                verifier_results={"driver_alive": False},
                reasons=(f"driver crashed: {type(exc).__name__}",),
            )
            return result, report

    async def _run_to_completion_inner(
        self, order: WorkOrderV1, *, timeout_sec: float | None = None
    ) -> tuple[WorkResultV1, VerificationReport]:
        card_row = self._conn.execute(
            "SELECT card_json FROM worker_cards WHERE worker_id = ?",
            (order.assigned_to,),
        ).fetchone()
        if card_row is None:
            raise ValidationError(f"unknown worker {order.assigned_to!r}")
        card = WorkerCardV1(**json.loads(card_row["card_json"]))
        adapter = self._adapters[card.adapter_type]
        handle = await adapter.start(order, {})
        self._runs[order.work_order_id] = (adapter, handle)
        try:
            # 十三审 HOTFIX-13.2：无显式硬截止即无 deadline——Worker 有
            # 进度就让它继续做；时间只是观察指标。
            hard_sec = timeout_sec
            if hard_sec is None:
                policy = order.inputs.get("execution_policy") or {}
                if policy.get("hard_deadline_sec") and policy.get(
                    "hard_deadline_source"
                ) in ("user", "benchmark", "admin_policy"):
                    hard_sec = float(policy["hard_deadline_sec"]) + 90
            deadline = (
                datetime.now(UTC) + timedelta(seconds=hard_sec)
                if hard_sec is not None
                else None
            )
            result: WorkResultV1 | None = None
            while deadline is None or datetime.now(UTC) < deadline:
                # 外部 cancel（cancel_order）已翻转状态：退出，不验证不改写。
                current = self.order(order.work_order_id)
                if current is not None and current.status == "CANCELLED":
                    return self._cancelled_result(order), VerificationReport(
                        accepted=False,
                        verifier_results={"cancelled": True},
                        reasons=("cancelled before completion",),
                    )
                try:
                    polled = await adapter.poll(handle)
                except Exception:  # noqa: BLE001 - cancel 后 handle 已移除等
                    current = self.order(order.work_order_id)
                    if current is not None and current.status == "CANCELLED":
                        return self._cancelled_result(order), VerificationReport(
                            accepted=False,
                            verifier_results={"cancelled": True},
                            reasons=("cancelled before completion",),
                        )
                    raise
                if isinstance(polled, WorkResultV1):
                    result = polled
                    break
                self._heartbeat(order.work_order_id, polled.progress_seq)
                await asyncio.sleep(self._poll_interval)
        finally:
            self._runs.pop(order.work_order_id, None)
        if result is None:
            await adapter.cancel(handle, "deadline_exceeded")
            self._transition(order.work_order_id, "FAILED", "deadline_exceeded")
            result = WorkResultV1(
                work_order_id=order.work_order_id,
                worker_id=order.assigned_to or "",
                lease_id=order.lease.lease_id if order.lease else "",
                status="FAILED",
                summary="deadline exceeded before submission",
            )
            # 十一审 PR-A：deadline 终态直接返回——不得继续走
            # SUBMITTED/VERIFYING（FAILED→SUBMITTED 是非法迁移；adapter
            # 层的 wall wrap-up 已先给过 graceful 机会）。
            return result, VerificationReport(
                accepted=False,
                verifier_results={"within_deadline": False},
                reasons=("deadline exceeded before submission",),
            )
        # 十三审 HOTFIX-13.2：INTERRUPTED = 进程死亡但可恢复——不验证、
        # 不 FAILED；保留 checkpoint 供 resume。
        if result.status == "INTERRUPTED":
            current = self.order(order.work_order_id)
            if current is not None and current.status not in (
                "ACCEPTED", "FAILED", "EXPIRED", "CANCELLED",
            ):
                self._transition(
                    order.work_order_id, "INTERRUPTED_RESUMABLE",
                    "worker_interrupted_resumable",
                )
            return result, VerificationReport(
                accepted=False,
                verifier_results={"interrupted": True},
                reasons=("worker interrupted (resumable from checkpoint)",),
            )
        # Stale-lease guard: a result under an old lease is late, not accepted.
        if order.lease and result.lease_id != order.lease.lease_id:
            self._transition(order.work_order_id, "FAILED", "stale_lease_result")
            report = VerificationReport(
                accepted=False,
                verifier_results={"lease_current": False},
                reasons=("result lease_id does not match current lease",),
            )
            return result, report
        # Attribution: the worker's artifact body is durable before verdicts.
        self._conn.execute(
            "INSERT OR REPLACE INTO work_results (work_order_id, lease_id, "
            "result_json, created_at) VALUES (?, ?, ?, ?)",
            (
                result.work_order_id,
                result.lease_id,
                result.model_dump_json(),
                _utcnow(),
            ),
        )
        self._transition(order.work_order_id, "SUBMITTED", "result_submitted")
        self._transition(order.work_order_id, "VERIFYING", "verification_started")
        report = verify_result(order, result)
        if result.status != "COMPLETED":
            report = VerificationReport(
                accepted=False,
                verifier_results={**report.verifier_results, "worker_completed": False},
                reasons=report.reasons + (f"worker status {result.status}",),
            )
        if report.accepted:
            self._transition(order.work_order_id, "ACCEPTED", "verification_passed")
        elif result.status == "BLOCKED":
            # 十六审 A3：Worker 诚实 BLOCKED（缺能力/缺输入）→ 订单
            # BLOCKED，不是 FAILED（语义失败 ≠ 基础设施失败）。
            self._transition(order.work_order_id, "BLOCKED", "worker_blocked")
        else:
            self._transition(order.work_order_id, "FAILED", "verification_failed")
        self._conn.execute(
            "UPDATE work_orders SET verify_report_json = ?, updated_at = ? WHERE work_order_id = ?",
            (
                json.dumps(
                    {
                        "accepted": report.accepted,
                        "checks": report.verifier_results,
                        "reasons": list(report.reasons),
                    },
                    ensure_ascii=False,
                ),
                _utcnow(),
                order.work_order_id,
            ),
        )
        self._update_circuit(order.assigned_to or "", order.capability, report.accepted)
        return result, report

    async def shutdown(self) -> None:
        """服务关闭：取消所有活动 run 的底层进程/任务（十审 W0：不留孤儿）。

        DB 状态不翻转——权威终态由 sweeper/重启对账决定；这里只保证
        子进程树不泄漏。
        """
        import contextlib

        for _wo_id, (adapter, handle) in list(self._runs.items()):
            with contextlib.suppress(Exception):  # 尽力而为
                await adapter.cancel(handle, "service_shutdown")

    def _cancelled_result(self, order: WorkOrderV1) -> WorkResultV1:
        return WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id=order.assigned_to or "",
            lease_id=order.lease.lease_id if order.lease else "",
            status="CANCELLED",
            summary="cancelled before completion",
            warnings=["cancelled"],
        )

    # ------------------------------------------------------------------
    # lease sweeper + reconciliation
    # ------------------------------------------------------------------
    async def sweep_expired(self) -> list[str]:
        """Mark RUNNING orders with expired leases EXPIRED (after reconcile)."""
        now = _utcnow()
        rows = self._conn.execute(
            "SELECT work_order_id, status, lease_expires_at, idempotency_key, "
            "order_json FROM work_orders WHERE status = 'RUNNING' "
            "AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
            (now,),
        ).fetchall()
        expired: list[str] = []
        for row in rows:
            order = WorkOrderV1(**json.loads(row["order_json"]))
            if order.side_effect_policy.class_ not in ("none", "sandbox_process"):
                # Side-effecting: reconcile before declaring lost (总纲 §9.7).
                card_row = self._conn.execute(
                    "SELECT card_json FROM worker_cards WHERE worker_id = ?",
                    (order.assigned_to,),
                ).fetchone()
                if card_row is not None:
                    card = WorkerCardV1(**json.loads(card_row["card_json"]))
                    adapter = self._adapters.get(card.adapter_type)
                    if adapter is not None and row["idempotency_key"]:
                        try:
                            state = await adapter.reconcile(row["idempotency_key"])
                        except Exception:  # noqa: BLE001 - unknown stays safe
                            state = "unknown"
                        if state in ("running", "completed", "unknown"):
                            # Do not mark expired nor re-dispatch: the work may
                            # still land. Operator/workflow decides explicitly.
                            continue
            self._transition(row["work_order_id"], "EXPIRED", "lease_expired")
            expired.append(row["work_order_id"])
        return expired

    # ------------------------------------------------------------------
    def circuit_open(self, worker_id: str, capability: str) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM work_orders WHERE worker_id = ? "
            "AND capability = ? AND status = 'FAILED' AND updated_at > ?",
            (
                worker_id,
                capability,
                (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
            ),
        ).fetchone()
        return int(row["n"]) >= CIRCUIT_FAILURE_THRESHOLD

    def order(self, work_order_id: str) -> WorkOrderV1 | None:
        row = self._conn.execute(
            "SELECT order_json FROM work_orders WHERE work_order_id = ?",
            (work_order_id,),
        ).fetchone()
        return WorkOrderV1(**json.loads(row["order_json"])) if row else None

    def active_orders_for_worker(self, worker_id: str) -> list[WorkOrderV1]:
        """该 worker 名下未终态的 WorkOrder（/worker disable 的 drain 护栏）。"""
        rows = self._conn.execute(
            "SELECT order_json FROM work_orders WHERE worker_id = ? "
            "AND status NOT IN ('ACCEPTED', 'REJECTED', 'EXPIRED', 'CANCELLED', 'FAILED')",
            (worker_id,),
        ).fetchall()
        return [WorkOrderV1(**json.loads(r["order_json"])) for r in rows]

    def order_times(self, work_order_id: str) -> dict[str, Any]:
        """十三审 HOTFIX-13.1：权威时间——created/started/finished 全部
        来自转移日志（worker_events 幂等键 wo:STATUS），TUI 不得再
        用本地第一次看到的时间计时。finished_at 一经写入不可变。"""
        row = self._conn.execute(
            "SELECT created_at FROM work_orders WHERE work_order_id = ?",
            (work_order_id,),
        ).fetchone()
        created_at = row["created_at"] if row else None
        rows = self._conn.execute(
            "SELECT idempotency_key, occurred_at FROM worker_events "
            "WHERE idempotency_key LIKE ?",
            (f"{work_order_id}:%",),
        ).fetchall()
        by_status = {}
        for r in rows:
            status = r["idempotency_key"].rsplit(":", 1)[-1]
            by_status[status] = r["occurred_at"]
        terminal = next(
            (s for s in ("ACCEPTED", "FAILED", "EXPIRED", "CANCELLED") if s in by_status),
            None,
        )
        started_at = by_status.get("RUNNING")
        finished_at = by_status.get(terminal) if terminal else None
        # 十三审：中断/暂停也要冻结计时（不是终态但 Worker 不在工作）。
        paused_at = next(
            (by_status[s] for s in ("INTERRUPTED_RESUMABLE", "BUDGET_PAUSED")
             if s in by_status),
            None,
        )
        duration_ms = None
        if started_at and finished_at:
            from datetime import datetime as _dt

            try:
                duration_ms = int(
                    (
                        _dt.fromisoformat(finished_at) - _dt.fromisoformat(started_at)
                    ).total_seconds()
                    * 1000
                )
            except ValueError:
                duration_ms = None
        return {
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "paused_at": paused_at,
            "duration_ms": duration_ms,
            "settled": terminal is not None,
        }

    def orders_for_mission(self, mission_id: str) -> list[WorkOrderV1]:
        rows = self._conn.execute(
            "SELECT order_json FROM work_orders WHERE mission_id = ? ORDER BY created_at",
            (mission_id,),
        ).fetchall()
        return [WorkOrderV1(**json.loads(r["order_json"])) for r in rows]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _insert(self, order: WorkOrderV1, scored) -> None:
        now = _utcnow()
        side = order.side_effect_policy
        self._conn.execute(
            "INSERT INTO work_orders (work_order_id, mission_id, task_id, worker_id, "
            "capability, status, order_json, lease_id, lease_expires_at, "
            "idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order.work_order_id,
                order.mission_id,
                order.task_id,
                order.assigned_to,
                order.capability,
                order.status,
                order.model_dump_json(),
                order.lease.lease_id if order.lease else None,
                order.lease.expires_at if order.lease else None,
                side.idempotency_key,
                now,
                now,
            ),
        )
        self._event(
            order.assigned_to or "",
            "rosclaw.worker.work_order.offered.v1",
            {
                "work_order_id": order.work_order_id,
                "score": scored.score,
                "features": scored.features,
                "policy": self._scheduler.policy_version,
                "reasons": list(scored.reasons),
            },
            order.work_order_id,
        )

    def _transition(self, work_order_id: str, to_status: str, reason: str) -> None:
        row = self._conn.execute(
            "SELECT status, order_json FROM work_orders WHERE work_order_id = ?",
            (work_order_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(f"unknown work order {work_order_id!r}")
        from_status = row["status"]
        if to_status == from_status:
            return
        if to_status not in _RUN_TRANSITIONS[from_status]:
            raise ValidationError(f"illegal work order transition {from_status} -> {to_status}")
        order = WorkOrderV1(**json.loads(row["order_json"]))
        order = order.model_copy(update={"status": to_status})
        self._conn.execute(
            "UPDATE work_orders SET status = ?, order_json = ?, updated_at = ? "
            "WHERE work_order_id = ?",
            (to_status, order.model_dump_json(), _utcnow(), work_order_id),
        )
        # 十四审 PR-14.2：终态/中断即结算 attempt（ACTIVE→SETTLED——
        # 活跃唯一约束随之释放，resume/retry 才能开新 attempt）。
        if to_status in (
            "ACCEPTED", "FAILED", "EXPIRED", "CANCELLED", "INTERRUPTED_RESUMABLE",
            # 十六审 P0-B：BLOCKED 也是结算点——不结算会让 ACTIVE 唯一
            # 索引挡住同 root 的能力升级 attempt（escalation 无法启动）。
            "BLOCKED",
        ):
            # PR-14.5：legacy 单（14.2 前创建，无 attempts 行）先补账再
            # 结算——重启对账的 INTERRUPTED_RESUMABLE 也有完整 Job 视图。
            root = order.root_work_order_id or order.work_order_id
            self._conn.execute(
                "INSERT OR IGNORE INTO worker_jobs "
                "(root_job_id, mission_id, user_goal, created_at) "
                "VALUES (?, ?, ?, ?)",
                (root, order.mission_id, order.goal, _utcnow()),
            )
            backfill_seq = self._conn.execute(
                "SELECT COALESCE(MAX(attempt_seq), 0) + 1 AS s FROM worker_attempts "
                "WHERE root_job_id = ?",
                (root,),
            ).fetchone()["s"]
            self._conn.execute(
                "INSERT OR IGNORE INTO worker_attempts (attempt_id, root_job_id, "
                "attempt_seq, actor, state, created_at) "
                "VALUES (?, ?, ?, 'native_agent', 'ACTIVE', ?)",
                (work_order_id, root, backfill_seq, _utcnow()),
            )
            self._conn.execute(
                "UPDATE worker_attempts SET state = 'SETTLED', "
                "termination_cause = ?, settled_at = ? "
                "WHERE attempt_id = ? AND state = 'ACTIVE'",
                (reason, _utcnow(), work_order_id),
            )
        verb = {
            "CLAIMED": "claimed",
            "RUNNING": "started",
            "SUBMITTED": "submitted",
            "VERIFYING": "verifying",
            "ACCEPTED": "accepted",
            "FAILED": "failed",
            "EXPIRED": "expired",
            "CANCELLED": "cancelled",
        }.get(to_status, "updated")
        entity = "work_result" if to_status == "SUBMITTED" else "work_order"
        self._event(
            order.assigned_to or "",
            f"rosclaw.worker.{entity}.{verb}.v1",
            {
                "work_order_id": work_order_id,
                "from": from_status,
                "to": to_status,
                "reason": reason,
            },
            f"{work_order_id}:{to_status}",
        )
        # 批次 B：同步桥到 AgentEventV2（service 提供的 recorder 负责调度到
        # 事件循环；manager 本身保持同步）。
        if self._event_recorder is not None:
            self._event_recorder(
                order.mission_id,
                to_status,
                {"work_order_id": work_order_id, "worker_id": order.assigned_to or ""},
            )

    def _heartbeat(self, work_order_id: str, progress_seq: int) -> None:
        self._conn.execute(
            "UPDATE work_orders SET heartbeat_seq = ?, last_heartbeat_at = ? "
            "WHERE work_order_id = ?",
            (progress_seq, _utcnow(), work_order_id),
        )

    def _update_circuit(self, worker_id: str, capability: str, success: bool) -> None:
        # Circuit state derives from work_orders history (see circuit_open);
        # nothing extra to persist in P0 — history is the record.
        return None

    def _event(self, worker_id: str, event_type: str, payload: dict, idem: str | None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO worker_events (event_id, worker_id, event_type, "
            "actor_id, payload_json, idempotency_key, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("wevt"),
                worker_id,
                event_type,
                self._actor_id,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                idem,
                _utcnow(),
            ),
        )
