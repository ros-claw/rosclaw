"""RetryCoordinator——唯一重试决策者（十四审 PR-14.2，总纲 §3.5）。

规则：
- agentd 是唯一重试决策者；Native Agent 只能"提出"retry/resume，协调器
  做幂等仲裁（自动+手动并发也只有一个 attempt）。
- 同一 root job 同时最多一个 ACTIVE attempt（应用层 CAS + DB 部分唯一
  索引兜底）；已有 ACTIVE attempt 时任何 retry 请求返回现有 attempt。
- auto retry 只接受结构化可重试 cause（PROVIDER_TRANSIENT/WORKER_CRASH/
  EVENT_PIPE_BROKEN）；"worker exited" 之类的进程表象永不是依据。
- USER_CANCELLED/USER_PAUSED/DELIVERABLE_REJECTED/TOOL_FAILED/
  PROVIDER_FATAL 不自动重试。
- 同一 (root, cause) 最多自动重试一次；手动 retry 由用户权威驱动，
  但 ACTIVE 去重同样生效。
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

    from rosclaw.contracts.worker.order import WorkOrderV1

#: 可自动重试的结构化 cause（总纲 §3.5 白名单）。
AUTO_RETRYABLE_CAUSES = frozenset({
    "PROVIDER_TRANSIENT",
    "WORKER_CRASH",
    "EVENT_PIPE_BROKEN",
})

#: 永不自动重试（用户动作/语义失败/验证拒绝）。
NEVER_AUTO_CAUSES = frozenset({
    "COMPLETED",
    "USER_CANCELLED",
    "USER_PAUSED",
    "BUDGET_HARD_PAUSED",
    "DELIVERABLE_REJECTED",
    "TOOL_FAILED",
    "PROVIDER_FATAL",
})

#: 旧摘要文本 → 结构化 cause（AdapterError 期的遗留表述，映射到
#: WORKER_CRASH/PROVIDER_TRANSIENT——绝不包含 "worker exited"）。
_LEGACY_CAUSE_HINTS = {
    "liveness lost": "WORKER_CRASH",
    "startup timeout": "WORKER_CRASH",
    "driver crashed": "WORKER_CRASH",
    "driver_crash": "WORKER_CRASH",
    "not found at start time": "WORKER_CRASH",
    "PROVIDER_TIMEOUT": "PROVIDER_TRANSIENT",
    "provider_timeout": "PROVIDER_TRANSIENT",
}

_CAUSE_RE = re.compile(r"\[([A-Z_]{3,})\]")


def parse_cause(summary: str) -> str | None:
    """从结果摘要解析结构化 termination cause（[CAUSE] 形式——
    十四审 PR-14.1 起 FAILED 摘要必带）；旧文本走遗留映射。"""
    match = _CAUSE_RE.search(summary or "")
    if match:
        return match.group(1)
    for hint, cause in _LEGACY_CAUSE_HINTS.items():
        if hint in (summary or ""):
            return cause
    return None


def should_auto_retry(cause: str | None) -> bool:
    return cause in AUTO_RETRYABLE_CAUSES


class RetryCoordinator:
    """(root_job_id 粒度 asyncio 锁 + worker_attempts 表 CAS）。

    spawn/candidates 以可调用注入（避免 service/dispatcher 循环依赖）。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        manager,
        candidates_fn,
        spawn_fn,
    ) -> None:
        self._conn = conn
        self._manager = manager
        self._candidates_fn = candidates_fn
        self._spawn_fn = spawn_fn
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, root_job_id: str) -> asyncio.Lock:
        lock = self._locks.get(root_job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[root_job_id] = lock
        return lock

    def active_attempt(self, root_job_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT attempt_id FROM worker_attempts "
            "WHERE root_job_id = ? AND state = 'ACTIVE'",
            (root_job_id,),
        ).fetchone()
        return str(row["attempt_id"]) if row else None

    def _fingerprint_seen(self, root_job_id: str, fingerprint: str) -> str | None:
        if not fingerprint:
            return None
        row = self._conn.execute(
            "SELECT attempt_id FROM worker_attempts "
            "WHERE root_job_id = ? AND failure_fingerprint = ?",
            (root_job_id, fingerprint),
        ).fetchone()
        return str(row["attempt_id"]) if row else None

    async def request_retry(
        self,
        order: WorkOrderV1,
        *,
        cause: str | None,
        actor: str,
        note: str = "",
        resume_session: str = "",
    ) -> tuple[WorkOrderV1 | None, bool, str]:
        """返回 (attempt_order, created, reason)。

        reason: created | active_exists | already_retried | not_retryable |
        worker_unavailable | scheduling_failed
        """
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import WorkOrderV1 as _WorkOrderV1

        root = order.root_work_order_id or order.work_order_id
        async with self._lock_for(root):
            active = self.active_attempt(root)
            if active is not None:
                existing = self._manager.order(active)
                return existing, False, "active_exists"
            # 同 root 已有**其他** attempt 成功（如 auto retry 已完成）——
            # 任务已达成，retry 返回该 attempt 而不是再烧一次。retry
            # 成功单本身（用户要重跑）是合法新 attempt，不在此列。
            succeeded = self._conn.execute(
                "SELECT a.attempt_id FROM worker_attempts a "
                "JOIN work_orders w ON w.work_order_id = a.attempt_id "
                "WHERE a.root_job_id = ? AND w.status = 'ACCEPTED' "
                "AND a.attempt_id <> ?",
                (root, order.work_order_id),
            ).fetchone()
            if succeeded is not None:
                return (
                    self._manager.order(str(succeeded["attempt_id"])),
                    False,
                    "already_succeeded",
                )
            cause = cause or "SIGNAL_UNKNOWN"
            if actor == "auto":
                if not should_auto_retry(cause):
                    return None, False, "not_retryable"
                fingerprint = f"auto:{cause}"
                seen = self._fingerprint_seen(root, fingerprint)
                if seen is not None:
                    return self._manager.order(seen), False, "already_retried"
            else:
                # 手动/用户权威：fingerprint 带 seq——允许同因再试，
                # ACTIVE 去重已由上面的检查保证。
                seq_count = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM worker_attempts WHERE root_job_id = ?",
                    (root,),
                ).fetchone()["c"]
                fingerprint = f"{actor}:{cause}#{seq_count + 1}"
            instructions = str(order.inputs.get("instructions") or order.goal)
            notes = order.inputs.get("steer_notes") or []
            if notes:
                instructions += "\n\n追加约束（来自 retry 前的 steer 备注）：" + "；".join(
                    str(n.get("note", "")) for n in notes
                )
            if note:
                instructions += f"\n\n（上一 attempt 终止原因[{cause}]——{note}）"
            inputs = {
                **dict(order.inputs),
                "instructions": instructions,
                "_attempt_actor": actor,
            }
            if actor == "auto":
                inputs["_auto_retried"] = True
                reuse = str(order.inputs.get("workspace") or "")
                if reuse:
                    inputs["_reuse_workspace"] = reuse
                inputs["instructions"] += (
                    "\n\n（上一次 attempt 因基础设施错误中断——workspace 里可能"
                    "已有部分成果：先检查现状，再继续，不要从零开始。）"
                )
            if resume_session:
                inputs["_resume_session"] = resume_session
            new_order = _WorkOrderV1(
                work_order_id=new_id("wo"),
                mission_id=order.mission_id,
                issued_by=order.issued_by,
                capability=order.capability,
                goal=order.goal,
                inputs=inputs,
                budgets=order.budgets,
                expected_output=order.expected_output,
                side_effect_policy=order.side_effect_policy,
                delegation_depth=0,
                max_delegation_depth=1,
                parent_work_order_id=order.work_order_id,
                root_work_order_id=root,
            )
            candidates = self._candidates_fn(order.assigned_to or "auto", order.capability)
            if not candidates:
                return None, False, "worker_unavailable"
            try:
                scheduled = self._manager.hire(
                    new_order, candidates,
                    attempt_actor=actor,
                    attempt_fingerprint=fingerprint,
                )
            except Exception:
                return None, False, "scheduling_failed"
            self._spawn_fn(scheduled)
            return scheduled, True, "created"
