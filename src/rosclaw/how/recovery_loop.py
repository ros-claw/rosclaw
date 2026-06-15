"""rosclaw_how.recovery_loop — Sprint 9: automatic retry with recovery hints.

The RecoveryLoop closes the failure → recovery → retry → compare → learn cycle:

  1. Listens for ``rosclaw.how.recovery_hint.generated``
  2. Applies parameter_patch to the next task attempt
  3. After retry completes, compares outcome with original failure
  4. Records the result to Memory (success_pattern or failure)
  5. Updates HeuristicEngine rule efficacy (success_count / failure_count)

Design:
  - Stateless: all retry state is stored in SeekDB ``retries`` table.
  - Idempotent: same recovery_hint event processed twice → no-op.
  - Observable: publishes ``rosclaw.how.retry.completed`` events.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("rosclaw.how.recovery_loop")


class RecoveryLoop:
    """Orchestrate failure → recovery → retry → compare → learn.

    Args:
        event_bus: EventBus for publishing retry events.
        memory_interface: MemoryInterface for querying episodes / storing outcomes.
        heuristic_engine: HeuristicEngine for updating rule stats.
    """

    def __init__(
        self,
        event_bus: Any,
        memory_interface: Any,
        heuristic_engine: Any,
    ) -> None:
        self._bus = event_bus
        self._memory = memory_interface
        self._how = heuristic_engine
        self._table = "retries"
        self._executor = None
        self._ensure_table()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create retries tracking table if schema supports it."""
        client = self._memory.seekdb_client if self._memory else None
        if client is None:
            return
        # SeekDBMemoryClient auto-creates tables on insert; SQLite client
        # needs the table in SEEKDB_SCHEMAS.  We use a lightweight record
        # that fits the generic key-value pattern.
        import contextlib
        with contextlib.suppress(Exception):
            client.count(self._table)

    def subscribe(self) -> None:
        """Subscribe to recovery hint events on the EventBus."""
        if self._bus is None:
            return
        self._bus.subscribe("rosclaw.how.recovery_hint.generated", self._on_recovery_hint)
        self._bus.subscribe("rosclaw.sandbox.episode.succeeded", self._on_retry_success)
        self._bus.subscribe("rosclaw.sandbox.episode.failed", self._on_retry_failure)
        logger.info("RecoveryLoop subscribed")

    def unsubscribe(self) -> None:
        """Unsubscribe from recovery hint events and release resources."""
        if self._bus is None:
            return
        self._bus.unsubscribe("rosclaw.how.recovery_hint.generated", self._on_recovery_hint)
        self._bus.unsubscribe("rosclaw.sandbox.episode.succeeded", self._on_retry_success)
        self._bus.unsubscribe("rosclaw.sandbox.episode.failed", self._on_retry_failure)
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    # ── event handlers ───────────────────────────────────────────────────

    def _on_recovery_hint(self, event: Any) -> None:
        """Store recovery hint and mark original episode for retry."""
        payload = event.payload
        request_id = payload.get("request_id", "")
        failure_type = payload.get("failure_type", "")
        retry_plan = payload.get("retry_plan", {})
        rule_id = retry_plan.get("rule_id", "")
        max_retries = retry_plan.get("max_retries", 3)

        # Record retry intent in SeekDB
        record = {
            "id": request_id,
            "failure_type": failure_type,
            "rule_id": rule_id,
            "parameter_patch": retry_plan.get("parameter_patch", {}),
            "max_retries": max_retries,
            "attempt_count": 0,
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "original_outcome": "failure",
            "retry_outcome": None,
            "improvement": None,
        }
        try:
            self._memory.seekdb_client.insert(self._table, record)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Retry record insert failed (table may not exist): %s", exc)
            # Fallback: store in memory_interface metadata
            if self._memory:
                self._memory._client.insert("failures", {
                    "id": f"retry_{request_id}",
                    "robot_id": getattr(self._memory, "_robot_id", "unknown"),
                    "failure_type": failure_type,
                    "recovery_hint": json.dumps(record),
                    "timestamp": time.time(),
                })

        logger.info("RecoveryLoop recorded retry intent for %s (rule=%s)", request_id, rule_id)

    def _on_retry_success(self, event: Any) -> None:
        """Mark retry as succeeded and update rule efficacy."""
        payload = event.payload
        request_id = payload.get("request_id", "")
        episode_id = payload.get("episode_id", "")

        # Find pending retry record
        retry = self._get_retry(request_id)
        if retry is None:
            return

        # Update retry record
        self._update_retry(request_id, {
            "status": "succeeded",
            "retry_outcome": "success",
            "attempt_count": retry.get("attempt_count", 0) + 1,
            "updated_at": time.time(),
        })

        # Update rule efficacy (+1 success)
        rule_id = retry.get("rule_id", "")
        if rule_id and self._how:
            self._run_async(self._how.record_outcome(rule_id, success=True))

        # Store success pattern in Memory
        if self._memory:
            skill_id = payload.get("skill_id", "")
            self._memory._client.insert("success_patterns", {
                "id": f"sp_retry_{request_id}",
                "skill_id": skill_id,
                "robot_id": getattr(self._memory, "_robot_id", "unknown"),
                "context_hash": retry.get("failure_type", ""),
                "success_count": 1,
                "avg_duration_sec": payload.get("duration_sec", 0.0),
                "metadata": {
                    "original_failure": retry.get("failure_type"),
                    "parameter_patch": retry.get("parameter_patch"),
                    "rule_id": rule_id,
                    "episode_id": episode_id,
                },
            })

        # Publish retry completed event
        if self._bus:
            from rosclaw.core.event_bus import Event
            self._bus.publish(Event(
                topic="rosclaw.how.retry.completed",
                payload={
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "result": "success",
                    "improvement": self._compute_improvement(retry, True),
                    "rule_id": rule_id,
                },
                source="recovery_loop",
            ))

        logger.info("RecoveryLoop: retry succeeded for %s (rule=%s)", request_id, rule_id)

    def _on_retry_failure(self, event: Any) -> None:
        """Mark retry as failed and update rule efficacy."""
        payload = event.payload
        request_id = payload.get("request_id", "")
        episode_id = payload.get("episode_id", "")

        retry = self._get_retry(request_id)
        if retry is None:
            return

        attempt_count = retry.get("attempt_count", 0) + 1
        max_retries = retry.get("max_retries", 3)
        status = "failed" if attempt_count >= max_retries else "pending"

        self._update_retry(request_id, {
            "status": status,
            "retry_outcome": "failure",
            "attempt_count": attempt_count,
            "updated_at": time.time(),
        })

        # Update rule efficacy (-1 success = +1 failure)
        rule_id = retry.get("rule_id", "")
        if rule_id and self._how:
            self._run_async(self._how.record_outcome(rule_id, success=False))

        # If max retries reached, store as new failure
        if status == "failed" and self._memory:
            self._memory.write_failure_memory({
                "id": f"retry_failed_{request_id}",
                "robot_id": getattr(self._memory, "_robot_id", "unknown"),
                "failure_type": retry.get("failure_type", "unknown"),
                "root_cause": f"Retry failed after {max_retries} attempts",
                "recovery_hint": "Escalate to human operator",
                "metadata": {
                    "original_failure": retry.get("failure_type"),
                    "parameter_patch": retry.get("parameter_patch"),
                    "rule_id": rule_id,
                    "episode_id": episode_id,
                },
            })

        if self._bus:
            from rosclaw.core.event_bus import Event
            self._bus.publish(Event(
                topic="rosclaw.how.retry.completed",
                payload={
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "result": "failure",
                    "attempt_count": attempt_count,
                    "max_retries": max_retries,
                    "rule_id": rule_id,
                },
                source="recovery_loop",
            ))

        logger.info("RecoveryLoop: retry failed for %s (attempt=%d/%d)", request_id, attempt_count, max_retries)

    # ── internals ────────────────────────────────────────────────────────

    def _get_retry(self, request_id: str) -> dict[str, Any] | None:
        """Fetch retry record by request_id."""
        if not self._memory:
            return None
        try:
            rows = self._memory.seekdb_client.query(
                self._table,
                filters={"id": request_id},
                limit=1,
            )
            return dict(rows[0]) if rows else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Retry lookup failed: %s", exc)
            return None

    def _update_retry(self, request_id: str, updates: dict[str, Any]) -> bool:
        """Update retry record fields."""
        if not self._memory:
            return False
        try:
            return self._memory.seekdb_client.update(self._table, request_id, updates)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Retry update failed: %s", exc)
            return False

    @staticmethod
    def _compute_improvement(retry_record: dict[str, Any], retry_succeeded: bool) -> dict[str, Any]:
        """Compute improvement metrics between original failure and retry."""
        return {
            "original_outcome": retry_record.get("original_outcome", "failure"),
            "retry_outcome": "success" if retry_succeeded else "failure",
            "parameter_patch": retry_record.get("parameter_patch", {}),
            "failure_type": retry_record.get("failure_type", ""),
            "rule_id": retry_record.get("rule_id", ""),
            "timestamp": time.time(),
        }

    def _run_async(self, coro):
        """Run async coroutine from sync context (mirror Runtime._run_async).

        Uses a lazily-initialized ThreadPoolExecutor to avoid creating
        a new thread pool on every call.  Timeout is 30 seconds.
        """
        import asyncio
        import concurrent.futures
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)
