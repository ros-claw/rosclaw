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
from collections.abc import Callable
from typing import Any

from rosclaw.sandbox.evidence import (
    SimulationEvidenceVerification,
    verify_simulation_receipt,
)

logger = logging.getLogger("rosclaw.how.recovery_loop")
MAX_RETRIES = 10
MAX_RETRY_HINT_BYTES = 16 * 1024


def _valid_text(value: Any, *, maximum: int, allow_empty: bool = True) -> bool:
    return isinstance(value, str) and (allow_empty or bool(value)) and len(value) <= maximum


def _validated_retry_hint(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    request_id = payload.get("request_id")
    failure_type = payload.get("failure_type", "")
    retry_plan = payload.get("retry_plan", {})
    if (
        not _valid_text(request_id, maximum=256, allow_empty=False)
        or not _valid_text(failure_type, maximum=256)
        or not isinstance(retry_plan, dict)
    ):
        return None

    rule_id = retry_plan.get("rule_id", "")
    max_retries = retry_plan.get("max_retries", 3)
    parameter_patch = retry_plan.get("parameter_patch", {})
    if (
        not _valid_text(rule_id, maximum=256)
        or isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 1 <= max_retries <= MAX_RETRIES
        or not isinstance(parameter_patch, dict)
        or len(parameter_patch) > 64
    ):
        return None
    try:
        encoded_patch = json.dumps(
            parameter_patch,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    if len(encoded_patch) > MAX_RETRY_HINT_BYTES:
        return None
    return {
        "request_id": request_id,
        "failure_type": failure_type,
        "rule_id": rule_id,
        "max_retries": max_retries,
        "parameter_patch": parameter_patch,
    }


def _trusted_physics_outcome(
    payload: dict[str, Any],
    verifier: Callable[[dict[str, Any]], SimulationEvidenceVerification],
) -> bool:
    """Only verified simulation evidence may change rule efficacy."""
    receipt = payload.get("simulation_receipt")
    attached_evidence = bool(
        payload.get("physics_executed") is True
        and payload.get("receipt_verified") is True
        and payload.get("data_quality_pass") is True
        and payload.get("evidence_domain") == "SIMULATION"
        and isinstance(receipt, dict)
    )
    if not attached_evidence:
        return False
    try:
        return verifier(receipt).verified
    except Exception:  # noqa: BLE001 - learning must fail closed
        return False


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
        *,
        receipt_verifier: Callable[[dict[str, Any]], SimulationEvidenceVerification] | None = None,
    ) -> None:
        self._bus = event_bus
        self._memory = memory_interface
        self._how = heuristic_engine
        self._receipt_verifier = receipt_verifier or verify_simulation_receipt
        self._table = "retries"
        self._ensure_table()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create retries tracking table if schema supports it."""
        client = self._memory.seekdb_client if self._memory else None
        if client is None:
            return
        # InMemoryKnowledgeStore auto-creates tables on insert; SQLite client
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

    # ── event handlers ───────────────────────────────────────────────────

    def _on_recovery_hint(self, event: Any) -> None:
        """Store recovery hint and mark original episode for retry."""
        hint = _validated_retry_hint(getattr(event, "payload", None))
        if hint is None:
            logger.warning("Ignoring malformed recovery hint")
            return
        request_id = hint["request_id"]
        failure_type = hint["failure_type"]
        rule_id = hint["rule_id"]
        max_retries = hint["max_retries"]

        # Record retry intent in SeekDB
        record = {
            "id": request_id,
            "failure_type": failure_type,
            "rule_id": rule_id,
            "parameter_patch": hint["parameter_patch"],
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
                self._memory._client.insert(
                    "failures",
                    {
                        "id": f"retry_{request_id}",
                        "robot_id": getattr(self._memory, "_robot_id", "unknown"),
                        "failure_type": failure_type,
                        "recovery_hint": json.dumps(record),
                        "timestamp": time.time(),
                    },
                )

        logger.info("RecoveryLoop recorded retry intent for %s (rule=%s)", request_id, rule_id)

    def _on_retry_success(self, event: Any) -> None:
        """Mark retry as succeeded and update rule efficacy."""
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        request_id = payload.get("request_id", "")
        episode_id = payload.get("episode_id", "")
        if not _trusted_physics_outcome(payload, self._receipt_verifier):
            logger.warning("Ignoring unverified retry-success evidence for %s", request_id)
            return

        # Find pending retry record
        retry = self._get_retry(request_id)
        if retry is None:
            return

        # Update retry record
        attempt_count = retry.get("attempt_count", 0)
        if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
            logger.warning("Ignoring malformed retry record for %s", request_id)
            return
        self._update_retry(
            request_id,
            {
                "status": "succeeded",
                "retry_outcome": "success",
                "attempt_count": attempt_count + 1,
                "updated_at": time.time(),
            },
        )

        # Update rule efficacy (+1 success)
        rule_id = retry.get("rule_id", "")
        if rule_id and self._how:
            self._run_async(self._how.record_outcome(rule_id, success=True))

        # Store success pattern in Memory
        if self._memory:
            skill_id = payload.get("skill_id", "")
            self._memory._client.insert(
                "success_patterns",
                {
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
                },
            )

        # Publish retry completed event
        if self._bus:
            from rosclaw.core.event_bus import Event

            self._bus.publish(
                Event(
                    topic="rosclaw.how.retry.completed",
                    payload={
                        "request_id": request_id,
                        "episode_id": episode_id,
                        "result": "success",
                        "improvement": self._compute_improvement(retry, True),
                        "rule_id": rule_id,
                    },
                    source="recovery_loop",
                )
            )

        logger.info("RecoveryLoop: retry succeeded for %s (rule=%s)", request_id, rule_id)

    def _on_retry_failure(self, event: Any) -> None:
        """Mark retry as failed and update rule efficacy."""
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        request_id = payload.get("request_id", "")
        episode_id = payload.get("episode_id", "")
        if not _trusted_physics_outcome(payload, self._receipt_verifier):
            logger.warning("Ignoring unverified retry-failure evidence for %s", request_id)
            return

        retry = self._get_retry(request_id)
        if retry is None:
            return

        previous_attempts = retry.get("attempt_count", 0)
        max_retries = retry.get("max_retries", 3)
        if (
            isinstance(previous_attempts, bool)
            or not isinstance(previous_attempts, int)
            or previous_attempts < 0
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 1 <= max_retries <= MAX_RETRIES
        ):
            logger.warning("Ignoring malformed retry record for %s", request_id)
            return
        attempt_count = previous_attempts + 1
        status = "failed" if attempt_count >= max_retries else "pending"

        self._update_retry(
            request_id,
            {
                "status": status,
                "retry_outcome": "failure",
                "attempt_count": attempt_count,
                "updated_at": time.time(),
            },
        )

        # Update rule efficacy (-1 success = +1 failure)
        rule_id = retry.get("rule_id", "")
        if rule_id and self._how:
            self._run_async(self._how.record_outcome(rule_id, success=False))

        # If max retries reached, store as new failure
        if status == "failed" and self._memory:
            self._memory.write_failure_memory(
                {
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
                }
            )

        if self._bus:
            from rosclaw.core.event_bus import Event

            self._bus.publish(
                Event(
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
                )
            )

        logger.info(
            "RecoveryLoop: retry failed for %s (attempt=%d/%d)",
            request_id,
            attempt_count,
            max_retries,
        )

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
        """Run async coroutine from sync context using the shared helper.

        Delegates to ``rosclaw.core.async_utils.run_sync`` so we never call
        ``asyncio.run`` from inside an already-running event loop.
        """
        from rosclaw.core.async_utils import run_sync

        return run_sync(coro)
