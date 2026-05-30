"""rosclaw_how.engine — HeuristicEngine: fast rule-based recovery.

The engine provides:
  * suggest_recovery()  — lookup heuristic_rules table (<10ms)
  * record_outcome()    — update rule efficacy counters
  * seed_defaults()     — populate initial safety rules

Design:
  - Pure rule-based, zero LLM calls in the hot path.
  - Condition matching: exact -> substring -> keyword overlap.
  - Outcome tracking: success_count / failure_count / priority.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("rosclaw.how.engine")


class HeuristicEngine:
    """Fast heuristic rule engine backed by SeekDB.

    Args:
        seekdb_client: Any SeekDBClient implementation (memory / SQLite).
        knowledge_interface: Optional knowledge interface for analogy fallback.
    """

    def __init__(
        self,
        seekdb_client: Any,
        knowledge_interface: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._seekdb = seekdb_client
        self._knowledge = knowledge_interface
        self._event_bus = event_bus
        self._table = "heuristic_rules"
        self._rule_cache: dict[str, dict[str, Any]] = {}
        self._cache_valid = False
        self._subscriptions: list[Any] = []

    # ── lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Warm the rule cache from SeekDB and subscribe to failure events."""
        try:
            rows = self._seekdb.query(self._table, limit=1_000)
            self._rule_cache = {str(r.get("id", "")): dict(r) for r in rows if r.get("id")}
            self._cache_valid = True
            logger.info("HeuristicEngine warmed %d rules", len(self._rule_cache))
        except Exception as exc:  # noqa: BLE001
            logger.warning("HeuristicEngine warm failed: %s", exc)
            self._cache_valid = False

        # CRITICAL FIX: subscribe to failure events on EventBus for active recovery
        if self._event_bus is not None:
            try:
                self._subscriptions.append(
                    self._event_bus.subscribe("praxis.failed", self._on_failure_event)
                )
                self._subscriptions.append(
                    self._event_bus.subscribe("firewall.action_blocked", self._on_failure_event)
                )
                self._subscriptions.append(
                    self._event_bus.subscribe("safety.violation", self._on_failure_event)
                )
                logger.info("HeuristicEngine subscribed to failure events")
            except Exception as exc:  # noqa: BLE001
                logger.warning("HeuristicEngine EventBus subscribe failed: %s", exc)

    async def shutdown(self) -> None:
        """Clear cache and unsubscribe from EventBus."""
        self._rule_cache.clear()
        self._cache_valid = False
        # CRITICAL FIX: unsubscribe from EventBus to prevent leaks
        for sub in self._subscriptions:
            try:
                if hasattr(sub, 'unsubscribe'):
                    sub.unsubscribe()
            except Exception:  # noqa: BLE001
                pass
        self._subscriptions.clear()

    # ── public API ───────────────────────────────────────────────────────

    async def suggest_recovery(
        self,
        error_log: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Return the best matching recovery strategy (<10ms when cached).

        Matching priority:
          1. Exact condition match on error_log (fastest)
          2. Substring match: condition text appears inside error_log
          3. Keyword overlap between error_log and condition

        Returns:
            {"rule_id": str, "condition": str, "action": str,
             "priority": int, "source": "heuristic"} or None.
        """
        if not error_log:
            return None

        # 1. Exact match
        result = self._query_exact(error_log)
        if result:
            return self._format(result)

        # 2. Substring match
        result = self._query_substring(error_log)
        if result:
            return self._format(result)

        # 3. Knowledge fallback (optional)
        if self._knowledge:
            return await self._knowledge_fallback(error_log, context)

        return None

    async def record_outcome(self, rule_id: str, success: bool) -> bool:
        """Increment success_count or failure_count for a rule.

        Also updates last_triggered timestamp.
        """
        if not rule_id:
            return False

        rule = self._rule_cache.get(rule_id)
        if rule is None:
            # Try to fetch from DB
            rows = self._seekdb.query(
                self._table, filters={"id": rule_id}, limit=1
            )
            if not rows:
                return False
            rule = dict(rows[0])

        col = "success_count" if success else "failure_count"
        new_val = int(rule.get(col, 0)) + 1
        try:
            self._seekdb.update(
                self._table,
                rule_id,
                {col: new_val, "last_triggered": time.time()},
            )
            # Update local cache
            rule[col] = new_val
            rule["last_triggered"] = time.time()
            self._rule_cache[rule_id] = rule
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_outcome failed for %s: %s", rule_id, exc)
            return False

    async def seed_defaults(self) -> int:
        """Populate heuristic_rules with v1.0 default safety rules.

        Returns number of rules inserted.
        """
        defaults = [
            ("joint limit exceeded", "Reduce Kp gain by 30% and re-validate", 1),
            ("collision detected", "Replan trajectory with larger clearance", 1),
            ("velocity exceeds limit", "Add output saturation clamp", 1),
            ("torque overflow", "Check PID anti-windup; clamp to rated limit", 2),
            ("timeout", "Reduce waypoint count; check network latency", 0),
            ("gripper slip", "Increase grip force by 20% and retry", 1),
            ("joint_limit_exceeded", "Reduce velocity by 50% and re-plan", 2),
            ("collision_detected", "Replan with obstacle avoidance", 2),
            ("timeout", "Increase timeout or simplify task", 0),
            ("gripper_slip", "Increase grip force by 20% and retry", 1),
            ("joint overload", "Reduce payload and re-home joints; check current limits", 3),
            ("collision avoidance", "Switch to compliant mode and back off 5cm", 2),
            ("communication timeout", "Retry with exponential backoff; check ROS master", 1),
            ("grasp slippage", "Increase gripper force by 15%, approach 2cm lower, reduce lateral speed", 2),
            ("collision predicted", "Adjust trajectory and increase safety clearance", 2),
            ("object not found", "Adjust camera angle and expand search range", 1),
            ("force exceeded", "Switch to compliant mode and reduce contact force", 3),
            ("unstable grasp", "Add support point and change grasp pose", 2),
            ("path blocked", "Request obstacle clearance or replan path", 1),
            ("sensor failure", "Switch to backup sensor and verify calibration", 2),
            ("communication lost", "Retry connection and fallback to local control", 3),
        ]
        inserted = 0
        for idx, (condition, action, priority) in enumerate(defaults):
            rid = f"rule_{idx}_{condition.replace(' ', '_')[:40]}"
            try:
                # Upsert via insert (SeekDBClient.insert is INSERT OR REPLACE)
                self._seekdb.insert(self._table, {
                    "id": rid,
                    "condition": condition,
                    "action": action,
                    "priority": priority,
                    "success_count": 0,
                    "failure_count": 0,
                })
                self._rule_cache[rid] = {
                    "id": rid, "condition": condition,
                    "action": action, "priority": priority,
                    "success_count": 0, "failure_count": 0,
                }
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Seed rule %s failed: %s", rid, exc)
        self._cache_valid = True
        logger.info("HeuristicEngine seeded %d default rules", inserted)
        return inserted

    # ── internals ────────────────────────────────────────────────────────

    def _query_exact(self, error_log: str) -> Optional[dict[str, Any]]:
        """Exact condition match."""
        if self._cache_valid:
            for rule in self._rule_cache.values():
                if rule.get("condition") == error_log:
                    return rule
        # Fallback to DB
        rows = self._seekdb.query(
            self._table,
            filters={"condition": error_log},
            order_by="-priority",
            limit=1,
        )
        return dict(rows[0]) if rows else None

    def _query_substring(self, error_log: str) -> Optional[dict[str, Any]]:
        """Substring match: condition text appears inside error_log."""
        error_lower = error_log.lower()
        best: Optional[dict[str, Any]] = None
        best_pri = -999

        rules = self._rule_cache.values() if self._cache_valid else self._seekdb.query(self._table, limit=1_000)

        for rule in rules:
            cond = str(rule.get("condition", "")).lower()
            if not cond:
                continue
            if cond in error_lower:
                pri = int(rule.get("priority", 0))
                if pri > best_pri:
                    best_pri = pri
                    best = dict(rule) if not isinstance(rule, dict) else rule

        return best

    async def _knowledge_fallback(
        self,
        error_log: str,
        context: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Optional knowledge-based analogy fallback."""
        try:
            # Placeholder: if Knowledge module provides analogy lookup
            analogy = self._knowledge.find_analogy(error_log)
            if analogy:
                return {
                    "rule_id": "analogy_" + str(analogy.get("id", "")),
                    "condition": error_log,
                    "action": str(analogy.get("action_suggestion", "")),
                    "priority": 0,
                    "source": "knowledge_analogy",
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Knowledge fallback failed: %s", exc)
        return None

    def _format(self, rule: dict[str, Any]) -> dict[str, Any]:
        return {
            "rule_id": str(rule.get("id", "")),
            "condition": str(rule.get("condition", "")),
            "action": str(rule.get("action", "")),
            "priority": int(rule.get("priority", 0)),
            "source": "heuristic",
            "success_count": int(rule.get("success_count", 0)),
            "failure_count": int(rule.get("failure_count", 0)),
        }

    async def generate_recovery_hint(
        self,
        failure_type: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Generate a recovery hint for a failure type.

        This is the canonical API used by Runtime.how.generate_recovery_hint().
        """
        rule = await self.suggest_recovery(failure_type, context)
        if rule is None:
            return None
        return {
            "hint": rule.get("action", ""),
            "rule_id": rule.get("rule_id", ""),
            "priority": rule.get("priority", 0),
            "source": rule.get("source", "heuristic"),
        }

    # ── retry plan ───────────────────────────────────────────────────────

    async def get_retry_plan(
        self,
        failure_type: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Build a structured retry plan for a failure type.

        Looks up the heuristic rule and converts it into a parameter patch
        that the caller can apply on the next attempt.

        Returns:
            {
                "failure_type": str,
                "action": "retry_with_adjustments",
                "parameter_patch": dict,
                "max_retries": int,
                "rule_id": str,
            } or None.
        """
        rule = await self.suggest_recovery(failure_type, context)
        if rule is None:
            return None

        from rosclaw.how.recovery import RecoveryEngine

        re = RecoveryEngine(self, event_bus=self._event_bus)
        retry_plan = re.build_retry_plan(failure_type, rule, context)
        return retry_plan

    # CRITICAL FIX: EventBus failure event handler for active recovery
    async def _on_failure_event(self, event: Any) -> None:
        """Handle failure events from EventBus and generate recovery hints."""
        payload = event.payload if hasattr(event, "payload") else {}
        failure_type = payload.get("error_log", payload.get("reason", "unknown_failure"))
        from rosclaw.how.recovery import RecoveryEngine
        re = RecoveryEngine(self, event_bus=self._event_bus)
        await re.generate_recovery_hint(
            failure_type,
            context=payload,
            request_id=payload.get("request_id", payload.get("episode_id", "")),
        )

    # ── stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        rules = list(self._rule_cache.values()) if self._cache_valid else []
        if not rules:
            return {"rule_count": 0, "total_success": 0, "total_failure": 0}
        return {
            "rule_count": len(rules),
            "total_success": sum(int(r.get("success_count", 0)) for r in rules),
            "total_failure": sum(int(r.get("failure_count", 0)) for r in rules),
            "cache_valid": self._cache_valid,
        }
