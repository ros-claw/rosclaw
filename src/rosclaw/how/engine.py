"""rosclaw_how.engine — HeuristicEngine: fast rule-based recovery.

The engine provides:
  * suggest_recovery()  — lookup heuristic_rules table (<10ms)
  * record_outcome()    — update rule efficacy counters
  * seed_defaults()     — populate initial safety rules
  * decide_recovery()   — InterventionRequest → InterventionDecision +
                          rule_id (so outcome tracking covers the
                          proactive intervention layer too)

Design:
  - Pure rule-based, zero LLM calls in the hot path.
  - Condition matching: exact → reactive substring → SAFETY_TAXONOMY
    fallback → knowledge analogy. The taxonomy ships as the fallback so
    hand-tuned reactive remediations (e.g. "compliant mode", "exponential
    backoff") still win when both could match. ``tax_<symptom>`` IDs
    never collide with reactive ``rule_<n>_<slug>`` IDs.
  - Outcome tracking: success_count / failure_count / priority.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any, Final

from .intervention import (
    SAFETY_TAXONOMY,
    InterventionDecision,
    InterventionRequest,
    compose,
    decide_strategy,
    diagnose,
    diagnose_safety,
    is_blocking,
    symptom_category,
)

logger = logging.getLogger("rosclaw.how.engine")

# Rule-ID prefix for SAFETY_TAXONOMY-derived rules. Distinct prefix so the
# reactive rule_id namespace (``rule_<n>_<slug>``) never collides.
_TAXONOMY_RULE_PREFIX: Final[str] = "tax_"

# S0 (info) → 0 … S4 (emergency stop) → 4. Hoisted to module level so the
# dict is built once at import time, not per-call.
_SEVERITY_PRIORITY: Final[dict[str, int]] = {
    "S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4,
}


def _taxonomy_rule_id(symptom: str) -> str:
    """Stable rule id for a taxonomy symptom (e.g. ``tax_Collision_Risk``)."""
    return f"{_TAXONOMY_RULE_PREFIX}{symptom}"


def _severity_priority(severity: str) -> int:
    """Map S0-S4 to a comparable int. S4 (emergency) has the highest priority."""
    return _SEVERITY_PRIORITY.get(severity, 0)


def _format_taxonomy_action(symptom: str, severity: str, strategy: str) -> str:
    """Human-readable action text for a taxonomy-derived rule."""
    return f"[{severity}/{strategy}] inject {symptom} guard (safety taxonomy)"


class HeuristicEngine:
    """Fast heuristic rule engine backed by SeekDB.

    Args:
        seekdb_client: Any SeekDBClient implementation (memory / SQLite).
        knowledge_interface: Optional knowledge interface for analogy fallback.
    """

    def __init__(
        self,
        seekdb_client: Any,
        knowledge_interface: Any | None = None,
        event_bus: Any | None = None,
        sense_runtime: Any | None = None,
    ) -> None:
        self._seekdb = seekdb_client
        self._knowledge = knowledge_interface
        self._event_bus = event_bus
        self._sense_runtime = sense_runtime
        self._table = "heuristic_rules"
        self._rule_cache: dict[str, dict[str, Any]] = {}
        self._cache_valid = False
        self._subscribed_topics: list[tuple[str, Any]] = []
        self._how_context_adapter: Any | None = None
        if sense_runtime is not None:
            try:
                from rosclaw.sense.adapters.how_context import HowContextAdapter
                self._how_context_adapter = HowContextAdapter(sense_runtime)
            except Exception:
                logger.warning("Failed to initialize HowContextAdapter", exc_info=True)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Warm the rule cache from SeekDB and subscribe to failure events."""
        try:
            rows = self._seekdb.query(self._table, limit=1_000)
            warmed = {str(r.get("id", "")): dict(r) for r in rows if r.get("id")}
            # Preserve any lazily-seeded taxonomy rows that haven't been
            # flushed to seekdb yet (best-effort persistence). Without
            # this merge, a re-initialize() would silently drop the
            # cached counters built up via `_taxonomy_rule`.
            for rid, row in self._rule_cache.items():
                if rid.startswith(_TAXONOMY_RULE_PREFIX) and rid not in warmed:
                    warmed[rid] = row
            self._rule_cache = warmed
            self._cache_valid = True
            logger.info("HeuristicEngine warmed %d rules", len(self._rule_cache))
        except Exception as exc:  # noqa: BLE001
            logger.warning("HeuristicEngine warm failed: %s", exc)
            self._cache_valid = False

        # CRITICAL FIX: subscribe to failure events on EventBus for active recovery
        # Store (topic, callback) tuples for later unsubscribe
        if self._event_bus is not None:
            try:
                self._event_bus.subscribe("praxis.failed", self._on_failure_sync_wrapper)
                self._subscribed_topics.append(("praxis.failed", self._on_failure_sync_wrapper))
                self._event_bus.subscribe("firewall.action_blocked", self._on_failure_sync_wrapper)
                self._subscribed_topics.append(("firewall.action_blocked", self._on_failure_sync_wrapper))
                self._event_bus.subscribe("safety.violation", self._on_failure_sync_wrapper)
                self._subscribed_topics.append(("safety.violation", self._on_failure_sync_wrapper))
                logger.info("HeuristicEngine subscribed to failure events")
            except Exception as exc:  # noqa: BLE001
                logger.warning("HeuristicEngine EventBus subscribe failed: %s", exc)

    async def shutdown(self) -> None:
        """Clear cache and unsubscribe from EventBus."""
        self._rule_cache.clear()
        self._cache_valid = False
        # CRITICAL FIX: unsubscribe from EventBus to prevent leaks
        if self._event_bus is not None:
            for topic, handler in self._subscribed_topics:
                with contextlib.suppress(Exception):
                    self._event_bus.unsubscribe(topic, handler)
        self._subscribed_topics.clear()

    # ── public API ───────────────────────────────────────────────────────

    async def suggest_recovery(
        self,
        error_log: str,
        context: dict[str, Any] | None = None,
        *,
        previous_scores: list[float] | None = None,
        current_iteration: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the best matching recovery strategy (<10ms when cached).

        Matching priority:
          1. Exact condition match on error_log (fastest)
          2. Substring match: condition text appears inside error_log
          3. Keyword overlap between error_log and condition

        ``previous_scores`` and ``current_iteration`` are accepted for API
        symmetry with :class:`HowClient` but are ignored by the local
        heuristic engine — it has no plateau-detection state router.

        Returns:
            {"rule_id": str, "condition": str, "action": str,
             "priority": int, "source": "heuristic"} or None.
        """
        if not error_log:
            return None

        # 1. Exact match (reactive rules in seekdb / cache)
        result = self._query_exact(error_log)
        if result:
            return self._format(result)

        # 2. Substring match against reactive cached rules. We try this BEFORE
        #    the taxonomy because the reactive rule library carries
        #    hand-written remediations ("Switch to compliant mode and back
        #    off 5cm") that are more actionable than the taxonomy's
        #    generic "[S2/SAFETY] inject Workspace_Boundary guard"
        #    template. Taxonomy is the fallback for symptoms reactive rules
        #    don't cover.
        result = self._query_substring(error_log)
        if result:
            return self._format(result)

        # 3. SAFETY_TAXONOMY lookup — extended S0-S4 vocabulary that
        #    covers physical-AI failure modes (collision, self-collision,
        #    OOM, NaN, …) the reactive rule list doesn't. When a symptom
        #    matches we return a synthetic rule keyed by ``tax_<symptom>``
        #    so ``record_outcome`` can track efficacy the same way as
        #    reactive rules.
        symptom, severity, strategy = diagnose_safety(error_log)
        if symptom is not None:
            tax_rule = self._taxonomy_rule(symptom, severity, strategy)
            if tax_rule is not None:
                return self._format(tax_rule)

        # 4. Knowledge fallback (optional)
        if self._knowledge:
            return await self._knowledge_fallback(error_log, context)

        return None

    async def advise(
        self,
        body_id: str,
        failure: str,
        episode_id: str,
        data_root: str | None = None,
    ) -> dict[str, Any]:
        """Return an evidence-backed intervention for a failed episode.

        Loads the raw practice events from ``data_root`` and uses the
        heuristic rule engine (plus optional knowledge analogy) to propose a
        recovery action.  The returned dict includes the episode evidence so
        the advice is traceable.
        """
        from pathlib import Path

        from rosclaw.practice.storage.layout import PracticeLayout

        root = Path(data_root or "/data/rosclaw/practice")
        layout = PracticeLayout(root)
        session_dir = layout.session_dir(episode_id)

        events: list[dict[str, Any]] = []
        if session_dir.exists():
            events_path = layout.events_jsonl_path(episode_id)
            if events_path.exists():
                try:
                    with open(events_path, encoding="utf-8") as f:
                        events = [json.loads(line) for line in f if line.strip()]
                except Exception as exc:
                    logger.warning("Failed to read episode events for HOW advise: %s", exc)

        error_log = f"{failure} on body {body_id} (episode {episode_id})"
        recovery = await self.suggest_recovery(
            error_log,
            context={
                "body_id": body_id,
                "episode_id": episode_id,
                "events": events,
            },
        )
        if recovery is None:
            recovery = {
                "rule_id": "fallback",
                "condition": failure,
                "action": "Review episode artifacts and logs for root cause.",
                "priority": 0,
                "source": "fallback",
            }

        return {
            "body_id": body_id,
            "failure": failure,
            "episode_id": episode_id,
            "intervention": recovery,
            "evidence": {
                "event_count": len(events),
                "sources": sorted({ev.get("source") for ev in events}),
            },
        }

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
        """Populate heuristic_rules with default safety rules.

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

        # Append the SAFETY_TAXONOMY rows so they show up in
        # ``list_rules`` / dashboards and ``record_outcome`` can attribute
        # successes to them. Action text is the SAFETY snippet's first line,
        # priority is the severity ordinal (S0=0 … S4=4).
        for symptom, entry in SAFETY_TAXONOMY.items():
            rid = _taxonomy_rule_id(symptom)
            severity = str(entry.get("severity", "S0"))
            strategy = str(entry.get("strategy", "NOOP"))
            condition = symptom.replace("_", " ").lower()
            action = _format_taxonomy_action(symptom, severity, strategy)
            priority = _severity_priority(severity)
            try:
                self._seekdb.insert(self._table, {
                    "id": rid,
                    "condition": condition,
                    "action": action,
                    "priority": priority,
                    "success_count": 0,
                    "failure_count": 0,
                })
                self._rule_cache[rid] = {
                    "id": rid,
                    "condition": condition,
                    "action": action,
                    "priority": priority,
                    "success_count": 0,
                    "failure_count": 0,
                    "severity": severity,
                    "strategy": strategy,
                    "source_taxonomy": True,
                }
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Seed taxonomy rule %s failed: %s", rid, exc)

        self._cache_valid = True
        logger.info("HeuristicEngine seeded %d default rules", inserted)
        return inserted

    # ── internals ────────────────────────────────────────────────────────

    def _query_exact(self, error_log: str) -> dict[str, Any] | None:
        """Exact condition match.

        Taxonomy rows (id prefix ``tax_``) are skipped here so a bare
        symptom string like ``"joint limit violation"`` doesn't pre-empt
        a reactive rule whose condition is a substring. The taxonomy has
        its own match path via ``diagnose_safety`` in ``suggest_recovery``.
        """
        if self._cache_valid:
            for rule in self._rule_cache.values():
                rid = str(rule.get("id", ""))
                if rid.startswith(_TAXONOMY_RULE_PREFIX):
                    continue
                if rule.get("condition") == error_log:
                    return rule
        # Fallback to DB
        rows = self._seekdb.query(
            self._table,
            filters={"condition": error_log},
            order_by="-priority",
            limit=1,
        )
        # Filter out taxonomy hits the DB may surface so the policy still
        # holds when the cache is cold.
        for row in rows:
            rid = str(row.get("id", ""))
            if rid.startswith(_TAXONOMY_RULE_PREFIX):
                continue
            return dict(row)
        return None

    def _query_substring(self, error_log: str) -> dict[str, Any] | None:
        """Substring match: condition text appears inside error_log.

        Taxonomy rows (id prefix ``tax_``) are skipped here — they have
        their own match path via ``diagnose_safety`` in ``suggest_recovery``
        which honors keyword tuples / severity, not lowercase substrings.
        """
        error_lower = error_log.lower()
        best: dict[str, Any] | None = None
        best_pri = -999

        rules = self._rule_cache.values() if self._cache_valid else self._seekdb.query(self._table, limit=1_000)

        for rule in rules:
            rid = str(rule.get("id", ""))
            if rid.startswith(_TAXONOMY_RULE_PREFIX):
                # Reserved for the taxonomy match path
                continue
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
        context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
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

    # ── SAFETY_TAXONOMY bridge ───────────────────────────────────────────

    def _taxonomy_rule(
        self,
        symptom: str,
        severity: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        """Return (or lazily seed) the cached rule for a taxonomy symptom.

        Allows ``suggest_recovery`` to surface S2/S3/S4 events even before
        ``seed_defaults`` is called — the row is created on demand and
        stored in the local cache; the DB row is upserted best-effort so
        ``record_outcome`` can still find it later.
        """
        rid = _taxonomy_rule_id(symptom)
        cached = self._rule_cache.get(rid)
        if cached is not None:
            return cached
        condition = symptom.replace("_", " ").lower()
        action = _format_taxonomy_action(symptom, severity, strategy)
        priority = _severity_priority(severity)
        row: dict[str, Any] = {
            "id": rid,
            "condition": condition,
            "action": action,
            "priority": priority,
            "success_count": 0,
            "failure_count": 0,
            "severity": severity,
            "strategy": strategy,
            "source_taxonomy": True,
        }
        # Persist best-effort so record_outcome can update counters across
        # restarts. The cache is the source of truth for the hot path.
        try:
            self._seekdb.insert(self._table, {
                k: v for k, v in row.items()
                if k not in ("severity", "strategy", "source_taxonomy")
            })
        except Exception as exc:  # noqa: BLE001 — non-fatal on read path
            logger.debug("Lazy taxonomy insert failed for %s: %s", rid, exc)
        self._rule_cache[rid] = row
        return row

    async def decide_recovery(
        self,
        request: InterventionRequest,
        *,
        recent_pattern_id: str | None = None,
    ) -> tuple[InterventionDecision, str | None]:
        """Run the proactive diagnose → policy → composer pipeline.

        Returns ``(decision, rule_id_or_None)``. When the final decision
        was driven by a safety symptom (``decision.strategy`` is one of
        SAFETY / STOP_UNSAFE / RESOURCE_REPAIR — see :func:`is_blocking`)
        AND that symptom has a taxonomy entry, the returned ``rule_id``
        is ``tax_<symptom>``. Callers pass it to :meth:`record_outcome`
        to keep efficacy counters.

        For decisions driven by the *optimization* or *feasibility* axis
        (CATALYST plateau / DIVERSIFY cooldown / FEASIBILITY_REPAIR /
        STABILIZE / EXPLOIT_BEST / DIAGNOSE / NOOP) the rule_id is
        ``None`` even if a symptom was *also* detected — crediting a
        taxonomy rule for an outcome that wasn't its decision would
        skew the efficacy counters.

        This method is ``async`` purely for caller symmetry with
        :meth:`suggest_recovery`; the body is pure sync and does not
        await anything. Future maintainers: do not assume the body
        contains awaitables.
        """
        state = diagnose(request)
        strategy, _reasons = decide_strategy(
            request, state, recent_pattern_id=recent_pattern_id,
        )
        # Software-resource symptoms (Memory_Exhaustion / Compile_Error) that
        # fell through the safety dispatch require a curated cluster match
        # before the composer emits anything — an off-topic synth cluster
        # would actively mislead the LLM (see safety_router.symptom_category).
        require_curated_match = (
            state.safety_symptom is not None
            and symptom_category(state.safety_symptom) == "software_resource"
        )
        decision = compose(
            strategy,
            state,
            recent_pattern_id=recent_pattern_id,
            require_curated_match=require_curated_match,
        )
        rule_id: str | None = None
        # Attribute the outcome to the taxonomy ONLY when the final
        # decision was actually a safety-branch outcome — this protects
        # the counters when cooldown swaps the strategy to DIVERSIFY or
        # the optimization axis overrides the safety classification
        # (e.g. an S1 Battery_Low symptom with a plateau-CATALYST
        # decision should not credit tax_Battery_Low).
        if (
            state.safety_symptom
            and state.safety_symptom in SAFETY_TAXONOMY
            and is_blocking(decision.strategy)
        ):
            entry = SAFETY_TAXONOMY[state.safety_symptom]
            self._taxonomy_rule(
                state.safety_symptom,
                str(entry.get("severity", "S0")),
                str(entry.get("strategy", "NOOP")),
            )
            rule_id = _taxonomy_rule_id(state.safety_symptom)
        return decision, rule_id

    async def generate_recovery_hint(
        self,
        failure_type: str,
        context: dict[str, Any] | None = None,
        *,
        previous_scores: list[float] | None = None,
        current_iteration: int | None = None,
    ) -> dict[str, Any] | None:
        """Generate a recovery hint for a failure type.

        This is the canonical API used by Runtime.how.generate_recovery_hint().
        ``previous_scores`` and ``current_iteration`` are forwarded for
        compatibility with service-backed engines that route on plateau state.
        """
        enriched_context = dict(context or {})
        enriched_context.setdefault("task", failure_type)
        if self._how_context_adapter is not None:
            try:
                enriched_context = self._how_context_adapter.apply(enriched_context)
            except Exception:
                logger.warning("HowContextAdapter failed for recovery hint", exc_info=True)
        rule = await self.suggest_recovery(
            failure_type, enriched_context,
            previous_scores=previous_scores, current_iteration=current_iteration,
        )
        if rule is None:
            return None
        return {
            "hint": rule.get("action", ""),
            "rule_id": rule.get("rule_id", ""),
            "priority": rule.get("priority", 0),
            "source": rule.get("source", "heuristic"),
            "body_readiness": enriched_context.get("body_readiness"),
            "body_block_reasons": enriched_context.get("body_block_reasons"),
        }

    # ── retry plan ───────────────────────────────────────────────────────

    async def get_retry_plan(
        self,
        failure_type: str,
        context: dict[str, Any] | None = None,
        *,
        previous_scores: list[float] | None = None,
        current_iteration: int | None = None,
    ) -> dict[str, Any] | None:
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
        rule = await self.suggest_recovery(
            failure_type, context,
            previous_scores=previous_scores, current_iteration=current_iteration,
        )
        if rule is None:
            return None

        from rosclaw.how.recovery import RecoveryEngine

        re = RecoveryEngine(self, event_bus=self._event_bus)
        retry_plan = re.build_retry_plan(failure_type, rule, context)
        return retry_plan

    def _on_failure_sync_wrapper(self, event: Any) -> None:
        """Sync EventBus callback that schedules async recovery handling."""
        from rosclaw.core.async_utils import fire_and_forget
        fire_and_forget(self._on_failure_async(event))

    # CRITICAL FIX: EventBus failure event handler for active recovery
    async def _on_failure_async(self, event: Any) -> None:
        """Handle failure events from EventBus and generate recovery hints."""
        payload = event.payload if hasattr(event, "payload") else {}
        failure_type = payload.get("error_log", payload.get("reason", "unknown_failure"))
        context = dict(payload)
        context.setdefault("task", failure_type)
        if self._how_context_adapter is not None:
            try:
                context = self._how_context_adapter.apply(context)
            except Exception:
                logger.warning("HowContextAdapter failed for failure event", exc_info=True)
        from rosclaw.how.recovery import RecoveryEngine
        re = RecoveryEngine(self, event_bus=self._event_bus)
        coro = re.generate_recovery_hint(
            failure_type,
            context=context,
            request_id=payload.get("request_id", payload.get("episode_id", "")),
        )
        hint = await coro
        if hint and self._event_bus is not None:
            event_payload = re.format_for_eventbus(hint, request_id=payload.get("request_id", payload.get("episode_id", "")))
            event_payload["body_readiness"] = context.get("body_readiness")
            event_payload["body_block_reasons"] = context.get("body_block_reasons")
            from rosclaw.core.event_bus import Event, EventPriority
            self._event_bus.publish(Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload=event_payload,
                source="heuristic_engine",
                priority=EventPriority.HIGH,
            ))

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
