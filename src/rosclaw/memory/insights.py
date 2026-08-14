"""MemoryInsightService (PR-DF-11 / flywheel §25-27).

The publisher half of ``rosclaw.memory.insight``: until now Auto subscribed
to the topic and nothing ever published.  This service watches failure
observations (critic judgments) and, when a failure recurs, checks the
structured store for *successful* recovery experience against the same
failure type — only then does it publish a typed insight:

* ``repeated_failure`` — a failure_type keeps recurring (>= threshold);
* ``similar_failure_with_patch`` — the recurring failure has a proven
  recovery (a How rule with successes, or a successful intervention
  memory).  This is the exact payload Auto's subscriber turns into a
  memory-guided Proposal (§27).

Insights carry the canonical typed fields (§26): insight_id, insight_type,
robot_id, body_id, task_id, skill_id, failure_type, evidence_refs,
memory_refs, recommended_search_space, confidence — plus the legacy field
names Auto reads today (insight_summary, search_space, failure_id).

Dedup: one insight per (insight_type, skill_id, failure_type) per cooldown
window so a failure storm doesn't flood Auto with duplicate proposals.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from rosclaw.core.event_topics import EventTopics

logger = logging.getLogger("rosclaw.memory.insights")

_DEFAULT_COOLDOWN_S = 900.0


class MemoryInsightService:
    """Generate typed memory insights from recurring failures + proven fixes."""

    def __init__(
        self,
        event_bus: Any,
        structured_store: Any,
        *,
        robot_id: str,
        failure_threshold: int = 3,
        cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        self._bus = event_bus
        self._store = structured_store
        self._robot_id = robot_id
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._counts: dict[tuple[str, str], int] = {}
        self._last_emitted: dict[tuple[str, str, str], float] = {}
        self._subscribed = False

    # -- lifecycle ------------------------------------------------------

    def subscribe(self) -> None:
        if self._bus is None or self._subscribed:
            return
        self._bus.subscribe(EventTopics.CRITIC_JUDGMENT, self._on_judgment)
        self._subscribed = True
        logger.info("MemoryInsightService subscribed")

    def unsubscribe(self) -> None:
        if self._bus is None or not self._subscribed:
            return
        self._bus.unsubscribe(EventTopics.CRITIC_JUDGMENT, self._on_judgment)
        self._subscribed = False

    # -- observation ------------------------------------------------------

    def _on_judgment(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status", "UNKNOWN"))
        if status == "SUCCESS":
            return
        context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
        outcome = context.get("outcome", {}) if isinstance(context.get("outcome"), dict) else {}
        skill_id = outcome.get("skill_name") or payload.get("skill_id") or "unknown"
        failure_type = payload.get("reason") or payload.get("failure_type") or "unknown"
        task_id = payload.get("task_id") or context.get("instruction") or ""
        episode_id = payload.get("episode_id", "")

        key = (skill_id, failure_type)
        self._counts[key] = self._counts.get(key, 0) + 1
        count = self._counts[key]
        if count < self._threshold:
            return

        evidence_refs = [f"critic_result:{episode_id}"] if episode_id else []
        self._maybe_emit(
            "repeated_failure",
            skill_id=skill_id,
            failure_type=failure_type,
            task_id=task_id,
            episode_id=episode_id,
            evidence_refs=evidence_refs,
            extra={"occurrences": count},
        )
        fix = self._find_proven_fix(failure_type)
        if fix is not None:
            self._maybe_emit(
                "similar_failure_with_patch",
                skill_id=skill_id,
                failure_type=failure_type,
                task_id=task_id,
                episode_id=episode_id,
                evidence_refs=evidence_refs + fix["evidence_refs"],
                extra={
                    "search_space": fix["search_space"],
                    "memory_refs": fix["memory_refs"],
                },
            )

    # -- insight emission ---------------------------------------------------

    def _find_proven_fix(self, failure_type: str) -> dict[str, Any] | None:
        """A How rule with verified successes, or a successful intervention memory."""
        if self._store is None:
            return None
        try:
            rules = self._store.query("heuristic_rules", {"failure_type": failure_type})
            proven = [
                r
                for r in rules
                if isinstance(r.get("success_count"), int) and r["success_count"] > 0
            ]
            if proven:
                best = max(proven, key=lambda r: r["success_count"])
                search_space = best.get("parameter_patch") or {}
                if isinstance(search_space, str):
                    import json

                    try:
                        search_space = json.loads(search_space)
                    except ValueError:
                        search_space = {}
                return {
                    "search_space": search_space if isinstance(search_space, dict) else {},
                    "memory_refs": [f"heuristic_rules:{best.get('id', '')}"],
                    "evidence_refs": [f"heuristic_rules:{best.get('id', '')}"],
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("rule lookup failed: %s", exc)
        try:
            interventions = self._store.query(
                "memory_items", {"memory_type": "intervention", "failure_type": failure_type}
            )
            successful = [m for m in interventions if m.get("outcome") == "SUCCESS"]
            if successful:
                mem = successful[0]
                return {
                    "search_space": {},
                    "memory_refs": [mem.get("id", "")],
                    "evidence_refs": list(mem.get("evidence_refs") or []),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("intervention memory lookup failed: %s", exc)
        return None

    def _maybe_emit(
        self,
        insight_type: str,
        *,
        skill_id: str,
        failure_type: str,
        task_id: str,
        episode_id: str,
        evidence_refs: list[str],
        extra: dict[str, Any],
    ) -> None:
        dedup_key = (insight_type, skill_id, failure_type)
        now = time.time()
        if now - self._last_emitted.get(dedup_key, 0.0) < self._cooldown_s:
            return
        self._last_emitted[dedup_key] = now
        search_space = extra.get("search_space", {})
        summary = (
            f"failure '{failure_type}' recurred {extra.get('occurrences', self._threshold)}x "
            f"on skill '{skill_id}'"
            if insight_type == "repeated_failure"
            else f"failure '{failure_type}' on skill '{skill_id}' has a proven recovery"
        )
        insight = {
            # canonical typed fields (§26)
            "insight_id": f"ins_{uuid.uuid4().hex[:16]}",
            "insight_type": insight_type,
            "robot_id": self._robot_id,
            "task_id": task_id,
            "skill_id": skill_id,
            "failure_type": failure_type,
            "evidence_refs": evidence_refs,
            "memory_refs": extra.get("memory_refs", []),
            "recommended_search_space": search_space,
            "confidence": 0.8 if insight_type == "similar_failure_with_patch" else 0.5,
            "created_at": now,
            # field names Auto's subscriber consumes today
            "insight_summary": summary,
            "search_space": search_space,
            "failure_id": episode_id,
        }
        try:
            from rosclaw.core.event_bus import Event

            self._bus.publish(
                Event(
                    topic=EventTopics.MEMORY_INSIGHT_CREATED,
                    payload=insight,
                    source="memory.insights",
                )
            )
            logger.info("memory insight published: %s (%s)", insight_type, dedup_key)
        except Exception as exc:  # noqa: BLE001
            logger.info("insight publish failed (non-fatal): %s", exc)
