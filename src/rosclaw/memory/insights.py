"""MemoryInsightService (PR-DF-11 / flywheel §25-27; Insight 2.0 in PR-DF-22 / §33).

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

Insight 2.0 detectors (§33):
* ``known_dead_end_revisited`` — a new proposal looks like a registered
  DeadEnd (same task, high direction/hypothesis similarity).  Carries
  ``recommended_action``: skip / narrow_search / stronger_evidence so
  Evolution stops re-hitting the same wall instead of grinding again.
* ``skill_regression`` / ``skill_improvement`` — a skill's rolling
  outcome window turned against (or toward) success.
* ``harmful_recovery_pattern`` — a "proven" fix exists yet failures
  keep recurring well past the threshold: the fix is not working.

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

_STOP_TOKENS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with",
    "by", "is", "be", "it", "this", "that", "from", "as", "at",
}


def _tokens(text: str) -> set[str]:
    """Word tokens (latin) plus CJK bigrams for the dead-end similarity check."""
    import re

    out: set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", text.lower()):
        if word not in _STOP_TOKENS:
            out.add(word)
    cjk = re.findall(r"[一-鿿]+", text)
    for chunk in cjk:
        out.update(chunk[i : i + 2] for i in range(max(1, len(chunk) - 1)))
    return out


def _summaries(
    insight_type: str, skill_id: str, failure_type: str, extra: dict[str, Any]
) -> str:
    if insight_type == "repeated_failure":
        return (
            f"failure '{failure_type}' recurred {extra.get('occurrences', 3)}x "
            f"on skill '{skill_id}'"
        )
    if insight_type == "similar_failure_with_patch":
        return f"failure '{failure_type}' on skill '{skill_id}' has a proven recovery"
    if insight_type == "known_dead_end_revisited":
        return (
            f"proposal on skill '{skill_id}' resembles a known dead end "
            f"({extra.get('dead_end_direction', '')}; recommended: "
            f"{extra.get('recommended_action', 'skip')})"
        )
    if insight_type == "skill_regression":
        return (
            f"skill '{skill_id}' regressed: success "
            f"{extra.get('previous_success_rate')} -> {extra.get('recent_success_rate')}"
        )
    if insight_type == "skill_improvement":
        return (
            f"skill '{skill_id}' improved: success "
            f"{extra.get('previous_success_rate')} -> {extra.get('recent_success_rate')}"
        )
    if insight_type == "harmful_recovery_pattern":
        return (
            f"recovery pattern for '{failure_type}' on skill '{skill_id}' "
            "is not working (failures persist past 2x threshold)"
        )
    return f"{insight_type} on skill '{skill_id}'"


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
        lineage_repository: Any = None,
    ) -> None:
        self._bus = event_bus
        self._store = structured_store
        self._robot_id = robot_id
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        # PR-DF-17 (phase-II §10): MemoryInsight --derived_from--> Memory
        # edges for every contributing memory; memory_refs stays the JSON
        # attribute, lineage_edges is the traversable relation.
        self._lineage = lineage_repository
        self._counts: dict[tuple[str, str], int] = {}
        self._last_emitted: dict[tuple[str, str, str], float] = {}
        # DF-22: rolling outcome windows for skill_regression/improvement
        self._outcomes: dict[str, list[bool]] = {}
        self._subscribed = False

    # -- lifecycle ------------------------------------------------------

    def subscribe(self) -> None:
        if self._bus is None or self._subscribed:
            return
        self._bus.subscribe(EventTopics.CRITIC_JUDGMENT, self._on_judgment)
        self._bus.subscribe("rosclaw.auto.proposal.created", self._on_proposal_created)
        self._subscribed = True
        logger.info("MemoryInsightService subscribed")

    def unsubscribe(self) -> None:
        if self._bus is None or not self._subscribed:
            return
        self._bus.unsubscribe(EventTopics.CRITIC_JUDGMENT, self._on_judgment)
        self._bus.unsubscribe("rosclaw.auto.proposal.created", self._on_proposal_created)
        self._subscribed = False

    # -- observation ------------------------------------------------------

    def _on_judgment(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status", "UNKNOWN"))
        context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
        outcome = context.get("outcome", {}) if isinstance(context.get("outcome"), dict) else {}
        skill_id = outcome.get("skill_name") or payload.get("skill_id") or "unknown"

        # DF-22: rolling outcome window for skill_regression/improvement —
        # tracked for SUCCESS and non-SUCCESS alike before the failure path.
        window = self._outcomes.setdefault(skill_id, [])
        window.append(status == "SUCCESS")
        del window[:-20]
        self._maybe_emit_skill_trend(skill_id, window, payload)

        if status == "SUCCESS":
            return
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
            if count >= self._threshold * 2:
                # DF-22 harmful_recovery_pattern: a "proven" fix exists yet
                # failures keep recurring well past the threshold — the fix
                # is not actually working for this failure mode.
                self._maybe_emit(
                    "harmful_recovery_pattern",
                    skill_id=skill_id,
                    failure_type=failure_type,
                    task_id=task_id,
                    episode_id=episode_id,
                    evidence_refs=evidence_refs + fix["evidence_refs"],
                    extra={
                        "occurrences": count,
                        "memory_refs": fix["memory_refs"],
                        "summary_hint": (
                            f"recovery {fix['memory_refs']} did not stop "
                            f"'{failure_type}' ({count}x)"
                        ),
                    },
                )
                return
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

    # -- DF-22 detectors ------------------------------------------------------

    def _maybe_emit_skill_trend(
        self, skill_id: str, window: list[bool], payload: dict[str, Any]
    ) -> None:
        """skill_regression / skill_improvement from the rolling window."""
        if len(window) < 10:
            return
        first = window[: len(window) // 2]
        second = window[len(window) // 2 :]
        first_rate = sum(first) / len(first)
        second_rate = sum(second) / len(second)
        episode_id = payload.get("episode_id", "")
        evidence_refs = [f"critic_result:{episode_id}"] if episode_id else []
        if first_rate >= 0.6 and second_rate <= 0.3:
            self._maybe_emit(
                "skill_regression",
                skill_id=skill_id,
                failure_type="skill_regression",
                task_id=payload.get("task_id", ""),
                episode_id=episode_id,
                evidence_refs=evidence_refs,
                extra={
                    "window": len(window),
                    "previous_success_rate": round(first_rate, 3),
                    "recent_success_rate": round(second_rate, 3),
                },
            )
        elif first_rate <= 0.3 and second_rate >= 0.6:
            self._maybe_emit(
                "skill_improvement",
                skill_id=skill_id,
                failure_type="skill_improvement",
                task_id=payload.get("task_id", ""),
                episode_id=episode_id,
                evidence_refs=evidence_refs,
                extra={
                    "window": len(window),
                    "previous_success_rate": round(first_rate, 3),
                    "recent_success_rate": round(second_rate, 3),
                },
            )

    def _on_proposal_created(self, event: Any) -> None:
        """known_dead_end_revisited (§33): don't re-hit a registered wall."""
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        # AutoProposalCreatedEvent publishes envelope fields at top level and
        # the proposal specifics in the nested payload dict — read both.
        details = {**payload, **(payload.get("payload") or {})}
        hit = self._find_similar_dead_end(
            str(details.get("task_id") or ""),
            str(details.get("hypothesis_statement") or ""),
            str(details.get("target_skill_id") or ""),
        )
        if hit is None:
            return
        dead_end, similarity = hit
        action = "skip" if similarity >= 0.8 else "narrow_search"
        proposal_id = str(details.get("proposal_id") or "")
        self._maybe_emit(
            "known_dead_end_revisited",
            skill_id=str(details.get("target_skill_id") or "unknown"),
            failure_type="known_dead_end_revisited",
            task_id=str(details.get("task_id") or ""),
            episode_id=proposal_id,
            evidence_refs=[f"dead_end:{dead_end.get('id', '')}"],
            extra={
                "dead_end_refs": [dead_end.get("id", "")],
                "recommended_action": action,
                "similarity": round(similarity, 3),
                "dead_end_direction": dead_end.get("direction", ""),
            },
        )

    def _find_similar_dead_end(
        self, task_id: str, hypothesis: str, skill_id: str
    ) -> tuple[dict[str, Any], float] | None:
        """Best dead-end match over (direction + rejection_reason) tokens."""
        if self._store is None or not hypothesis:
            return None
        try:
            dead_ends = list(self._store.query("dead_ends", {}, limit=100_000))
        except Exception as exc:  # noqa: BLE001
            logger.debug("dead_end lookup failed: %s", exc)
            return None
        hyp_tokens = _tokens(hypothesis)
        if not hyp_tokens:
            return None
        best: tuple[dict[str, Any], float] | None = None
        for dead_end in dead_ends:
            if task_id and dead_end.get("task_id") not in (None, "", task_id):
                continue
            ref_tokens = _tokens(
                f"{dead_end.get('direction', '')} {dead_end.get('rejection_reason', '')}"
            )
            if not ref_tokens:
                continue
            jaccard = len(hyp_tokens & ref_tokens) / len(hyp_tokens | ref_tokens)
            if skill_id and skill_id in str(dead_end.get("direction", "")):
                jaccard = min(1.0, jaccard + 0.2)
            if best is None or jaccard > best[1]:
                best = (dead_end, jaccard)
        if best is not None and best[1] >= 0.5:
            return best
        return None

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
        summary = extra.get("summary_hint") or _summaries(insight_type, skill_id, failure_type, extra)
        confidence = (
            0.8
            if insight_type in ("similar_failure_with_patch", "known_dead_end_revisited")
            else 0.5
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
            "confidence": confidence,
            "created_at": now,
            # DF-22 extension fields (absent for the DF-11 types)
            "dead_end_refs": extra.get("dead_end_refs", []),
            "recommended_action": extra.get("recommended_action", ""),
            "similarity": extra.get("similarity"),
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
        self._link_sources(insight["insight_id"], insight["memory_refs"])
        for dead_end_ref in insight["dead_end_refs"]:
            self._link_typed(insight["insight_id"], "dead_end", str(dead_end_ref))

    def _link_typed(self, insight_id: str, entity_type: str, entity_id: str) -> None:
        if self._lineage is None or not entity_id:
            return
        from rosclaw.storage.lineage_types import LineageEntityType, LineageRelation

        try:
            self._lineage.link(
                str(LineageEntityType.MEMORY_INSIGHT),
                insight_id,
                str(LineageRelation.DERIVED_FROM),
                entity_type,
                entity_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("insight lineage link failed (non-fatal): %s", exc)

    def _link_sources(self, insight_id: str, memory_refs: list[str]) -> None:
        """MemoryInsight --derived_from--> each contributing memory (§10)."""
        if self._lineage is None:
            return
        from rosclaw.storage.lineage_types import LineageEntityType, LineageRelation

        for ref in memory_refs:
            ref = str(ref)
            if not ref:
                continue
            if ref.startswith("heuristic_rules:"):
                entity_type, entity_id = "heuristic_rule", ref.split(":", 1)[1]
            else:
                entity_type, entity_id = str(LineageEntityType.MEMORY), ref
            if not entity_id:
                continue
            try:
                self._lineage.link(
                    str(LineageEntityType.MEMORY_INSIGHT),
                    insight_id,
                    str(LineageRelation.DERIVED_FROM),
                    entity_type,
                    entity_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("insight lineage link failed (non-fatal): %s", exc)
