"""Auto event subscribers — listen to external module events."""

import logging
from collections.abc import Callable
from typing import Any

try:
    from rosclaw.core.event_bus import Event
    from rosclaw.core.event_topics import EventTopics
except ImportError:
    Event = dict
    EventTopics = None

logger = logging.getLogger("rosclaw.auto.events.subscribers")


class AutoSubscriber:
    """Subscribe to Event Bus topics and route them to AutoEngine."""

    def __init__(self, engine: Any, event_bus: Any | None = None):
        self.engine = engine
        self._bus = event_bus
        self._handlers: dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        praxis_failed = (
            EventTopics.PRAXIS_FAILED if EventTopics is not None else "rosclaw.praxis.failed"
        )
        sandbox_blocked = (
            EventTopics.SANDBOX_ACTION_BLOCKED
            if EventTopics is not None
            else "rosclaw.sandbox.action.blocked"
        )
        how_recovery = (
            EventTopics.HOW_RECOVERY_HINT_GENERATED
            if EventTopics is not None
            else "rosclaw.how.recovery_hint.generated"
        )
        self._handlers[praxis_failed] = self._on_praxis_failed
        self._handlers["rosclaw.practice.failed"] = self._on_praxis_failed
        self._handlers["rosclaw.darwin.benchmark.completed"] = self._on_benchmark_completed
        self._handlers[sandbox_blocked] = self._on_sandbox_rejected
        self._handlers["rosclaw.sandbox.rejected"] = self._on_sandbox_rejected
        self._handlers[how_recovery] = self._on_how_suggestion
        self._handlers["rosclaw.how.suggestion"] = self._on_how_suggestion
        self._handlers["rosclaw.memory.insight"] = self._on_memory_insight

    def subscribe_all(self) -> None:
        if self._bus is None:
            logger.warning("No event bus provided; running in local-only mode.")
            return
        for topic, handler in self._handlers.items():
            self._bus.subscribe(topic, handler)
            logger.info("Subscribed to %s", topic)

    def unsubscribe_all(self) -> None:
        if self._bus is None:
            return
        for topic, handler in self._handlers.items():
            self._bus.unsubscribe(topic, handler)
            logger.info("Unsubscribed from %s", topic)

    def _extract_payload(self, event: Any) -> dict:
        if isinstance(event, dict):
            payload = event
        elif hasattr(event, "payload"):
            payload = event.payload if isinstance(event.payload, dict) else {}
        else:
            payload = {}
        for key in [
            "task_id",
            "skill_id",
            "failure_mode",
            "phase",
            "severity",
            "evidence",
            "event_id",
            "metrics",
            "regression_detected",
            "rejection_reason",
            "sandbox_result",
            "suggestion",
            "search_space",
            "insight_type",
            "insight_summary",
            "failure_id",
            "robot_id",
            "request_id",
            "episode_id",
            "instruction",
            "outcome",
            "final_state",
            "reason",
            "action",
            "violations",
            "hint",
            "rule_id",
            "skill_name",
        ]:
            if hasattr(event, key) and key not in payload:
                payload[key] = getattr(event, key)
        return payload

    def _on_praxis_failed(self, event: Any) -> None:
        p = self._extract_payload(event)
        task_id = p.get("task_id") or p.get("request_id") or "unknown"
        skill_id = p.get("skill_id") or p.get("skill_name") or "unknown"
        final_state = p.get("final_state") if isinstance(p.get("final_state"), dict) else {}
        outcome = p.get("outcome") if isinstance(p.get("outcome"), dict) else {}
        failure_mode = (
            p.get("failure_mode")
            or final_state.get("status")
            or outcome.get("status")
            or "runtime_failure"
        )
        phase = p.get("phase", "")
        severity = p.get("severity", "medium")
        evidence = p.get("evidence") or {
            "instruction": p.get("instruction", ""),
            "final_state": final_state,
            "outcome": outcome,
        }

        fc = self.engine.create_failure_case(
            praxis_event_id=p.get("event_id") or p.get("episode_id", ""),
            task_id=task_id,
            skill_id=skill_id,
            phase=phase,
            failure_mode=failure_mode,
            severity=severity,
            evidence=evidence,
        )
        logger.info("AutoSubscriber: created FailureCase %s for %s", fc.id, failure_mode)

        recent = self.engine.list_failures(task_id)
        if len(recent) >= self.engine.config.trigger_repeated_failure_threshold:
            prop = self.engine.create_proposal(
                failure_case_id=fc.id,
                task=task_id,
                target_skill=skill_id,
                hypothesis_statement=f"Auto-repair for repeated {failure_mode} failures",
                search_space=evidence.get("search_space", {"param_range": [0.0, 1.0]}),
                source="failure_guided",
            )
            logger.info("AutoSubscriber: auto-created Proposal %s (threshold reached)", prop.id)

    def _on_benchmark_completed(self, event: Any) -> None:
        p = self._extract_payload(event)
        if not p.get("regression_detected", False):
            return
        task_id = p.get("task_id", "unknown")
        skill_id = p.get("skill_id", "unknown")
        prop = self.engine.create_proposal(
            failure_case_id="",
            task=task_id,
            target_skill=skill_id,
            hypothesis_statement="Benchmark regression detected; auto-optimization triggered",
            search_space={"param_range": [0.0, 1.0]},
            source="benchmark_guided",
        )
        logger.info("AutoSubscriber: created benchmark-guided Proposal %s", prop.id)

    def _on_sandbox_rejected(self, event: Any) -> None:
        p = self._extract_payload(event)
        task_id = p.get("task_id") or p.get("request_id") or "unknown"
        direction = p.get("rejection_reason") or p.get("reason") or "sandbox_rejected"
        self.engine.register_deadend(
            task_id=task_id,
            direction=direction,
            rejection_reason="Sandbox rejected candidate skill",
            evidence=[str(p.get("sandbox_result") or p.get("violations", ""))],
        )
        logger.info("AutoSubscriber: registered dead-end for task %s", task_id)

    def _on_how_suggestion(self, event: Any) -> None:
        p = self._extract_payload(event)
        task_id = p.get("task_id") or p.get("request_id") or "unknown"
        skill_id = p.get("skill_id") or p.get("skill_name") or "unknown"
        suggestion = p.get("suggestion") or p.get("hint", "")
        if not suggestion:
            return
        self.engine.create_proposal(
            failure_case_id=p.get("failure_id", ""),
            task=task_id,
            target_skill=skill_id,
            hypothesis_statement=suggestion,
            search_space=p.get("search_space", {}),
            source="how_guided",
        )
        logger.info("AutoSubscriber: created how-guided Proposal for task %s", task_id)

    def _on_memory_insight(self, event: Any) -> None:
        p = self._extract_payload(event)
        insight_type = p.get("insight_type", "")
        if insight_type != "similar_failure_with_patch":
            return
        task_id = p.get("task_id", "")
        skill_id = p.get("skill_id", "")
        self.engine.create_proposal(
            failure_case_id=p.get("failure_id", ""),
            task=task_id,
            target_skill=skill_id,
            hypothesis_statement=p.get("insight_summary", "Memory-guided repair proposal"),
            search_space=p.get("search_space", {}),
            source="memory_guided",
        )
        logger.info("AutoSubscriber: created memory-guided Proposal for task %s", task_id)
