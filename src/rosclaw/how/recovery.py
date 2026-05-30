"""rosclaw_how.recovery — RecoveryEngine: event-driven recovery hint generation.

The RecoveryEngine bridges failure events to actionable recovery hints.
It subscribes to failure topics on the EventBus and publishes
rosclaw.how.recovery_hint.generated events autonomously.

Design:
  - Event-driven: subscribes to failure topics, publishes recovery hints.
  - Multi-rule matching: ranks all matching rules by confidence.
  - Context-aware: considers robot_id, task_type, history.
  - RecoveryHint includes confidence scoring and provenance tracking.
  - RetryPlan is a structured parameter patch for the next attempt.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

logger = logging.getLogger("rosclaw.how.recovery")


class RecoveryEngine:
    """Event-driven RecoveryHint generator.

    Can operate in two modes:
      - Passive: caller invokes generate_recovery_hint() directly.
      - Active: subscribes to failure topics on EventBus, publishes
        recovery hints autonomously.

    Subscribes to:
      - rosclaw.sandbox.episode.failed
      - rosclaw.sandbox.action.blocked
      - rosclaw.runtime.execution.failed

    Publishes:
      - rosclaw.how.recovery_hint.generated
      - rosclaw.how.recovery_hint.failed (when no rule matches)

    Args:
        heuristic_engine: HeuristicEngine instance for rule lookup.
        event_bus: Optional EventBus for autonomous event-driven mode.
    """

    def __init__(
        self,
        heuristic_engine: Any,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._how = heuristic_engine
        self._event_bus = event_bus
        self._subscribed = False

    # -- lifecycle --------------------------------------------------------

    def initialize(self) -> None:
        """Subscribe to failure topics on the EventBus."""
        if self._event_bus is None:
            logger.debug("RecoveryEngine initialized without EventBus (passive mode)")
            return

        self._event_bus.subscribe(
            "rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed
        )
        self._event_bus.subscribe(
            "rosclaw.sandbox.action.blocked", self._on_sandbox_action_blocked
        )
        self._event_bus.subscribe(
            "rosclaw.runtime.execution.failed", self._on_runtime_execution_failed
        )
        self._subscribed = True
        logger.info("RecoveryEngine subscribed to 3 failure topics")

    def shutdown(self) -> None:
        """Unsubscribe from failure topics."""
        if self._event_bus is None or not self._subscribed:
            return

        self._event_bus.unsubscribe(
            "rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed
        )
        self._event_bus.unsubscribe(
            "rosclaw.sandbox.action.blocked", self._on_sandbox_action_blocked
        )
        self._event_bus.unsubscribe(
            "rosclaw.runtime.execution.failed", self._on_runtime_execution_failed
        )
        self._subscribed = False
        logger.info("RecoveryEngine unsubscribed from failure topics")

    # -- event handlers ---------------------------------------------------

    def _on_sandbox_episode_failed(self, event: Any) -> None:
        """Handle sandbox episode failure: generate and publish recovery hint."""
        self._handle_failure_event(event, source="sandbox_episode")

    def _on_sandbox_action_blocked(self, event: Any) -> None:
        """Handle sandbox action blocked: generate and publish recovery hint."""
        self._handle_failure_event(event, source="sandbox_action_blocked")

    def _on_runtime_execution_failed(self, event: Any) -> None:
        """Handle runtime execution failure: generate and publish recovery hint."""
        self._handle_failure_event(event, source="runtime_execution")

    def _handle_failure_event(self, event: Any, source: str) -> None:
        """Core handler: extract failure info, generate hint, publish event."""
        if self._how is None or self._event_bus is None:
            return

        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            return

        failure_type = payload.get("failure_type", "")
        request_id = payload.get("request_id", "")
        if not failure_type:
            error_log = payload.get("error_log", "")
            violations = payload.get("violations", [])
            if error_log:
                failure_type = error_log
            elif violations:
                failure_type = "; ".join(
                    v.get("description", "") for v in violations
                )

        if not failure_type:
            return

        import asyncio

        try:
            hint = asyncio.run(self.generate_recovery_hint(
                failure_type,
                context={
                    "request_id": request_id,
                    "source": source,
                    "event_payload": payload,
                },
                sources=[source],
            ))

            if hint:
                event_payload = self.format_for_eventbus(hint, request_id=request_id)
                from rosclaw.core.event_bus import Event, EventPriority

                self._event_bus.publish(Event(
                    topic="rosclaw.how.recovery_hint.generated",
                    payload=event_payload,
                    source="recovery_engine",
                    priority=EventPriority.HIGH,
                ))
                logger.info("Published recovery hint for %s (req=%s)", failure_type, request_id)
            else:
                from rosclaw.core.event_bus import Event, EventPriority

                self._event_bus.publish(Event(
                    topic="rosclaw.how.recovery_hint.failed",
                    payload={
                        "request_id": request_id,
                        "failure_type": failure_type,
                        "reason": "no_matching_rule",
                    },
                    source="recovery_engine",
                    priority=EventPriority.HIGH,
                ))
                logger.warning("No recovery hint for %s (req=%s)", failure_type, request_id)
        except Exception as exc:
            logger.error("RecoveryEngine handler failed: %s", exc)

    # -- public API -------------------------------------------------------

    async def generate_recovery_hint(
        self,
        failure_type: str,
        context: Optional[dict[str, Any]] = None,
        sources: Optional[list[str]] = None,
        event_bus: Optional[Any] = None,
        request_id: str = "",
    ) -> Optional[dict[str, Any]]:
        """Build a RecoveryHint dict from failure metadata.

        Uses multi-rule matching: ranks all matching rules by confidence
        and returns the best one. If no heuristic rule matches, falls
        back to knowledge analogy.

        Returns:
            {
                "failure_type": str,
                "hint": str,
                "confidence": float,
                "source": list[str],
                "retry_plan": dict,
                "all_candidates": list[dict],
            } or None.
        """
        if not failure_type:
            return None

        ctx = context or {}
        srcs = sources or []

        # Multi-rule matching: get all matching rules, rank by confidence
        candidates = await self._find_all_candidates(failure_type, ctx)
        if not candidates:
            return None

        best = candidates[0]
        confidence = best.get("_confidence", 0.5)

        # Build retry plan
        retry_plan = self.build_retry_plan(failure_type, best, ctx)

        hint = {
            "failure_type": failure_type,
            "hint": best.get("action", ""),
            "confidence": round(confidence, 2),
            "source": srcs + [f"heuristic:{best.get('rule_id', 'unknown')}"],
            "retry_plan": retry_plan,
            "all_candidates": [
                {
                    "rule_id": c.get("rule_id", ""),
                    "action": c.get("action", ""),
                    "confidence": round(c.get("_confidence", 0.5), 2),
                }
                for c in candidates[:3]
            ],
        }
        logger.info(
            "RecoveryHint for %s: confidence=%.2f, candidates=%d",
            failure_type, confidence, len(candidates)
        )

        # Publish recovery_hint.generated to EventBus
        if event_bus is not None:
            try:
                event_bus.publish({
                    "topic": "rosclaw.how.recovery_hint.generated",
                    "payload": self.format_for_eventbus(hint, request_id=request_id),
                })
            except Exception as exc:
                logger.warning("Failed to publish recovery_hint event: %s", exc)

        return hint

    async def _find_all_candidates(
        self,
        failure_type: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Find all matching rules and rank them by confidence score."""
        if self._how is None:
            return []

        candidates: list[dict[str, Any]] = []

        try:
            rule = await self._how.suggest_recovery(failure_type, context=context)
            if rule:
                rule["_confidence"] = self._compute_confidence(rule)
                candidates.append(rule)
        except Exception as exc:
            logger.debug("Heuristic lookup failed: %s", exc)

        if not candidates and hasattr(self._how, "_knowledge") and self._how._knowledge:
            try:
                fallback = await self._how._knowledge_fallback(failure_type, context)
                if fallback:
                    fallback["_confidence"] = 0.3
                    candidates.append(fallback)
            except Exception as exc:
                logger.debug("Knowledge fallback failed: %s", exc)

        candidates.sort(key=lambda r: r.get("_confidence", 0), reverse=True)
        return candidates

    @staticmethod
    def _compute_confidence(rule: dict[str, Any]) -> float:
        """Compute confidence score with time decay and trigger threshold."""
        success = int(rule.get("success_count", 0))
        failure = int(rule.get("failure_count", 0))
        total = success + failure
        last_triggered = rule.get("last_triggered", 0)

        if total == 0:
            return 0.5
        base = success / total

        if last_triggered:
            days_since = (time.time() - last_triggered) / 86400
            time_decay = max(0.5, math.exp(-0.1 * days_since))
        else:
            time_decay = 0.5

        trigger_penalty = min(1.0, total / 3.0)
        confidence = base * time_decay * trigger_penalty
        return round(min(1.0, max(0.0, confidence)), 2)

    def build_retry_plan(
        self,
        failure_type: str,
        rule: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a structured RetryPlan from a matched rule."""
        ctx = context or {}
        action_text = rule.get("action", "").lower()
        patch: dict[str, Any] = {}

        if "grip" in action_text or "grasp" in failure_type.lower():
            patch["gripper_force_offset"] = 0.15
            patch["approach_offset_z"] = -0.02
            patch["lateral_speed_factor"] = 0.8

        if "collision" in action_text or "clearance" in action_text:
            patch["safety_clearance"] = 0.05
            patch["velocity_factor"] = 0.5

        if "joint" in action_text or "limit" in action_text:
            patch["velocity_factor"] = 0.5
            patch["joint_range_reduction"] = 0.2

        if "timeout" in action_text or "backoff" in action_text or "connection" in action_text:
            patch["timeout_multiplier"] = 2.0
            patch["max_retries"] = 3

        if "compliant" in action_text or "force" in action_text:
            patch["control_mode"] = "compliant"
            patch["force_limit_factor"] = 0.7

        if "sensor" in action_text or "camera" in action_text:
            patch["sensor_fusion"] = True
            patch["camera_angle_offset"] = 15.0

        # Context-aware: robot-specific patches
        robot_id = ctx.get("robot_id", "")
        if robot_id and "ur5e" in str(robot_id).lower():
            patch["_robot_specific"] = "ur5e_safe_mode"

        if not patch:
            patch["velocity_factor"] = 0.8

        return {
            "action": "retry_with_adjustments",
            "parameter_patch": patch,
            "safety_override": [],
            "max_retries": ctx.get("max_retries", 3),
            "rule_id": rule.get("rule_id", ""),
        }

    def format_for_eventbus(
        self,
        recovery_hint: dict[str, Any],
        *,
        request_id: str = "",
    ) -> dict[str, Any]:
        """Format a RecoveryHint for EventBus publishing."""
        return {
            "request_id": request_id,
            "failure_type": recovery_hint.get("failure_type", ""),
            "hint": recovery_hint.get("hint", ""),
            "confidence": recovery_hint.get("confidence", 0.0),
            "source": recovery_hint.get("source", []),
            "retry_plan": recovery_hint.get("retry_plan", {}),
            "all_candidates": recovery_hint.get("all_candidates", []),
        }


class RecoveryFormatter:
    """Format recovery suggestions into EventBus-compatible payloads."""

    @staticmethod
    def to_event_payload(
        rule: dict[str, Any],
        *,
        request_id: str = "",
        source: str = "heuristic_engine",
    ) -> dict[str, Any]:
        """Convert a rule dict into an EventBus payload."""
        return {
            "request_id": request_id,
            "rule_id": rule.get("rule_id", ""),
            "condition": rule.get("condition", ""),
            "suggestion": rule.get("action", ""),
            "priority": rule.get("priority", 0),
            "source": rule.get("source", source),
            "success_count": rule.get("success_count", 0),
            "failure_count": rule.get("failure_count", 0),
        }

    @staticmethod
    def apply_trajectory_adjustment(
        trajectory: list[list[float]],
        suggestion: str,
    ) -> list[list[float]]:
        """Best-effort trajectory adjustment based on suggestion text."""
        suggestion_lower = suggestion.lower()

        if "reduce" in suggestion_lower and ("velocity" in suggestion_lower or "kp" in suggestion_lower):
            factor = 0.5 if "50" in suggestion_lower else 0.7
            return [[v * factor for v in wp] for wp in trajectory]

        if "grip" in suggestion_lower and "force" in suggestion_lower:
            offset = 0.2 if "20" in suggestion_lower else 0.1
            return [
                wp[:-1] + [wp[-1] + offset] if wp else wp
                for wp in trajectory
            ]

        return list(trajectory)


def format_recovery_suggestion(
    recovery: Optional[dict[str, Any]],
    *,
    request_id: str = "",
) -> str:
    """Human-readable recovery suggestion string."""
    if not recovery:
        return "No heuristic recovery available."
    action = recovery.get("action", "")
    source = recovery.get("source", "heuristic")
    return f"[{source}] {action}"
