"""rosclaw_how.recovery — RecoveryEngine: generate RecoveryHint from failures.

The RecoveryEngine bridges failure events to actionable recovery hints.
It consumes FailureMemory / FirewallActionBlocked / CriticFailureReason /
SimilarEpisode inputs and produces RecoveryHint / RetryPlan outputs.

Design:
  - Stateless; all rule data lives in HeuristicEngine (SeekDB).
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
    """Generate RecoveryHint and RetryPlan from failure events.

    Args:
        heuristic_engine: HeuristicEngine instance for rule lookup.
    """

    def __init__(self, heuristic_engine: Any) -> None:
        self._how = heuristic_engine

    async def generate_recovery_hint(
        self,
        failure_type: str,
        context: Optional[dict[str, Any]] = None,
        sources: Optional[list[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Build a RecoveryHint dict from failure metadata.

        Returns:
            {
                "failure_type": str,
                "hint": str,
                "confidence": float,
                "source": list[str],
                "retry_plan": dict,
            } or None.
        """
        if not failure_type:
            return None

        ctx = context or {}
        srcs = sources or []

        # Look up heuristic rule for this failure type
        rule = None
        if self._how is not None:
            try:
                rule = await self._how.suggest_recovery(failure_type, context=ctx)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Heuristic lookup failed: %s", exc)

        if rule is None:
            return None

        # Build confidence from rule stats (with time decay & trigger threshold)
        confidence = self._compute_confidence(rule)

        # Build retry plan
        retry_plan = self.build_retry_plan(failure_type, rule, ctx)

        hint = {
            "failure_type": failure_type,
            "hint": rule.get("action", ""),
            "confidence": round(confidence, 2),
            "source": srcs + [f"heuristic:{rule.get('rule_id', 'unknown')}"],
            "retry_plan": retry_plan,
        }
        logger.info("RecoveryHint generated for %s (confidence=%.2f)", failure_type, confidence)
        return hint

    @staticmethod
    def _compute_confidence(rule: dict[str, Any]) -> float:
        """Compute confidence score with time decay and trigger threshold.

        Formula:
            confidence = base * time_decay * trigger_penalty

        - base: success_rate (success / total), 0.5 if no data
        - time_decay: e^(-0.1 * days_since_last_trigger), min 0.5
        - trigger_penalty: min(1.0, total_triggers / 3)
              (need at least 3 triggers for full confidence)
        """
        success = int(rule.get("success_count", 0))
        failure = int(rule.get("failure_count", 0))
        total = success + failure
        last_triggered = rule.get("last_triggered", 0)

        # Base confidence from success rate
        if total == 0:
            base = 0.5
        else:
            base = success / total

        # Time decay: confidence fades if rule hasn't been used recently
        if last_triggered:
            days_since = (time.time() - last_triggered) / 86400
            time_decay = max(0.5, math.exp(-0.1 * days_since))
        else:
            time_decay = 0.5  # Never triggered = lower confidence

        # Minimum trigger threshold (need 3+ triggers for full confidence)
        if total == 0:
            return 0.5  # Neutral confidence for untested rules
        trigger_penalty = min(1.0, total / 3.0)

        confidence = base * time_decay * trigger_penalty
        return round(min(1.0, max(0.0, confidence)), 2)

    def build_retry_plan(
        self,
        failure_type: str,
        rule: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a structured RetryPlan from a matched rule.

        Returns a dict with parameter patches for the next attempt:
        {
            "action": "retry_with_adjustments",
            "parameter_patch": {
                "gripper_force_offset": 0.15,
                "approach_offset_z": -0.02,
                "lateral_speed_factor": 0.8,
                ...
            },
            "safety_override": [],
            "max_retries": 3,
        }
        """
        ctx = context or {}
        action_text = rule.get("action", "").lower()
        patch: dict[str, Any] = {}

        # Grasp / gripper failures
        if "grip" in action_text or "grasp" in failure_type.lower():
            patch["gripper_force_offset"] = 0.15
            patch["approach_offset_z"] = -0.02
            patch["lateral_speed_factor"] = 0.8

        # Collision / path blocked
        if "collision" in action_text or "clearance" in action_text:
            patch["safety_clearance"] = 0.05
            patch["velocity_factor"] = 0.5

        # Joint limit / overload
        if "joint" in action_text or "limit" in action_text:
            patch["velocity_factor"] = 0.5
            patch["joint_range_reduction"] = 0.2

        # Timeout / communication / connection
        if "timeout" in action_text or "backoff" in action_text or "connection" in action_text:
            patch["timeout_multiplier"] = 2.0
            patch["max_retries"] = 3

        # Force / compliant mode
        if "compliant" in action_text or "force" in action_text:
            patch["control_mode"] = "compliant"
            patch["force_limit_factor"] = 0.7

        # Sensor / camera
        if "sensor" in action_text or "camera" in action_text:
            patch["sensor_fusion"] = True
            patch["camera_angle_offset"] = 15.0

        # Default: always include these
        if not patch:
            patch["velocity_factor"] = 0.8

        plan = {
            "action": "retry_with_adjustments",
            "parameter_patch": patch,
            "safety_override": [],
            "max_retries": ctx.get("max_retries", 3),
            "rule_id": rule.get("rule_id", ""),
        }
        return plan

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

        # Reduce velocity / joint limits
        if "reduce" in suggestion_lower and ("velocity" in suggestion_lower or "kp" in suggestion_lower):
            factor = 0.5 if "50" in suggestion_lower else 0.7
            return [[v * factor for v in wp] for wp in trajectory]

        # Increase grip force (add small offset to last DOF if gripper)
        if "grip" in suggestion_lower and "force" in suggestion_lower:
            offset = 0.2 if "20" in suggestion_lower else 0.1
            return [
                wp[:-1] + [wp[-1] + offset] if wp else wp
                for wp in trajectory
            ]

        # Default: no transformation
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
