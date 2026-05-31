"""BasicCritic — Event-driven success detection and reward computation.

Subscribes to execution events and publishes critic judgments:
  - rosclaw.critic.success.detected  (success)
  - rosclaw.critic.judgment          (all outcomes with reward)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.critic.basic_critic")


class BasicCritic(LifecycleMixin):
    """Simple rule-based critic for episode success detection.

    Reward rules:
      - success completion: +1.0
      - failure: -1.0
      - firewall/safety blocked: -0.5 (safe but incomplete)
      - timeout: -0.3
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__()
        self.robot_id = robot_id
        self.event_bus = event_bus
        self._judgments: list[dict[str, Any]] = []
        self._subscribed_topics: list[tuple[str, Any]] = []

    def _do_initialize(self) -> None:
        if self.event_bus is None:
            logger.info("Initialized in passive mode for %s", self.robot_id)
            return

        handlers = {
            "skill.execution.complete": self._on_skill_complete,
            "praxis.completed": self._on_praxis_completed,
            "praxis.failed": self._on_praxis_failed,
            "firewall.action_blocked": self._on_firewall_blocked,
            "safety.violation": self._on_safety_violation,
        }
        for topic, handler in handlers.items():
            self.event_bus.subscribe(topic, handler)
            self._subscribed_topics.append((topic, handler))

        logger.info("Initialized for %s", self.robot_id)

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            for topic, handler in self._subscribed_topics:
                try:
                    self.event_bus.unsubscribe(topic, handler)
                except Exception:
                    pass
        self._subscribed_topics.clear()

    def _on_skill_complete(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        result = payload.get("result", {})
        status = result.get("status", "unknown")
        episode_id = payload.get("episode_id", payload.get("correlation_id", "unknown"))

        if status == "success":
            self._judge(episode_id, "SUCCESS", 1.0, payload)
        elif status == "failure":
            error = result.get("error", "unknown")
            reward = -0.3 if "timeout" in error.lower() else -1.0
            self._judge(episode_id, "FAILED", reward, payload, reason=error)

    def _on_praxis_completed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        outcome = payload.get("outcome", {})
        episode_id = payload.get("practice_id", "unknown")
        reward = outcome.get("reward", 1.0)
        self._judge(episode_id, "SUCCESS", reward, payload)

    def _on_praxis_failed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        outcome = payload.get("outcome", {})
        episode_id = payload.get("practice_id", "unknown")
        reward = outcome.get("reward", -1.0)
        error = payload.get("error_log", "unknown")
        self._judge(episode_id, "FAILED", reward, payload, reason=error)

    def _on_firewall_blocked(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = payload.get("episode_id", payload.get("correlation_id", "unknown"))
        reason = payload.get("reason", "firewall blocked")
        self._judge(episode_id, "BLOCKED", -0.5, payload, reason=reason)

    def _on_safety_violation(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = payload.get("episode_id", payload.get("correlation_id", "unknown"))
        violations = payload.get("violations", [])
        reason = violations[0] if isinstance(violations, list) and violations else "safety violation"
        self._judge(episode_id, "BLOCKED", -0.5, payload, reason=reason)

    def _judge(
        self,
        episode_id: str,
        status: str,
        reward: float,
        context: dict[str, Any],
        reason: str = "",
    ) -> None:
        judgment = {
            "episode_id": episode_id,
            "robot_id": self.robot_id,
            "status": status,
            "reward": reward,
            "reason": reason,
            "timestamp": time.time(),
            "context": context,
        }
        self._judgments.append(judgment)

        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="rosclaw.critic.success.detected",
                payload={
                    "episode_id": episode_id,
                    "robot_id": self.robot_id,
                    "success": status == "SUCCESS",
                    "reward": reward,
                    "reason": reason,
                },
                source="basic_critic",
                priority=EventPriority.NORMAL,
            ))
            self.event_bus.publish(Event(
                topic="rosclaw.critic.judgment",
                payload=judgment,
                source="basic_critic",
                priority=EventPriority.NORMAL,
            ))

    def get_stats(self) -> dict[str, Any]:
        total = len(self._judgments)
        if total == 0:
            return {"total": 0, "success_rate": 0.0, "avg_reward": 0.0}
        successes = sum(1 for j in self._judgments if j["status"] == "SUCCESS")
        avg_reward = sum(j["reward"] for j in self._judgments) / total
        return {
            "total": total,
            "success_rate": successes / total,
            "avg_reward": round(avg_reward, 3),
            "by_status": {
                "SUCCESS": sum(1 for j in self._judgments if j["status"] == "SUCCESS"),
                "FAILED": sum(1 for j in self._judgments if j["status"] == "FAILED"),
                "BLOCKED": sum(1 for j in self._judgments if j["status"] == "BLOCKED"),
            },
        }
