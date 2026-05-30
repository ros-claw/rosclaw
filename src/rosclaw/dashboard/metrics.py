"""DashboardMetrics — Collect and aggregate runtime metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProviderMetric:
    """Single provider invocation metric."""
    provider: str
    capability: str
    latency_ms: float
    status: str  # ok, error, timeout
    timestamp: float = field(default_factory=time.time)


@dataclass
class SandboxMetric:
    """Sandbox validation result metric."""
    action_type: str
    is_safe: bool
    violations: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeMetric:
    """Practice episode summary metric."""
    episode_id: str
    robot_id: str
    status: str
    reward: Optional[float] = None
    duration_sec: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class DashboardMetrics:
    """In-memory metrics collector with rolling windows."""

    # Known topics whitelist — prevents unbounded dict growth from rogue/invalid topics.
    _KNOWN_TOPICS: set[str] = {
        "rosclaw.runtime.started",
        "skill.execution.start",
        "skill.execution.complete",
        "praxis.completed",
        "praxis.failed",
        "rosclaw.practice.event.created",
        "rosclaw.sandbox.episode.started",
        "rosclaw.sandbox.episode.finished",
        "rosclaw.sandbox.action.blocked",
        "rosclaw.provider.inference.completed",
        "rosclaw.critic.success.detected",
        "rosclaw.dashboard.trace.updated",
        "rosclaw.how.recovery_hint.generated",
        "rosclaw.memory.write.completed",
    }

    def __init__(self, max_history: int = 1000, max_trace_history: int = 100):
        self.max_history = max_history
        self.max_trace_history = max_trace_history
        self._provider_metrics: list[ProviderMetric] = []
        self._sandbox_metrics: list[SandboxMetric] = []
        self._episode_metrics: list[EpisodeMetric] = []
        self._traces: list[dict[str, Any]] = []
        self._event_counts: dict[str, int] = {}
        self._module_health: dict[str, str] = {}
        self._start_time = time.time()

    # ── Provider metrics ──

    def record_provider_call(self, provider: str, capability: str, latency_ms: float, status: str) -> None:
        self._provider_metrics.append(ProviderMetric(provider, capability, latency_ms, status))
        self._trim(self._provider_metrics)

    def get_provider_stats(self) -> dict[str, Any]:
        if not self._provider_metrics:
            return {"total": 0, "success_rate": 0.0, "avg_latency_ms": 0.0}

        total = len(self._provider_metrics)
        successes = sum(1 for m in self._provider_metrics if m.status == "ok")
        avg_latency = sum(m.latency_ms for m in self._provider_metrics) / total

        by_provider: dict[str, dict] = {}
        for m in self._provider_metrics:
            by_provider.setdefault(m.provider, {"calls": 0, "errors": 0})
            by_provider[m.provider]["calls"] += 1
            if m.status != "ok":
                by_provider[m.provider]["errors"] += 1

        return {
            "total": total,
            "success_rate": successes / total,
            "avg_latency_ms": round(avg_latency, 2),
            "by_provider": by_provider,
        }

    # ── Sandbox metrics ──

    def record_sandbox_validation(self, action_type: str, is_safe: bool, violations: Optional[list[str]] = None) -> None:
        self._sandbox_metrics.append(SandboxMetric(action_type, is_safe, violations or []))
        self._trim(self._sandbox_metrics)

    def get_sandbox_stats(self) -> dict[str, Any]:
        if not self._sandbox_metrics:
            return {"total": 0, "block_rate": 0.0}

        total = len(self._sandbox_metrics)
        blocks = sum(1 for m in self._sandbox_metrics if not m.is_safe)
        return {
            "total": total,
            "block_rate": blocks / total,
            "recent_violations": [
                {"action": m.action_type, "violations": m.violations}
                for m in self._sandbox_metrics[-10:] if not m.is_safe
            ],
        }

    # ── Episode metrics ──

    def record_episode(self, episode_id: str, robot_id: str, status: str, reward: Optional[float] = None, duration_sec: Optional[float] = None) -> None:
        self._episode_metrics.append(EpisodeMetric(episode_id, robot_id, status, reward, duration_sec))
        self._trim(self._episode_metrics)

    def get_episode_stats(self) -> dict[str, Any]:
        if not self._episode_metrics:
            return {"total": 0, "success_rate": 0.0, "avg_reward": 0.0}

        total = len(self._episode_metrics)
        successes = sum(1 for m in self._episode_metrics if m.status == "success")
        rewards = [m.reward for m in self._episode_metrics if m.reward is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        return {
            "total": total,
            "success_rate": successes / total,
            "avg_reward": round(avg_reward, 3),
            "recent": [
                {
                    "episode_id": m.episode_id,
                    "robot_id": m.robot_id,
                    "status": m.status,
                    "reward": m.reward,
                }
                for m in self._episode_metrics[-5:]
            ],
        }

    # ── Event counts ──

    def increment_event(self, topic: str) -> None:
        self._event_counts[topic] = self._event_counts.get(topic, 0) + 1
        # Prevent unbounded dict growth: cap total unique topics.
        if len(self._event_counts) > 200:
            # Drop oldest entries to make room.
            for _key in list(self._event_counts.keys())[:50]:
                del self._event_counts[_key]

    def get_event_counts(self) -> dict[str, int]:
        return dict(self._event_counts)

    # ── Module health ──

    def set_module_health(self, module: str, status: str) -> None:
        self._module_health[module] = status

    def get_module_health(self) -> dict[str, str]:
        return dict(self._module_health)

    # ── Uptime ──

    def get_uptime_sec(self) -> float:
        return time.time() - self._start_time

    # ── Snapshot ──

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_sec": round(self.get_uptime_sec(), 1),
            "module_health": self.get_module_health(),
            "provider": self.get_provider_stats(),
            "sandbox": self.get_sandbox_stats(),
            "episodes": self.get_episode_stats(),
            "event_counts": self.get_event_counts(),
            "traces": self.get_latest_traces(),
        }

    def record_trace(self, trace: dict[str, Any]) -> None:
        """Record a full Runtime→Provider→Sandbox→Practice→Memory→How trace."""
        self._traces.append(trace)
        self._trim(self._traces, maxsize=self.max_trace_history)

    def get_latest_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent traces."""
        return self._traces[-limit:]

    def _trim(self, lst: list, maxsize: int | None = None) -> None:
        """Trim list to maxsize, defaulting to self.max_history."""
        limit = maxsize if maxsize is not None else self.max_history
        if limit < 0:
            limit = 0
        if len(lst) > limit:
            del lst[: len(lst) - limit]
