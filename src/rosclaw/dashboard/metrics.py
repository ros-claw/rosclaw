"""DashboardMetrics — Collect and aggregate runtime metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.core.event_topics import EventTopics


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
    reward: float | None = None
    duration_sec: float | None = None
    timestamp: float = field(default_factory=time.time)


class DashboardMetrics:
    """In-memory metrics collector with rolling windows."""

    # Known topics whitelist — prevents unbounded dict growth from rogue/invalid topics.
    _KNOWN_TOPICS: set[str] = {
        EventTopics.RUNTIME_STARTED,
        EventTopics.SKILL_EXECUTION_START,
        EventTopics.SKILL_EXECUTION_COMPLETE,
        EventTopics.PRAXIS_COMPLETED,
        EventTopics.PRAXIS_FAILED,
        EventTopics.PRACTICE_EVENT_CREATED,
        EventTopics.SANDBOX_EPISODE_STARTED,
        EventTopics.SANDBOX_EPISODE_FINISHED,
        EventTopics.SANDBOX_ACTION_BLOCKED,
        EventTopics.PROVIDER_INFERENCE_COMPLETED,
        EventTopics.CRITIC_SUCCESS_DETECTED,
        EventTopics.DASHBOARD_TRACE_UPDATED,
        EventTopics.HOW_RECOVERY_HINT_GENERATED,
        EventTopics.MEMORY_WRITE_COMPLETED,
        EventTopics.SENSE_STATE_UPDATED,
        EventTopics.SENSE_BODY_UPDATED,
        EventTopics.SENSE_EVENT_DETECTED,
        EventTopics.SENSE_READINESS_UPDATED,
        EventTopics.SENSE_CAPABILITY_BLOCKED,
        EventTopics.SENSE_CAPABILITY_DEGRADED,
        # Legacy aliases and topics not yet in EventTopics
        "rosclaw.auto.proposal.created",
        "rosclaw.auto.champion.promoted",
        "rosclaw.auto.experiment.completed",
        "rosclaw.auto.deadend.registered",
        "rosclaw.how.evidence.generated",
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
        self._auto_proposals: list[dict[str, Any]] = []
        self._auto_experiments: list[dict[str, Any]] = []
        self._auto_champions: list[dict[str, Any]] = []
        self._auto_deadends: list[dict[str, Any]] = []
        self._evidence_traces: list[dict[str, Any]] = []
        self._body_sense_history: list[dict[str, Any]] = []
        self._latest_sense: dict[str, Any] | None = None
        self._realsense_state: dict[str, dict[str, Any]] = {
            "d405": {"online": False, "last_frame_at": None, "frame_count": 0, "fps": 0.0},
            "d435i": {"online": False, "last_frame_at": None, "frame_count": 0, "fps": 0.0},
        }
        self._realsense_frame_times: dict[str, list[float]] = {"d405": [], "d435i": []}

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

    def record_sandbox_validation(self, action_type: str, is_safe: bool, violations: list[str] | None = None) -> None:
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

    def record_episode(self, episode_id: str, robot_id: str, status: str, reward: float | None = None, duration_sec: float | None = None) -> None:
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

    def increment_event(self, topic: str, payload: dict[str, Any] | None = None) -> None:
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

    # ── Body sense metrics ──

    def record_body_sense(self, body_sense: dict[str, Any]) -> None:
        """Record a BodySense snapshot from the sense runtime."""
        if not isinstance(body_sense, dict):
            return
        self._latest_sense = body_sense
        self._body_sense_history.append(body_sense)
        self._trim(self._body_sense_history)

    def get_body_sense_stats(self) -> dict[str, Any]:
        """Return the latest sense snapshot plus rolling-window stats."""
        if self._latest_sense is None:
            return {
                "available": False,
                "robot_id": None,
                "overall_status": "unknown",
                "blocked_capabilities": [],
                "degraded_capabilities": [],
                "risk_summary": None,
                "main_reasons": [],
                "recommended_actions": [],
                "history_count": len(self._body_sense_history),
            }
        return {
            "available": True,
            "robot_id": self._latest_sense.get("robot_id"),
            "overall_status": self._latest_sense.get("overall_status"),
            "blocked_capabilities": list(self._latest_sense.get("blocked_capabilities", [])),
            "degraded_capabilities": list(self._latest_sense.get("degraded_capabilities", [])),
            "risk_summary": self._latest_sense.get("risk_summary"),
            "main_reasons": list(self._latest_sense.get("main_reasons", [])),
            "recommended_actions": list(self._latest_sense.get("recommended_actions", [])),
            "history_count": len(self._body_sense_history),
            "timestamp": self._latest_sense.get("timestamp"),
        }

    # ── RealSense stream metrics ──

    def set_realsense_online(self, camera_key: str, online: bool, info: dict[str, Any] | None = None) -> None:
        """Update online status and metadata for a RealSense camera."""
        camera_key = camera_key.lower()
        if camera_key not in self._realsense_state:
            return
        self._realsense_state[camera_key]["online"] = online
        if info:
            self._realsense_state[camera_key].update(info)

    def record_realsense_frame(
        self,
        camera_key: str,
        frame_type: str,
        path: str,
        latency_ms: float | None = None,
        drop_count: int | None = None,
    ) -> None:
        """Record arrival of a RealSense frame and compute rolling FPS."""
        camera_key = camera_key.lower()
        if camera_key not in self._realsense_state:
            return
        now = time.time()
        state = self._realsense_state[camera_key]
        state["last_frame_at"] = now
        state["last_frame_type"] = frame_type
        state["last_frame_path"] = path
        state["frame_count"] = state.get("frame_count", 0) + 1
        if latency_ms is not None:
            state["last_latency_ms"] = latency_ms
        if drop_count is not None:
            state["drop_count"] = drop_count

        times = self._realsense_frame_times.setdefault(camera_key, [])
        times.append(now)
        # Keep a 2-second window for FPS estimation.
        cutoff = now - 2.0
        while times and times[0] < cutoff:
            times.pop(0)
        if len(times) > 1:
            elapsed = now - times[0]
            state["fps"] = round((len(times) - 1) / elapsed, 2) if elapsed > 0 else 0.0
        else:
            state["fps"] = 0.0

    def get_realsense_state(self) -> dict[str, Any]:
        """Return current RealSense stream state for dashboard."""
        return {
            "cameras": dict(self._realsense_state),
            "dual_online": (
                self._realsense_state.get("d405", {}).get("online", False)
                and self._realsense_state.get("d435i", {}).get("online", False)
            ),
        }

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
            "sense": self.get_body_sense_stats(),
            "realsense": self.get_realsense_state(),
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
