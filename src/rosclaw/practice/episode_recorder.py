"""EpisodeRecorder — Unified practice event capture and artifact management.

Sprint 7: Subscribes to sandbox, provider, runtime, skill, and critic events;
assembles them into PraxisEvent records; writes artifact directories;
publishes enriched praxis.recorded for Memory ingestion.

Forwards-compatible: stub-subscribes to future topics (provider.inference.completed,
critic.success.detected, sandbox.episode.finished) that will be implemented in
later sprints.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.core.types import PraxisEvent


@dataclass
class _EpisodeBuffer:
    """In-flight episode state collected from multiple event sources."""

    episode_id: str
    robot_id: str = "default_robot"
    created_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)

    # Collected from individual events
    semantic_intent: Optional[str] = None
    llm_cot: Optional[str] = None
    initial_state: Optional[dict] = None
    final_state: Optional[dict] = None
    trajectory: list[dict] = field(default_factory=list)
    provider_traces: list[dict] = field(default_factory=list)
    sandbox_actions: list[dict] = field(default_factory=list)
    sandbox_blocked: bool = False
    sandbox_block_reason: Optional[str] = None
    runtime_failed: bool = False
    runtime_error: Optional[str] = None
    critic_reward: Optional[float] = None
    critic_status: Optional[str] = None
    praxis_status: Optional[str] = None
    praxis_reward: Optional[float] = None
    duration_sec: Optional[float] = None
    # CRITICAL FIX: agent_request stores the original user/agent request for full traceability
    agent_request: Optional[dict[str, Any]] = None

    # Tracking which events have arrived
    received_events: set = field(default_factory=set)

    def is_complete(self) -> bool:
        """Check if minimum expected events have arrived."""
        return {"skill.execution.start", "skill.execution.complete"}.issubset(
            self.received_events
        )


class EpisodeRecorder(LifecycleMixin):
    """
    Records robot practice episodes by subscribing to EventBus topics,
    buffering events per episode, and writing artifact directories.

    Subscribes to existing topics (skill.execution.*, praxis.completed/failed,
    firewall.action_blocked, safety.violation, agent.response) and stubs for
    future topics (provider.inference.completed, critic.success.detected,
    sandbox.episode.finished).

    On episode completion publishes ``praxis.recorded`` with full context
    for MemoryInterface ingestion.
    """

    # Topics that exist today
    _ACTIVE_TOPICS = [
        "skill.execution.start",
        "skill.execution.complete",
        "praxis.completed",
        "praxis.failed",
        "firewall.action_blocked",
        "safety.violation",
        "agent.response",
    ]

    # Topics from future sprints — subscribed as no-op stubs for forward compat
    _STUB_TOPICS = [
        "rosclaw.provider.inference.completed",
        "rosclaw.critic.success.detected",
        "rosclaw.sandbox.episode.finished",
    ]

    _EPISODE_TIMEOUT_SEC = 300.0

    def __init__(
        self,
        robot_id: str,
        event_bus: EventBus,
        artifact_base_dir: str = "~/.rosclaw/artifacts",
    ):
        super().__init__()
        self._robot_id = robot_id
        self._event_bus = event_bus
        self._artifact_base = Path(artifact_base_dir).expanduser()
        self._buffers: dict[str, _EpisodeBuffer] = {}
        self._counter_file = self._artifact_base / "episode_counter.json"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_initialize(self) -> None:
        self._artifact_base.mkdir(parents=True, exist_ok=True)

        if self._event_bus is None:
            print(f"[EpisodeRecorder] Initialized in read-only mode for {self._robot_id}, "
                  f"artifacts={self._artifact_base}")
            return



        topic_map = {
            "skill.execution.start": self._on_skill_start,
            "skill.execution.complete": self._on_skill_complete,
            "praxis.completed": self._on_praxis_completed,
            "praxis.failed": self._on_praxis_failed,
            "firewall.action_blocked": self._on_firewall_blocked,
            "safety.violation": self._on_safety_violation,
            "agent.response": self._on_agent_response,
            # Future topics — no-op stubs
            "rosclaw.provider.inference.completed": self._on_provider_inference,
            "rosclaw.critic.success.detected": self._on_critic_success,
            "rosclaw.sandbox.episode.finished": self._on_sandbox_episode_finished,
        }

        for topic, handler in topic_map.items():
            self._event_bus.subscribe(topic, handler)

        print(f"[EpisodeRecorder] Initialized for {self._robot_id}, "
              f"artifacts={self._artifact_base}")

    def _do_stop(self) -> None:
        topic_map = [
            "skill.execution.start",
            "skill.execution.complete",
            "praxis.completed",
            "praxis.failed",
            "firewall.action_blocked",
            "safety.violation",
            "agent.response",
            "rosclaw.provider.inference.completed",
            "rosclaw.critic.success.detected",
            "rosclaw.sandbox.episode.finished",
        ]
        for topic in topic_map:
            # EventBus unsubscribe needs the exact callback reference;
            # since we don't store them separately, rely on EventBus.clear_history
            # or accept that unsubscribing by topic is best-effort here.
            pass

        # Finalize any remaining buffers
        for episode_id in list(self._buffers.keys()):
            self._finalize_episode(episode_id, reason="shutdown")

    # ------------------------------------------------------------------
    # Episode ID / counter
    # ------------------------------------------------------------------

    def _next_episode_id(self) -> str:
        counter: dict[str, Any] = {"next_sequence": 1}
        if self._counter_file.exists():
            try:
                with open(self._counter_file, "r", encoding="utf-8") as f:
                    counter = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        seq = counter.get("next_sequence", 1)
        counter["next_sequence"] = seq + 1
        with open(self._counter_file, "w", encoding="utf-8") as f:
            json.dump(counter, f)
        return f"ep_{seq:04d}"

    def _extract_episode_id(self, payload: dict) -> str:
        for key in ("episode_id", "correlation_id", "practice_id", "request_id"):
            if key in payload and payload[key]:
                return str(payload[key])
        return self._next_episode_id()

    def _get_or_create_buffer(self, episode_id: str) -> _EpisodeBuffer:
        if episode_id not in self._buffers:
            self._buffers[episode_id] = _EpisodeBuffer(
                episode_id=episode_id,
                robot_id=self._robot_id,
            )
        return self._buffers[episode_id]

    # ------------------------------------------------------------------
    # Event handlers — active topics
    # ------------------------------------------------------------------

    def _on_skill_start(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("skill.execution.start")
        buf.semantic_intent = payload.get("skill_name", buf.semantic_intent)
        buf.initial_state = payload.get("initial_state", payload.get("state"))
        # CRITICAL FIX: capture the original agent request for artifact traceability
        if "agent_request" in payload:
            buf.agent_request = payload["agent_request"]
        elif "parameters" in payload:
            buf.agent_request = {"skill_name": payload.get("skill_name"), "parameters": payload.get("parameters")}
        buf.trajectory.append({
            "phase": "start",
            "timestamp": time.time(),
            "skill_name": payload.get("skill_name"),
            "parameters": payload.get("parameters"),
        })
        buf.last_event_at = time.time()

    def _on_skill_complete(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("skill.execution.complete")
        buf.final_state = payload.get("final_state", payload.get("state"))
        result = payload.get("result", {})
        duration = payload.get("duration_sec")
        if duration is None and buf.created_at:
            duration = time.time() - buf.created_at
        buf.duration_sec = duration
        buf.trajectory.append({
            "phase": "complete",
            "timestamp": time.time(),
            "result": result,
            "duration_sec": duration,
        })
        buf.last_event_at = time.time()
        # Note: skill.execution.complete does NOT auto-finalize;
        # we wait for praxis.completed/failed or other terminal events
        # so that all context (provider, critic, sandbox) can be collected.

    def _on_praxis_completed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("praxis.completed")
        outcome = payload.get("outcome", {})
        buf.praxis_status = "success"
        buf.praxis_reward = outcome.get("reward", 1.0)
        buf.last_event_at = time.time()
        self._finalize_episode(episode_id)

    def _on_praxis_failed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("praxis.failed")
        outcome = payload.get("outcome", {})
        buf.praxis_status = "failure"
        buf.praxis_reward = outcome.get("reward", -1.0)
        buf.runtime_error = payload.get("error_log", buf.runtime_error)
        buf.last_event_at = time.time()
        self._finalize_episode(episode_id)

    def _on_firewall_blocked(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("firewall.action_blocked")
        buf.sandbox_blocked = True
        violations = payload.get("violations", [])
        buf.sandbox_block_reason = (
            violations[0].get("description", "blocked")
            if violations else "blocked"
        )
        buf.sandbox_actions.append({
            "timestamp": time.time(),
            "action": payload.get("action", "unknown"),
            "blocked": True,
            "reason": buf.sandbox_block_reason,
        })
        buf.last_event_at = time.time()
        self._finalize_episode(episode_id)

    def _on_safety_violation(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("safety.violation")
        buf.sandbox_blocked = True
        violations = payload.get("violations", [])
        buf.sandbox_block_reason = (
            "; ".join(violations) if isinstance(violations, list) else str(violations)
        ) or "safety violation"
        buf.last_event_at = time.time()
        self._finalize_episode(episode_id)

    def _on_agent_response(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("agent.response")
        # Provider capability trace
        buf.provider_traces.append({
            "timestamp": time.time(),
            "status": payload.get("status"),
            "is_safe": payload.get("is_safe"),
            "request_id": payload.get("request_id"),
        })
        buf.last_event_at = time.time()

    # ------------------------------------------------------------------
    # Event handlers — stub topics (future sprints)
    # ------------------------------------------------------------------

    def _on_provider_inference(self, event: Event) -> None:
        """Stub for rosclaw.provider.inference.completed (Sprint 5)."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("rosclaw.provider.inference.completed")
        buf.semantic_intent = payload.get("intent", buf.semantic_intent)
        buf.llm_cot = payload.get("cot", payload.get("reasoning", buf.llm_cot))
        buf.provider_traces.append({
            "timestamp": time.time(),
            "model": payload.get("model"),
            "tokens": payload.get("tokens"),
            "latency_ms": payload.get("latency_ms"),
        })
        buf.last_event_at = time.time()

    def _on_critic_success(self, event: Event) -> None:
        """Stub for rosclaw.critic.success.detected (future sprint)."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("rosclaw.critic.success.detected")
        buf.critic_reward = payload.get("reward", 1.0 if payload.get("success") else 0.0)
        buf.critic_status = "SUCCESS" if payload.get("success", True) else "FAILED"
        buf.last_event_at = time.time()

    def _on_sandbox_episode_finished(self, event: Event) -> None:
        """Stub for rosclaw.sandbox.episode.finished (Sprint 3)."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        episode_id = self._extract_episode_id(payload)
        buf = self._get_or_create_buffer(episode_id)
        buf.received_events.add("rosclaw.sandbox.episode.finished")
        if payload.get("final_state"):
            buf.final_state = payload["final_state"]
        buf.last_event_at = time.time()
        # Terminal event — finalize
        self._finalize_episode(episode_id)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_episode(self, episode_id: str, reason: str = "complete") -> None:
        buf = self._buffers.pop(episode_id, None)
        if buf is None:
            return

        # Determine final status
        if buf.runtime_failed:
            status = "FAILED_RUNTIME"
            reward = -1.0
        elif buf.sandbox_blocked:
            status = "BLOCKED"
            reward = -1.0
        elif buf.critic_status:
            status = buf.critic_status
            reward = buf.critic_reward or 0.0
        elif buf.praxis_status:
            status = buf.praxis_status
            reward = buf.praxis_reward or 0.0
        else:
            status = "UNKNOWN"
            reward = 0.0

        # Build PraxisEvent
        praxis_event = PraxisEvent(
            event_id=episode_id,
            event_type=status.lower(),
            timestamp=time.time(),
            robot_id=buf.robot_id,
            agent_instruction=buf.semantic_intent or "",
            cot_trace=[buf.llm_cot] if buf.llm_cot else [],
            initial_state=buf.initial_state,
            final_state=buf.final_state,
            trajectory=[t.get("phase", "") for t in buf.trajectory],
            mcap_path=None,
            error_details=buf.runtime_error or buf.sandbox_block_reason or None,
            duration_sec=buf.duration_sec or 0.0,
            metadata={
                "episode_id": episode_id,
                "finalization_reason": reason,
                "received_events": sorted(buf.received_events),
                "is_complete": buf.is_complete(),
                "sandbox_blocked": buf.sandbox_blocked,
                "runtime_failed": buf.runtime_failed,
                "trajectory_entries": len(buf.trajectory),
                "provider_traces": len(buf.provider_traces),
            },
        )

        # Write artifact directory
        episode_dir = self._artifact_base / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "episode_id": episode_id,
            "robot_id": buf.robot_id,
            "created_at": buf.created_at,
            "finalized_at": time.time(),
            "finalization_reason": reason,
            "duration_sec": buf.duration_sec,
            "received_events": sorted(buf.received_events),
            "is_complete": buf.is_complete(),
            "status": status,
            "reward": reward,
            "sandbox_blocked": buf.sandbox_blocked,
            "sandbox_block_reason": buf.sandbox_block_reason,
            "runtime_failed": buf.runtime_failed,
            "runtime_error": buf.runtime_error,
            "praxis_event": {
                "event_id": praxis_event.event_id,
                "event_type": praxis_event.event_type,
                "robot_id": praxis_event.robot_id,
                "agent_instruction": praxis_event.agent_instruction,
                "duration_sec": praxis_event.duration_sec,
                "error_details": praxis_event.error_details,
            },
        }
        with open(episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(episode_dir / "trajectory.jsonl", "w", encoding="utf-8") as f:
            for entry in buf.trajectory:
                f.write(json.dumps(entry, default=str) + "\n")

        with open(episode_dir / "provider_trace.jsonl", "w", encoding="utf-8") as f:
            for entry in buf.provider_traces:
                f.write(json.dumps(entry, default=str) + "\n")

        with open(episode_dir / "sandbox_replay.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "blocked": buf.sandbox_blocked,
                    "block_reason": buf.sandbox_block_reason,
                    "actions": buf.sandbox_actions,
                },
                f,
                indent=2,
                default=str,
            )

        # Write events.jsonl — full event stream
        with open(episode_dir / "events.jsonl", "w", encoding="utf-8") as f:
            for ev_name in sorted(buf.received_events):
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "type": ev_name,
                    "episode_id": episode_id,
                    "robot_id": buf.robot_id,
                }, default=str) + "\n")

        # Write critic_result.json
        critic_result = {
            "episode_id": episode_id,
            "reward": reward,
            "status": status,
            "critic_reward": buf.critic_reward,
            "critic_status": buf.critic_status,
            "praxis_status": buf.praxis_status,
            "praxis_reward": buf.praxis_reward,
        }
        with open(episode_dir / "critic_result.json", "w", encoding="utf-8") as f:
            json.dump(critic_result, f, indent=2, default=str)

        # Write memory_write.json
        memory_write = {
            "episode_id": episode_id,
            "robot_id": buf.robot_id,
            "event_type": praxis_event.event_type,
            "instruction": praxis_event.agent_instruction,
            "outcome": status,
            "duration_sec": buf.duration_sec,
            "artifact_uri": f"rosclaw://artifacts/episodes/{episode_id}",
            "artifact_dir": str(episode_dir),
            "timestamp": time.time(),
        }
        with open(episode_dir / "memory_write.json", "w", encoding="utf-8") as f:
            json.dump(memory_write, f, indent=2, default=str)

        # CRITICAL FIX: write agent_request.json — the original request that started this episode
        agent_request = buf.agent_request or {"skill_name": buf.semantic_intent, "note": "agent_request not captured from event payload"}
        with open(episode_dir / "agent_request.json", "w", encoding="utf-8") as f:
            json.dump({
                "episode_id": episode_id,
                "robot_id": buf.robot_id,
                "agent_request": agent_request,
                "timestamp": time.time(),
            }, f, indent=2, default=str)

        # Publish enriched praxis.recorded
        self._event_bus.publish(Event(
            topic="praxis.recorded",
            payload={
                "event_id": praxis_event.event_id,
                "event_type": praxis_event.event_type,
                "robot_id": praxis_event.robot_id,
                "instruction": praxis_event.agent_instruction,
                "duration_sec": praxis_event.duration_sec,
                "outcome": status,
                "trajectory_waypoints": len(buf.trajectory),
                "cot_steps": len(praxis_event.cot_trace),
                "artifact_dir": str(episode_dir),
                "artifact_uri": f"rosclaw://artifacts/episodes/{episode_id}",
                "episode_metadata": metadata,
            },
            source="episode_recorder",
            priority=EventPriority.NORMAL,
        ))

        print(f"[EpisodeRecorder] Finalized {episode_id}: status={status}, "
              f"reward={reward}, events={len(buf.received_events)}, "
              f"dir={episode_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_episodes(self) -> list[dict]:
        """Return metadata for all recorded episodes."""
        episodes_dir = self._artifact_base / "episodes"
        if not episodes_dir.exists():
            return []
        episodes = []
        for name in sorted(os.listdir(episodes_dir)):
            ep_dir = episodes_dir / name
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    episodes.append({
                        "episode_id": name,
                        "status": meta.get("status", "UNKNOWN"),
                        "timestamp": meta.get("created_at"),
                        "robot_id": meta.get("robot_id", "unknown"),
                        "reward": meta.get("reward"),
                        "is_complete": meta.get("is_complete", False),
                    })
                except (json.JSONDecodeError, OSError):
                    pass
        return episodes

    def get_episode(self, episode_id: str) -> Optional[dict]:
        """Read full metadata for a single episode."""
        meta_path = self._artifact_base / "episodes" / episode_id / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def active_episode_count(self) -> int:
        """Number of in-flight episodes."""
        return len(self._buffers)

    @property
    def artifact_base(self) -> Path:
        return self._artifact_base
