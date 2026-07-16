"""Minimal rollout recorder that emits PracticeEventEnvelope JSONL traces.

This is intentionally lighter than the full PracticeRecorder/RuntimeBus path so
proposal-only and shadow rollouts can run from the CLI without starting the
runtime kernel.  Events use the canonical ``practice.event.v1`` envelope so
downstream tooling can replay them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rosclaw.practice.ids import generate_event_id
from rosclaw.practice.schemas import PracticeEventEnvelope
from rosclaw.practice.storage.layout import generate_practice_id


class RolloutRecorder:
    """Write a JSONL trace of practice events for a rollout."""

    def __init__(
        self,
        trace_path: str | Path,
        robot_id: str,
        *,
        practice_id: str | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
        body_id: str | None = None,
        policy_id: str | None = None,
        task_id: str | None = None,
    ):
        self.trace_path = Path(trace_path)
        self.practice_id = practice_id or generate_practice_id()
        self.session_id = session_id or generate_event_id()
        self.episode_id = episode_id or generate_event_id()
        self.robot_id = robot_id
        self.body_id = body_id
        self.policy_id = policy_id
        self.task_id = task_id
        self._sequence = 0

    def _emit(
        self,
        source: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        parent_event_id: str | None = None,
        action_id: str | None = None,
        frame_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        self._sequence += 1
        envelope = PracticeEventEnvelope(
            practice_id=self.practice_id,
            session_id=self.session_id,
            episode_id=self.episode_id,
            robot_id=self.robot_id,
            body_id=self.body_id,
            source=source,  # type: ignore[arg-type]
            event_type=event_type,
            trace_id=self.session_id,
            timestamp_ns=int(time.time_ns()),
            source_timestamp_ns=int(time.monotonic() * 1e9),
            sequence_id=self._sequence,
            parent_event_id=parent_event_id,
            frame_id=frame_id,
            task_id=self.task_id,
            action_id=action_id,
            policy_id=self.policy_id,
            payload=payload,
            tags=tags or [],
        )
        line = envelope.model_dump_json() + "\n"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(line)
        return envelope.event_id

    def record_runtime_started(self, runtime_info: dict[str, Any]) -> str:
        return self._emit("runtime", "rollout.runtime.started", runtime_info)

    def record_session_created(self, session_info: dict[str, Any], *, parent_event_id: str | None = None) -> str:
        return self._emit(
            "runtime",
            "rollout.session.created",
            session_info,
            parent_event_id=parent_event_id,
        )

    def record_observation_validated(
        self,
        snapshot: dict[str, Any],
        validation: dict[str, Any],
        *,
        parent_event_id: str | None = None,
        frame_id: str | None = None,
    ) -> str:
        return self._emit(
            "runtime",
            "rollout.observation.validated",
            {"snapshot": snapshot, "validation": validation},
            parent_event_id=parent_event_id,
            frame_id=frame_id,
        )

    def record_observation_failed(
        self,
        snapshot: dict[str, Any],
        validation: dict[str, Any],
        *,
        parent_event_id: str | None = None,
        frame_id: str | None = None,
    ) -> str:
        return self._emit(
            "runtime",
            "rollout.observation.failed",
            {"snapshot": snapshot, "validation": validation},
            parent_event_id=parent_event_id,
            frame_id=frame_id,
        )

    def record_inference(
        self,
        inference_result: dict[str, Any],
        latency_ms: float,
        *,
        parent_event_id: str | None = None,
        frame_id: str | None = None,
    ) -> str:
        return self._emit(
            "provider",
            "rollout.policy.inference",
            {"inference": inference_result, "latency_ms": round(latency_ms, 3)},
            parent_event_id=parent_event_id,
            frame_id=frame_id,
            action_id=inference_result.get("proposal_id"),
        )

    def record_action_mapping(
        self,
        proposal: dict[str, Any],
        mapping_report: dict[str, Any],
        mapped_action: dict[str, Any],
        latency_ms: float,
        *,
        parent_event_id: str | None = None,
        frame_id: str | None = None,
    ) -> str:
        return self._emit(
            "runtime",
            "rollout.action.mapped",
            {
                "proposal": proposal,
                "mapping": mapping_report,
                "mapped_action": mapped_action,
                "latency_ms": round(latency_ms, 3),
            },
            parent_event_id=parent_event_id,
            frame_id=frame_id,
            action_id=proposal.get("proposal_id"),
        )

    def record_sandbox_decision(
        self,
        mapped_action: dict[str, Any],
        decision: dict[str, Any],
        latency_ms: float,
        *,
        parent_event_id: str | None = None,
        frame_id: str | None = None,
    ) -> str:
        return self._emit(
            "sandbox",
            "rollout.sandbox.decision",
            {
                "mapped_action": mapped_action,
                "decision": decision,
                "latency_ms": round(latency_ms, 3),
            },
            parent_event_id=parent_event_id,
            frame_id=frame_id,
            action_id=mapped_action.get("action_id"),
        )

    def record_step(
        self,
        step_index: int,
        step_info: dict[str, Any],
        *,
        parent_event_id: str | None = None,
    ) -> str:
        return self._emit(
            "runtime",
            "rollout.step",
            {"step_index": step_index, **step_info},
            parent_event_id=parent_event_id,
            frame_id=str(step_index),
        )

    def record_failure(
        self,
        failure_type: str,
        description: str,
        severity: str = "high",
        *,
        related_event_ids: list[str] | None = None,
    ) -> str:
        return self._emit(
            "runtime",
            "failure_event",
            {
                "failure_type": failure_type,
                "severity": severity,
                "description": description,
                "related_event_ids": related_event_ids or [],
            },
        )

    def record_summary(self, result: dict[str, Any]) -> str:
        return self._emit(
            "runtime",
            "episode.summary",
            {
                "outcome": result.get("stop_reason", "unknown"),
                "success": result.get("stop_reason") == "completed",
                "event_count": self._sequence,
                "metrics": result.get("metrics", {}),
                "hardware_actions_executed": result.get("hardware_actions_executed", 0),
            },
        )

    def read_trace(self) -> list[dict[str, Any]]:
        """Read back the JSONL trace."""
        if not self.trace_path.exists():
            return []
        lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
