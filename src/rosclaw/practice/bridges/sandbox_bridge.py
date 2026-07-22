"""Convert verified sandbox outcomes into durable Practice events."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.practice.writers.jsonl_writer import JsonlWriter


def _artifact_quality(receipt: dict[str, Any]) -> dict[str, Any]:
    expected = receipt.get("artifact_hashes") or {}
    references = receipt.get("artifacts") or []
    paths: dict[str, Path] = {}
    for reference in references:
        if not isinstance(reference, str):
            continue
        parsed = urlparse(reference)
        if parsed.scheme not in {"", "file"}:
            continue
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else reference)
        paths[path.name] = path.expanduser().resolve()
    hash_valid = bool(expected)
    for name, expected_hash in expected.items():
        path = paths.get(str(name))
        if path is None or not path.is_file():
            hash_valid = False
            break
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected_hash).removeprefix("sha256:"):
            hash_valid = False
            break
    required = all(
        bool(receipt.get(field)) for field in ("body_snapshot_hash", "model_hash", "action_hash")
    )
    return {
        "required_events_complete": required and bool(receipt.get("physics_executed")),
        "monotonic_timestamp": True,
        "missing_state_ratio": 0.0 if receipt.get("final_qpos") else 1.0,
        "duplicate_event_count": 0,
        "artifact_hash_valid": hash_valid,
        "trace_link_complete": bool(receipt.get("trace_id") or receipt.get("scenario_id")),
        "body_snapshot_match": bool(receipt.get("body_snapshot_hash")),
        "replayable": bool(receipt.get("request") and expected),
    }


class SandboxPracticeBridge(LifecycleMixin):
    """Persist low-frequency sandbox semantics and publish episode terminals."""

    _TOPICS = (
        "rosclaw.runtime.action.receipt",
        "firewall.action_allowed",
        "firewall.action_blocked",
    )

    def __init__(self, robot_id: str, event_bus: EventBus, output_dir: str) -> None:
        super().__init__()
        self._robot_id = robot_id
        self._bus = event_bus
        self._writer = JsonlWriter(Path(output_dir).expanduser() / "sandbox" / "events.jsonl")

    def _do_initialize(self) -> None:
        for topic in self._TOPICS:
            self._bus.subscribe(topic, self._on_event)

    def _do_stop(self) -> None:
        for topic in self._TOPICS:
            self._bus.unsubscribe(topic, self._on_event)
        self._writer.close()

    def _on_event(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        receipt = payload.get("simulation_receipt") or payload.get("simulation_result") or payload
        if not isinstance(receipt, dict):
            return
        if receipt.get("evidence_domain") != "SIMULATION":
            return
        episode_id = str(
            receipt.get("scenario_id")
            or receipt.get("action_id")
            or payload.get("request_id")
            or event.event_id
        )
        physics_executed = bool(receipt.get("physics_executed"))
        success = bool(receipt.get("is_safe", False)) and physics_executed
        quality = _artifact_quality(receipt)
        quality_pass = all(
            bool(quality[key])
            for key in (
                "required_events_complete",
                "artifact_hash_valid",
                "trace_link_complete",
                "body_snapshot_match",
                "replayable",
            )
        )
        record = {
            "schema_version": "practice.event.v1",
            "practice_id": f"sandbox:{episode_id}",
            "episode_id": episode_id,
            "robot_id": payload.get("robot_id", self._robot_id),
            "source": "sandbox",
            "event_type": (
                "sandbox.physics_rollout.completed"
                if physics_executed
                else "sandbox.physics_rollout.failed"
            ),
            "timestamp_ns": time.time_ns(),
            "trace_id": event.trace_id or receipt.get("trace_id"),
            "action_id": receipt.get("action_id"),
            "payload": {
                "success": success,
                "reason": receipt.get("reason"),
                "violations": receipt.get("violations", []),
                "metrics": receipt.get("metrics", {}),
                "artifacts": receipt.get("artifacts", []),
                "evidence_domain": "SIMULATION",
                "physics_executed": physics_executed,
            },
            "quality": quality,
            "tags": ["physics" if physics_executed else "no-physics"],
        }
        self._writer.write(record)
        terminal_payload = {
            "episode_id": episode_id,
            "request_id": payload.get("request_id") or receipt.get("action_id") or episode_id,
            "robot_id": record["robot_id"],
            "success": success,
            "outcome": "SUCCESS" if success else "FAILURE",
            "reward": 1.0 if success else -1.0,
            "reason": receipt.get("reason"),
            "physics_executed": physics_executed,
            "receipt_verified": quality["artifact_hash_valid"],
            "data_quality_pass": quality_pass,
            "evidence_domain": "SIMULATION",
            "data_quality": quality,
            "simulation_receipt": receipt,
        }
        terminal_topic = (
            "rosclaw.sandbox.episode.succeeded" if success else "rosclaw.sandbox.episode.failed"
        )
        self._bus.publish(
            Event(
                topic=terminal_topic,
                payload=terminal_payload,
                source="sandbox_practice_bridge",
                priority=EventPriority.HIGH if not success else EventPriority.NORMAL,
                trace_id=event.trace_id,
            )
        )
        self._bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.finished",
                payload=terminal_payload,
                source="sandbox_practice_bridge",
                trace_id=event.trace_id,
            )
        )


__all__ = ["SandboxPracticeBridge"]
