"""Convert verified sandbox outcomes into durable Practice events."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.practice.writers.jsonl_writer import JsonlWriter
from rosclaw.sandbox.evidence import (
    SimulationEvidenceVerification,
    verify_simulation_receipt,
)

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_STATE_SAMPLES = 25_001


def _bounded_text(value: Any, maximum: int = 1024) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        return ""
    return str(value)[:maximum]


def _bounded_list(value: Any, *, maximum: int = 50) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_bounded_text(item, 512) for item in value[:maximum]]


def _bounded_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:64]:
        key = _bounded_text(raw_key, 128)
        if (
            raw_value is None
            or isinstance(raw_value, bool)
            or isinstance(raw_value, (int, float))
            and math.isfinite(float(raw_value))
        ):
            result[key] = raw_value
        elif isinstance(raw_value, str):
            result[key] = raw_value[:512]
    return result


def _bounded_episode_id(value: Any) -> str:
    text = _bounded_text(value, 4096)
    if len(text) <= 256:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{text[:239]}-{digest}"


def _is_bounded_file(path: Path | None) -> bool:
    try:
        return bool(
            path is not None and path.is_file() and path.stat().st_size <= _MAX_ARTIFACT_BYTES
        )
    except OSError:
        return False


def _artifact_quality(receipt: dict[str, Any]) -> dict[str, Any]:
    raw_expected = receipt.get("artifact_hashes")
    expected = raw_expected if isinstance(raw_expected, dict) else {}
    raw_references = receipt.get("artifacts")
    references = raw_references if isinstance(raw_references, list) else []
    paths: dict[str, Path] = {}
    duplicate_paths: set[str] = set()
    contract_bounded = len(expected) <= 100 and len(references) <= 100
    for reference in references[:100]:
        if not isinstance(reference, str):
            continue
        parsed = urlparse(reference)
        if parsed.scheme not in {"", "file"}:
            continue
        if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
            continue
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else reference)
        resolved = path.expanduser().resolve()
        if resolved.name in paths and paths[resolved.name] != resolved:
            duplicate_paths.add(resolved.name)
        paths[resolved.name] = resolved
    hash_valid = bool(expected) and contract_bounded and not duplicate_paths
    for name, expected_hash in list(expected.items())[:100]:
        path = paths.get(str(name))
        if (
            Path(str(name)).name != str(name)
            or not _SHA256_RE.fullmatch(str(expected_hash))
            or not _is_bounded_file(path)
        ):
            hash_valid = False
            break
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            hash_valid = False
            break
        if actual != str(expected_hash).removeprefix("sha256:"):
            hash_valid = False
            break
    required = receipt.get("schema_version") == "rosclaw.simulation_receipt.v1" and all(
        bool(receipt.get(field)) for field in ("body_snapshot_hash", "model_hash", "action_hash")
    )
    monotonic = False
    missing_state_ratio = 1.0
    duplicate_event_count = 0
    states_path = paths.get("trajectory_states.json")
    if _is_bounded_file(states_path):
        try:
            states_payload = json.loads(states_path.read_text(encoding="utf-8"))
            states = states_payload.get("states") if isinstance(states_payload, dict) else None
            if isinstance(states, list) and 0 < len(states) <= _MAX_STATE_SAMPLES:
                times = [item.get("time") for item in states if isinstance(item, dict)]
                steps = [item.get("step") for item in states if isinstance(item, dict)]
                monotonic = (
                    len(times) == len(states)
                    and all(
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in times
                    )
                    and all(
                        float(left) < float(right)
                        for left, right in zip(times, times[1:], strict=False)
                    )
                )
                duplicate_event_count = len(steps) - len(set(steps))
                missing = sum(
                    1
                    for item in states
                    if not isinstance(item, dict)
                    or any(key not in item for key in ("qpos", "qvel", "command"))
                )
                missing_state_ratio = missing / len(states)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    raw_replay = receipt.get("replay_report")
    replay = raw_replay if isinstance(raw_replay, dict) else {}
    raw_embedded_quality = receipt.get("data_quality")
    embedded_quality = raw_embedded_quality if isinstance(raw_embedded_quality, dict) else {}
    replayable = bool(
        replay.get("verified") is True
        and replay.get("environment_match") is True
        and replay.get("hashes_verified") is True
        and replay.get("deterministic_label") is True
        and not replay.get("mismatches")
    )
    return {
        "required_events_complete": required and receipt.get("physics_executed") is True,
        "monotonic_timestamp": monotonic,
        "missing_state_ratio": missing_state_ratio,
        "duplicate_event_count": duplicate_event_count,
        "artifact_hash_valid": hash_valid,
        "trace_link_complete": bool(receipt.get("trace_id") or receipt.get("scenario_id")),
        "body_snapshot_match": embedded_quality.get("body_snapshot_match") is True,
        "replayable": replayable,
    }


class SandboxPracticeBridge(LifecycleMixin):
    """Persist low-frequency sandbox semantics and publish episode terminals."""

    _TOPICS = (
        "rosclaw.runtime.action.receipt",
        "firewall.action_allowed",
        "firewall.action_blocked",
    )

    def __init__(
        self,
        robot_id: str,
        event_bus: EventBus,
        output_dir: str,
        *,
        receipt_verifier: Callable[[dict[str, Any]], SimulationEvidenceVerification] | None = None,
    ) -> None:
        super().__init__()
        self._robot_id = robot_id
        self._bus = event_bus
        self._writer = JsonlWriter(Path(output_dir).expanduser() / "sandbox" / "events.jsonl")
        self._receipt_verifier = receipt_verifier or verify_simulation_receipt
        self._seen_receipts: set[tuple[str, str, str]] = set()
        self._seen_order: deque[tuple[str, str, str]] = deque()

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
        if (
            receipt.get("schema_version") != "rosclaw.simulation_receipt.v1"
            or receipt.get("evidence_domain") != "SIMULATION"
        ):
            return
        receipt_key = (
            _bounded_text(receipt.get("scenario_id"), 256),
            _bounded_text(receipt.get("action_hash"), 128),
            _bounded_text(receipt.get("seed"), 64),
        )
        if receipt_key in self._seen_receipts:
            return
        if len(self._seen_order) >= 1024:
            self._seen_receipts.discard(self._seen_order.popleft())
        self._seen_order.append(receipt_key)
        self._seen_receipts.add(receipt_key)
        episode_id = _bounded_episode_id(
            receipt.get("scenario_id")
            or receipt.get("action_id")
            or payload.get("request_id")
            or event.event_id
        )
        physics_executed = receipt.get("physics_executed") is True
        quality = _artifact_quality(receipt)
        try:
            independent_replay = self._receipt_verifier(receipt).verified
        except Exception:  # noqa: BLE001 - ingestion must fail closed
            independent_replay = False
        quality["independent_replay_verified"] = independent_replay
        quality_pass = (
            all(
                bool(quality[key])
                for key in (
                    "required_events_complete",
                    "artifact_hash_valid",
                    "trace_link_complete",
                    "body_snapshot_match",
                    "replayable",
                    "monotonic_timestamp",
                    "independent_replay_verified",
                )
            )
            and quality["missing_state_ratio"] == 0.0
            and quality["duplicate_event_count"] == 0
        )
        success = receipt.get("is_safe") is True and physics_executed and quality_pass
        record = {
            "schema_version": "practice.event.v1",
            "practice_id": f"sandbox:{episode_id}",
            "episode_id": episode_id,
            "robot_id": _bounded_text(payload.get("robot_id", self._robot_id), 256),
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
                "reason": _bounded_text(receipt.get("reason"), 1024),
                "violations": _bounded_list(receipt.get("violations")),
                "metrics": _bounded_metrics(receipt.get("metrics")),
                "artifacts": _bounded_list(receipt.get("artifacts"), maximum=20),
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
            "receipt_verified": quality_pass,
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
