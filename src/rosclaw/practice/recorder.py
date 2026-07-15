"""PracticeRecorder — runtime consumer + legacy DataFlywheel recorder.

In Runtime Kernel v2 the recorder is a first-class RuntimeBus consumer: it
subscribes to runtime events, builds a practice session on ``practice.start``,
writes events to the local filesystem layout, and finalizes the episode on
``practice.stop``.

The legacy EventBus/DataFlywheel API is preserved for existing callers and
tests. When constructed with a robot id string and an optional EventBus, the
recorder creates an internal RuntimeBus wrapper and continues to publish
``praxis.completed`` / ``praxis.failed`` / ``heuristic.recovery_executed`` on
that EventBus.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.data.flywheel import DataFlywheel, EventType
from rosclaw.practice.artifact_store import ArtifactStore
from rosclaw.practice.config import (
    DEFAULT_DATA_ROOT,
    PracticeSession,
    PracticeSummary,
    RecorderConfig,
)
from rosclaw.practice.ids import generate_episode_id
from rosclaw.practice.schemas import SCHEMA_VERSION, EpisodeSummaryPayload, PracticeEventEnvelope
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout, generate_practice_id
from rosclaw.practice.writers.jsonl_writer import JsonlWriter
from rosclaw.practice.writers.mcap_writer import McapWriter
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.component import RuntimeConsumer
from rosclaw.runtime.event import RuntimeEvent

logger = logging.getLogger("rosclaw.practice.recorder")

_ALLOWED_SOURCES = {
    "dds",
    "ros2",
    "camera",
    "agent",
    "provider",
    "sandbox",
    "runtime",
    "human",
    "system",
}


def _normalize_outcome(outcome: str | None) -> str:
    """Return a canonical outcome string for storage and schema validation.

    Unknown or empty values are preserved as ``unknown``.  ``FAILED`` is
    mapped to ``failure`` because :class:`EpisodeSummaryPayload` expects the
    latter literal.
    """
    if not outcome:
        return "unknown"
    upper = outcome.upper()
    if upper == "SUCCESS":
        return "success"
    if upper == "FAILED":
        return "failure"
    if upper == "PARTIAL":
        return "partial"
    if upper == "UNKNOWN":
        return "unknown"
    return outcome.lower()


class PracticeRecorder(RuntimeConsumer):
    """Records runtime events and preserves the legacy EventBus/DataFlywheel API.

    Two construction styles are supported:

    1. Runtime Kernel v2::

           bus = RuntimeBus()
           recorder = PracticeRecorder(bus, data_root="/data/rosclaw/practice")
           recorder.initialize(); recorder.start()

    2. Legacy style (preserved for compatibility)::

           recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
           recorder.initialize(); recorder.start_recording()
    """

    def __init__(
        self,
        runtime_bus_or_robot_id: RuntimeBus | str,
        joint_dof: int = 6,
        event_bus: EventBus | None = None,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        publish_to_event_bus: bool = True,
        auto_start_on_skill: bool = True,
        config: RecorderConfig | None = None,
    ) -> None:
        # Resolve the RuntimeBus and robot id from the first argument.
        if isinstance(runtime_bus_or_robot_id, RuntimeBus):
            runtime_bus = runtime_bus_or_robot_id
            self.robot_id = getattr(runtime_bus, "robot_id", None) or "default_robot"
        else:
            self.robot_id = runtime_bus_or_robot_id
            runtime_bus = RuntimeBus(event_bus=event_bus or EventBus())

        super().__init__("practice_recorder", runtime_bus)
        self.joint_dof = joint_dof
        self._legacy_event_bus = event_bus
        self._publish_to_event_bus = publish_to_event_bus
        self._auto_start_on_skill = auto_start_on_skill
        self._config = config or RecorderConfig()

        # Runtime Kernel v2 recording state.
        self.layout = PracticeLayout(data_root)
        self._catalog: PracticeCatalog | None = None
        self._artifact_store: ArtifactStore | None = None
        self._writer: JsonlWriter | None = None
        self._mcap_writer: McapWriter | None = None
        self._mcap_path: Path | None = None
        self._session: PracticeSession | None = None
        self._summary: PracticeSummary | None = None
        self._event_count = 0
        self._source_event_count = 0
        self._failure_labels: list[str] = []
        self._lock = threading.RLock()

        # Legacy state.
        self._flywheel: DataFlywheel | None = None
        self._recording = False
        self._failure_context: dict[str, Any] = {
            "previous_scores": [],
            "current_iteration": 0,
            "last_error": "",
        }
        self._knowledge_ingest_log: list[dict] = []
        self._legacy_subscriptions: list[tuple[str, Any]] = []

    @property
    def config(self) -> RecorderConfig:
        """Return the recorder's active configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_initialize(self) -> None:
        """Initialize the file layout and legacy DataFlywheel."""
        self.layout.ensure_directories()
        try:
            self._flywheel = DataFlywheel(
                robot_id=self.robot_id,
                joint_dof=self.joint_dof,
            )
        except Exception as exc:
            logger.warning("Failed to initialize DataFlywheel: %s", exc)
            self._flywheel = None

        # Legacy subscriptions on the raw EventBus.
        if self._legacy_event_bus is not None:
            self._legacy_event_bus.subscribe("skill.execution.complete", self._on_skill_complete)
            self._legacy_event_bus.subscribe(
                "knowledge.ingest_complete", self._on_knowledge_ingest_complete
            )
            self._legacy_subscriptions = [
                ("skill.execution.complete", self._on_skill_complete),
                ("knowledge.ingest_complete", self._on_knowledge_ingest_complete),
            ]

        logger.info("Recorder initialized for %s (flywheel mode)", self.robot_id)

    def _do_start(self) -> None:
        """Start runtime subscriptions."""
        # Runtime Kernel v2: subscribe to all runtime events.
        super()._do_start()

    def _do_stop(self) -> None:
        """Stop legacy and runtime subscriptions, finalize any open session."""
        # Unsubscribe legacy callbacks first.
        if self._legacy_event_bus is not None:
            for topic, callback in self._legacy_subscriptions:
                try:
                    self._legacy_event_bus.unsubscribe(topic, callback)
                except Exception as exc:
                    logger.debug("Legacy unsubscribe failed: %s", exc)
            self._legacy_subscriptions.clear()

        # Finalize an open runtime session if the stop event was missed.
        if self._session is not None:
            self._finalize_runtime_session("UNKNOWN")

        super()._do_stop()

    # ------------------------------------------------------------------
    # Runtime Kernel v2 consumer API
    # ------------------------------------------------------------------

    def on_event(self, event: RuntimeEvent) -> None:
        """Handle every runtime event.

        ``RuntimeBus`` canonicalizes legacy topic aliases (e.g.
        ``skill.execution.start`` -> ``skill.invoke``), so the recorder listens
        for both forms.
        """
        canonical = event.type
        if canonical == "practice.start":
            self._on_practice_start(event)
        elif canonical == "practice.stop":
            self._on_practice_stop(event)
        elif canonical in ("skill.execution.start", "skill.invoke"):
            if self._session is None and self._auto_start_on_skill:
                self._auto_start_session_from_skill(event)
            if self._session is not None:
                self._record_event(event)
        elif self._session is not None:
            self._record_event(event)

    def _auto_start_session_from_skill(self, event: RuntimeEvent) -> None:
        """Create a default practice session when a skill starts without one."""
        payload = event.payload or {}
        skill_id = (
            payload.get("skill_name")
            or payload.get("skill_id")
            or event.metadata.get("skill_id")
            or "unknown"
        )
        robot_id = event.robot or event.metadata.get("robot_id") or self.robot_id
        practice_id = generate_practice_id()
        logger.info(
            "Auto-starting practice session %s for skill %s (robot %s)",
            practice_id,
            skill_id,
            robot_id,
        )
        self._on_practice_start(
            RuntimeEvent(
                id=practice_id,
                timestamp=event.timestamp,
                source="practice_recorder",
                robot=robot_id,
                body_id=event.body_id,
                type="practice.start",
                payload={
                    "practice_id": practice_id,
                    "robot_id": robot_id,
                    "skill_id": skill_id,
                    "sources": {"runtime": True, "skill": True},
                    "auto_started": True,
                },
                metadata={"trigger_event_id": event.id},
            )
        )

    def _on_practice_start(self, event: RuntimeEvent) -> None:
        if self._session is not None:
            logger.warning(
                "PracticeRecorder received practice.start while already recording %s; ignoring",
                self._session.practice_id,
            )
            return

        payload = event.payload or {}
        practice_id = payload.get("practice_id") or generate_practice_id()
        robot_id = payload.get("robot_id") or event.robot or self.robot_id
        robot_type = payload.get("robot_type") or event.metadata.get("robot_type")
        task_id = payload.get("task_id") or event.metadata.get("task_id")
        task_name = payload.get("task_name") or event.metadata.get("task_name")
        skill_id = payload.get("skill_id") or event.metadata.get("skill_id")
        session_id = payload.get("session_id") or event.id
        episode_id = payload.get("episode_id") or generate_episode_id()

        session_dir = (
            Path(payload["session_dir"])
            if payload.get("session_dir")
            else self.layout.create_session_dirs(practice_id)
        )
        start_time_utc = _format_utc(event.timestamp)
        start_time_ns = _datetime_to_ns(event.timestamp)

        self._session = PracticeSession(
            practice_id=practice_id,
            robot_id=robot_id,
            task_id=task_id,
            task_name=task_name,
            skill_id=skill_id,
            session_dir=session_dir,
            start_time_ns=start_time_ns,
            start_time_utc=start_time_utc,
            robot_type=robot_type,
            session_id=session_id,
            episode_id=episode_id,
            tags=list(event.metadata.get("tags", [])),
            metadata=dict(event.metadata.get("session_metadata", {})),
        )
        if event.body_id:
            self._session.metadata["body_id"] = event.body_id
        self._session.metadata["sources"] = payload.get("sources", {})
        self._session.metadata["seekdb_enabled"] = payload.get("seekdb_enabled", False)

        self._event_count = 0
        self._source_event_count = 0
        self._failure_labels = []
        self._summary = None

        self._writer = JsonlWriter(self.layout.events_jsonl_path(practice_id), rotate_mb=None)
        self._catalog = PracticeCatalog(self.layout.catalog_db_path)
        self._artifact_store = ArtifactStore(self.layout.data_root, layout=self.layout)

        if self._config.mcap_enabled and McapWriter.is_available():
            try:
                self._mcap_path = self.layout.mcap_path(practice_id)
                self._mcap_writer = McapWriter(
                    self._mcap_path,
                    compression=self._config.mcap_compression,
                    chunk_size_bytes=self._config.mcap_chunk_size_bytes,
                )
                logger.info("MCAP recording enabled: %s", self._mcap_path)
            except Exception as e:
                logger.warning("Failed to enable MCAP recording: %s", e)
                self._mcap_writer = None
                self._mcap_path = None

        self._catalog.insert_practice(
            {
                "practice_id": practice_id,
                "session_id": session_id,
                "episode_id": episode_id,
                "robot_id": robot_id,
                "robot_type": robot_type,
                "task_id": task_id,
                "task_name": task_name,
                "skill_id": skill_id,
                "start_time": start_time_utc,
                "manifest_path": str(self.layout.manifest_path(practice_id)),
                "events_jsonl_path": str(self.layout.events_jsonl_path(practice_id)),
                "outcome": "running",
            }
        )

        self.layout.write_manifest(
            self._session,
            sources=payload.get("sources", {}),
            seekdb_enabled=payload.get("seekdb_enabled", False),
        )

        logger.info("PracticeRecorder started session: %s", practice_id)

    def _on_practice_stop(self, event: RuntimeEvent) -> None:
        if self._session is None:
            logger.warning("PracticeRecorder received practice.stop with no active session")
            return

        payload = event.payload or {}
        duration_ms = payload.get("duration_ms")
        if duration_ms is None:
            duration_ms = (
                _datetime_to_ns(event.timestamp) - self._session.start_time_ns
            ) / 1_000_000.0

        outcome = payload.get("outcome", "UNKNOWN")
        reward = payload.get("reward")
        failure_labels = payload.get("failure_labels", [])
        event_count = payload.get("event_count", self._event_count)

        self._summary = PracticeSummary(
            practice_id=self._session.practice_id,
            robot_id=self._session.robot_id,
            outcome=outcome,
            reward=reward,
            duration_ms=duration_ms,
            event_count=event_count,
            artifact_dir=self._session.session_dir,
            mcap_path=self._mcap_path,
            failure_labels=failure_labels,
        )

        self._finalize_runtime_session(
            outcome, reward=reward, duration_ms=duration_ms, event_count=event_count
        )

    def _finalize_runtime_session(
        self,
        outcome: str,
        reward: float | None = None,
        duration_ms: float | None = None,
        event_count: int | None = None,
        failure_labels: list[str] | None = None,
    ) -> None:
        if self._session is None:
            return

        if duration_ms is None:
            duration_ms = (time.monotonic_ns() - self._session.start_time_ns) / 1_000_000.0
        if event_count is None:
            event_count = self._event_count

        self._summary = PracticeSummary(
            practice_id=self._session.practice_id,
            robot_id=self._session.robot_id,
            outcome=outcome,
            reward=reward,
            duration_ms=duration_ms,
            event_count=event_count,
            artifact_dir=self._session.session_dir,
            mcap_path=self._mcap_path,
            failure_labels=failure_labels or list(self._failure_labels),
        )

        if self._catalog is not None:
            self._catalog.update_practice(
                self._session.practice_id,
                {
                    "end_time": _utc_now_iso(),
                    "duration_ms": duration_ms,
                    "outcome": outcome,
                    "reward": reward,
                },
            )

        # Write v2 episode summary artifacts and catalog records.
        self._write_v2_episode_summary(outcome, reward, duration_ms, event_count, failure_labels)

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._mcap_writer is not None:
            try:
                self._mcap_writer.close()
            except Exception as e:
                logger.error("Failed to close MCAP writer: %s", e)
                self._mcap_path = None
            finally:
                self._mcap_writer = None

        sources = self._session.metadata.get("sources", {}) if self._session.metadata else {}
        seekdb_enabled = (
            self._session.metadata.get("seekdb_enabled", False) if self._session.metadata else False
        )

        self.layout.write_manifest(
            self._session, summary=self._summary, sources=sources, seekdb_enabled=seekdb_enabled
        )
        self.layout.finalize_session(
            self._session.practice_id,
            self._session,
            self._summary,
            sources=sources,
        )

        logger.info("PracticeRecorder stopped session: %s (%s)", self._session.practice_id, outcome)

        self._session = None
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None

    def _write_v2_episode_summary(
        self,
        outcome: str,
        reward: float | None,
        duration_ms: float | None,
        event_count: int,
        failure_labels: list[str] | None,
    ) -> None:
        """Write episode summary to ArtifactStore and catalog v2 tables."""
        if self._session is None or self._artifact_store is None or self._catalog is None:
            return

        session_id = self._session.session_id or self._session.practice_id
        episode_id = self._session.episode_id or session_id
        if not episode_id:
            logger.warning("Cannot write episode summary: missing episode_id")
            return
        started_at = self._session.start_time_utc
        ended_at = _utc_now_iso()
        labels = failure_labels or list(self._failure_labels)

        summary_payload = EpisodeSummaryPayload(
            episode_id=episode_id,
            session_id=session_id,
            body_id=self._session.metadata.get("body_id"),
            skill_id=self._session.skill_id,
            policy_id=self._session.metadata.get("policy_id"),
            outcome=cast(
                Literal["success", "failure", "partial", "unknown"], _normalize_outcome(outcome)
            ),
            success=outcome == "SUCCESS",
            failure_labels=labels,
            event_count=event_count,
            metrics={
                "duration_ms": duration_ms,
                "reward": reward,
                "source_event_count": self._source_event_count,
            },
        )

        try:
            record = self._artifact_store.write_yaml(
                f"summary_{episode_id}",
                summary_payload.model_dump(mode="json"),
                session_id=session_id,
                episode_id=episode_id,
                artifact_type="summary",
                metadata={"schema_version": EpisodeSummaryPayload.__name__},
            )
            if self._catalog is not None:
                self._catalog.insert_artifact_v2(
                    {
                        "artifact_id": record.artifact_id,
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "artifact_type": record.artifact_type,
                        "path": record.path,
                        "sha256": record.sha256,
                        "size_bytes": record.size_bytes,
                        "schema_name": record.schema_name,
                        "created_at": record.created_at,
                        "metadata": record.metadata,
                    }
                )
        except Exception as e:
            logger.error("Failed to write episode summary YAML: %s", e)

        try:
            self._catalog.insert_session(
                {
                    "session_id": session_id,
                    "practice_id": self._session.practice_id,
                    "body_id": self._session.metadata.get("body_id"),
                    "task_name": self._session.task_name,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "status": "closed",
                    "outcome": summary_payload.outcome,
                    "event_count": event_count,
                    "artifact_count": len(
                        self._artifact_store.list_artifacts(session_id, episode_id)
                    ),
                    "metadata": dict(self._session.metadata),
                }
            )
        except Exception as e:
            logger.error("Failed to insert session into catalog v2: %s", e)

        try:
            self._catalog.insert_episode(
                {
                    "episode_id": episode_id,
                    "session_id": session_id,
                    "body_id": self._session.metadata.get("body_id"),
                    "skill_id": self._session.skill_id,
                    "policy_id": self._session.metadata.get("policy_id"),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "outcome": summary_payload.outcome,
                    "success": summary_payload.success,
                    "failure_labels": labels,
                    "metrics": summary_payload.metrics,
                }
            )
        except Exception as e:
            logger.error("Failed to insert episode into catalog v2: %s", e)

    def _record_event(self, event: RuntimeEvent) -> None:
        envelope = self._runtime_to_envelope(event)

        # Capture body_id from incoming events so the episode summary can be
        # indexed by body even when the session was started without one.
        if envelope.body_id and self._session is not None:
            self._session.metadata.setdefault("body_id", envelope.body_id)

        with self._lock:
            self._event_count += 1
            envelope.sequence_id = self._event_count
            if envelope.event_type not in {"runtime.start", "runtime.stop"}:
                self._source_event_count += 1

        if self._writer is not None:
            try:
                self._writer.write(envelope.model_dump(mode="json"))
            except Exception as e:
                logger.error("Failed to write event to JSONL: %s", e)

        if self._mcap_writer is not None:
            try:
                self._mcap_writer.write(envelope.model_dump(mode="json"))
            except Exception as e:
                logger.error("Failed to write event to MCAP: %s", e)

        if self._catalog is not None:
            try:
                self._catalog.insert_event(
                    {
                        "event_id": envelope.event_id,
                        "practice_id": envelope.practice_id,
                        "source": envelope.source,
                        "event_type": envelope.event_type,
                        "timestamp_ns": envelope.timestamp_ns,
                        "timestamp_utc": envelope.timestamp_utc,
                        "action_id": envelope.action_id,
                        "task_id": envelope.task_id,
                        "skill_id": envelope.skill_id,
                        "payload_ref": json.dumps(envelope.payload_ref)
                        if envelope.payload_ref
                        else None,
                        "tags": ",".join(envelope.tags),
                    }
                )
            except Exception as e:
                logger.error("Failed to insert event into catalog: %s", e)

        if self._publish_to_event_bus and self._legacy_event_bus is not None:
            try:
                self._legacy_event_bus.publish(
                    Event(
                        topic="practice.event",
                        payload=envelope.model_dump(mode="json"),
                        source=f"practice_recorder:{envelope.source}",
                    )
                )
            except Exception as e:
                logger.error("Failed to publish practice.event: %s", e)

    def _runtime_to_envelope(self, event: RuntimeEvent) -> PracticeEventEnvelope:
        if self._session is None:
            raise RuntimeError("No active practice session")
        payload = event.payload or {}

        # If a producer already embedded a full PracticeEventEnvelope, reuse it.
        if payload.get("schema_version") == SCHEMA_VERSION and "practice_id" in payload:
            return PracticeEventEnvelope(**payload)

        md = event.metadata or {}
        source = md.get("source", event.source)
        if source not in _ALLOWED_SOURCES:
            source = "system"

        ts_ns = _datetime_to_ns(event.timestamp)
        ts_utc = _format_utc(event.timestamp)

        body_id = event.body_id or self._session.metadata.get("body_id")
        if event.body_id and "body_id" not in self._session.metadata:
            self._session.metadata["body_id"] = event.body_id

        return PracticeEventEnvelope(
            practice_id=self._session.practice_id,
            session_id=self._session.session_id,
            episode_id=self._session.episode_id,
            robot_id=event.robot or self._session.robot_id,
            body_id=body_id,
            source=source,
            event_type=event.type,
            timestamp_ns=ts_ns,
            timestamp_utc=ts_utc,
            source_timestamp_ns=md.get("source_timestamp_ns"),
            trace_id=md.get("trace_id") or self._session.practice_id,
            parent_event_id=md.get("parent_event_id"),
            frame_id=md.get("frame_id"),
            task_id=md.get("task_id") or self._session.task_id,
            skill_id=md.get("skill_id") or self._session.skill_id,
            action_id=md.get("action_id"),
            payload=payload,
            payload_ref=md.get("payload_ref", {}),
            quality=md.get("quality", {}),
            tags=list(md.get("tags", [])),
        )

    @property
    def session(self) -> PracticeSession | None:
        return self._session

    @property
    def summary(self) -> PracticeSummary | None:
        return self._summary

    # ------------------------------------------------------------------
    # Legacy DataFlywheel / EventBus API (preserved for compatibility)
    # ------------------------------------------------------------------

    def _on_skill_complete(self, event) -> None:
        """Handle skill.execution.complete and publish praxis.completed/failed."""
        if self._legacy_event_bus is None:
            return
        payload = event.payload if hasattr(event, "payload") else {}
        if not isinstance(payload, dict):
            return
        result = payload.get("result", {})
        status = result.get("status")
        correlation_id = (
            payload.get("correlation_id")
            or payload.get("episode_id")
            or payload.get("request_id")
            or ""
        )
        skill_name = payload.get("skill_name", "")

        self._failure_context["current_iteration"] += 1
        iteration = self._failure_context["current_iteration"]

        if status == "success":
            self._failure_context["previous_scores"].append(result.get("reward", 1.0))
            self._legacy_event_bus.publish(
                Event(
                    topic="praxis.completed",
                    payload={
                        "practice_id": correlation_id,
                        "episode_id": correlation_id,
                        "correlation_id": correlation_id,
                        "event_type": "praxis.completed",
                        "robot_id": self.robot_id,
                        "outcome": {
                            "status": "success",
                            "reward": result.get("reward", 1.0),
                            "skill_name": skill_name,
                            "details": {"details": result.get("details", {})},
                        },
                        "current_iteration": iteration,
                        "previous_scores": list(self._failure_context["previous_scores"]),
                        "timestamp": time.time(),
                    },
                    source="practice_recorder",
                    priority=EventPriority.NORMAL,
                )
            )
        elif status == "failure":
            error = result.get("error", "")
            self._failure_context["last_error"] = error
            self._failure_context["previous_scores"].append(result.get("reward", -1.0))
            self._legacy_event_bus.publish(
                Event(
                    topic="praxis.failed",
                    payload={
                        "practice_id": correlation_id,
                        "episode_id": correlation_id,
                        "correlation_id": correlation_id,
                        "event_type": "praxis.failed",
                        "robot_id": self.robot_id,
                        "outcome": {
                            "status": "failure",
                            "reward": result.get("reward", -1.0),
                            "skill_name": skill_name,
                            "details": {"details": result.get("details", {})},
                        },
                        "error_log": error,
                        "current_iteration": iteration,
                        "previous_scores": list(self._failure_context["previous_scores"]),
                        "timestamp": time.time(),
                    },
                    source="practice_recorder",
                    priority=EventPriority.HIGH,
                )
            )

    def _on_knowledge_ingest_complete(self, event) -> None:
        """Handle knowledge.ingest_complete and log to knowledge_ingest_log."""
        payload = event.payload if hasattr(event, "payload") else {}
        if not isinstance(payload, dict):
            return
        self._knowledge_ingest_log.append(
            {
                "practice_id": payload.get("practice_id", "unknown"),
                "knowledge_version": payload.get("knowledge_version", "unknown"),
                "status": payload.get("status", "unknown"),
                "ingest_timestamp": payload.get("timestamp", time.time()),
            }
        )

    def record_recovery_outcome(
        self, rule_id: str, success: bool, duration: float, correlation_id: str = ""
    ) -> None:
        """Record heuristic recovery outcome and publish event."""
        if self._legacy_event_bus is None:
            return
        self._legacy_event_bus.publish(
            Event(
                topic="heuristic.recovery_executed",
                payload={
                    "rule_id": rule_id,
                    "success": success,
                    "duration": duration,
                    "robot_id": self.robot_id,
                    "timestamp": time.time(),
                    "correlation_id": correlation_id,
                },
                source="practice_recorder",
                priority=EventPriority.NORMAL,
            )
        )
        logger.info("Published heuristic.recovery_executed for rule %s", rule_id)

    def start_recording(self) -> None:
        """Start a recording session."""
        self._recording = True
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop recording."""
        self._recording = False
        logger.info("Recording stopped")

    def log_state(self, joint_positions: list[float], timestamp: float) -> None:
        """Log a robot state sample."""
        if not self._recording or self._flywheel is None:
            return
        import numpy as np

        from rosclaw.data.flywheel import RobotState as FlywheelRobotState

        state = FlywheelRobotState(
            timestamp=timestamp,
            joint_positions=np.array(joint_positions),
            joint_velocities=np.zeros(self.joint_dof),
            joint_torques=np.zeros(self.joint_dof),
        )
        self._flywheel.on_control_cycle(state)

    def mark_event(self, event_type: EventType, metadata: dict | None = None) -> str:
        """Mark an event in the recording."""
        if self._flywheel is None:
            return ""
        return self._flywheel.trigger_event(event_type, metadata)

    def export_session(self, output_path: Path) -> Path:
        """Export recorded session to LeRobot format."""
        if self._flywheel is None:
            raise RuntimeError("Flywheel not initialized")
        return self._flywheel.export_to_lerobot(output_path)

    def record_praxis_event(
        self,
        event=None,
        event_id: str = "",
        event_type: str = "",
        instruction: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Record a praxis event on the timeline."""
        if not self._recording or self._flywheel is None:
            return ""

        from rosclaw.core.types import PraxisEvent
        from rosclaw.data.flywheel import EventType as FlywheelEventType

        if isinstance(event, PraxisEvent):
            data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "instruction": event.agent_instruction,
                "duration_sec": event.duration_sec,
                "cot_trace": event.cot_trace,
                "trajectory": event.trajectory,
            }
            try:
                fw_type = FlywheelEventType[event.event_type.upper()]
            except KeyError:
                fw_type = FlywheelEventType.MILESTONE
            return self._flywheel.trigger_event(fw_type, data)

        try:
            fw_type = FlywheelEventType[event_type.upper()]
        except KeyError:
            fw_type = FlywheelEventType.MILESTONE
        return self._flywheel.trigger_event(
            fw_type,
            {
                "event_id": event_id,
                "instruction": instruction,
                **(metadata or {}),
            },
        )

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def failure_context(self) -> dict:
        """Get current failure context."""
        return self._failure_context.copy()

    @property
    def knowledge_ingest_log(self) -> list[dict]:
        """Get knowledge ingest completion log."""
        return self._knowledge_ingest_log.copy()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _datetime_to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
