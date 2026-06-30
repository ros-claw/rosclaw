"""PracticeCoordinator - Physical Data Flywheel Runtime core."""

from __future__ import annotations

import json
import logging
import threading
import time

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.practice.adapters.base import SourceAdapter
from rosclaw.practice.adapters.mock_agent_adapter import MockAgentAdapter
from rosclaw.practice.adapters.mock_runtime_adapter import MockRuntimeAdapter
from rosclaw.practice.adapters.sense_adapter import SenseAdapter
from rosclaw.practice.config import PracticeConfig, PracticeSession, PracticeSummary
from rosclaw.practice.schemas import PracticeEventEnvelope
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout, generate_practice_id
from rosclaw.practice.writers.jsonl_writer import JsonlWriter

from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent

logger = logging.getLogger("rosclaw.practice.coordinator")


class PracticeCoordinator(LifecycleMixin):
    """Coordinates a practice session: adapters, writers, catalog, SeekDB."""

    def __init__(
        self,
        config: PracticeConfig | None = None,
        runtime_bus: RuntimeBus | None = None,
        recorder=None,
    ):
        super().__init__()
        self.config = config or PracticeConfig()
        self.layout = PracticeLayout(self.config.data_root)
        self.catalog = PracticeCatalog(self.layout.catalog_db_path)
        self._event_bus = self.config.event_bus or EventBus()
        self._runtime_bus = runtime_bus or RuntimeBus(event_bus=self._event_bus)
        self._recorder = recorder
        self._adapters: list[SourceAdapter] = []
        self._writer: JsonlWriter | None = None
        self._session: PracticeSession | None = None
        self._summary: PracticeSummary | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._poll_hz = 10.0
        self._event_count = 0
        self._source_event_count = 0
        self._failure_labels: list[str] = []
        self._lock = threading.RLock()

    @property
    def event_bus(self) -> EventBus:
        """Public access to the coordinator's event bus for subscribers."""
        return self._event_bus

    @property
    def runtime_bus(self) -> RuntimeBus:
        """Public access to the RuntimeBus."""
        return self._runtime_bus

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_initialize(self) -> None:
        self.layout.ensure_directories()
        self._build_adapters()

    def _do_start(self) -> None:
        self._start_session()

    def _do_stop(self) -> None:
        self._stop_session()

    # ------------------------------------------------------------------
    # Adapter setup
    # ------------------------------------------------------------------

    def _build_adapters(self) -> None:
        sources = self.config.sources
        if self.config.mock:
            if sources.agent:
                self._adapters.append(
                    MockAgentAdapter(self.config.robot_id, task=self.config.task_name or "mock task")
                )
            if sources.runtime:
                self._adapters.append(MockRuntimeAdapter(self.config.robot_id))
            if sources.dds:
                self._adapters.append(
                    SenseAdapter(
                        self.config.robot_id,
                        sense_runtime=getattr(self.config, "sense_runtime", None),
                        scenario="normal",
                    )
                )
            return

        # Non-mock P0: only DDS via SenseRuntime if available
        if sources.dds:
            self._adapters.append(
                SenseAdapter(
                    self.config.robot_id,
                    sense_runtime=getattr(self.config, "sense_runtime", None),
                    scenario="normal",
                )
            )

    def register_adapter(self, adapter: SourceAdapter) -> None:
        self._adapters.append(adapter)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _start_session(self) -> None:
        practice_id = generate_practice_id()
        session_dir = self.layout.create_session_dirs(practice_id)
        start_ns = time.monotonic_ns()
        from datetime import UTC, datetime

        start_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._session = PracticeSession(
            practice_id=practice_id,
            robot_id=self.config.robot_id,
            robot_type=self.config.robot_type,
            task_id=self.config.task_id,
            task_name=self.config.task_name,
            skill_id=self.config.skill_id,
            session_dir=session_dir,
            start_time_ns=start_ns,
            start_time_utc=start_utc,
        )
        self._event_count = 0
        self._summary = None

        if self._recorder is None:
            self._writer = JsonlWriter(
                self.layout.events_jsonl_path(practice_id),
                rotate_mb=self.config.recorder.jsonl_rotate_mb if self.config.recorder.jsonl_rotate_mb > 0 else None,
            )
            self.catalog.insert_practice(
                {
                    "practice_id": practice_id,
                    "robot_id": self.config.robot_id,
                    "robot_type": self.config.robot_type,
                    "task_id": self.config.task_id,
                    "task_name": self.config.task_name,
                    "skill_id": self.config.skill_id,
                    "start_time": start_utc,
                    "manifest_path": str(self.layout.manifest_path(practice_id)),
                    "events_jsonl_path": str(self.layout.events_jsonl_path(practice_id)),
                    "outcome": "running",
                }
            )
            self.layout.write_manifest(
                self._session,
                sources=self._sources_dict(),
                seekdb_enabled=bool(self.config.seekdb.url),
            )
        else:
            self._runtime_bus.publish(
                RuntimeEvent(
                    type="practice.start",
                    source="practice_coordinator",
                    robot=self.config.robot_id,
                    payload={
                        "practice_id": practice_id,
                        "robot_id": self.config.robot_id,
                        "robot_type": self.config.robot_type,
                        "task_id": self.config.task_id,
                        "task_name": self.config.task_name,
                        "skill_id": self.config.skill_id,
                        "session_id": practice_id,
                        "session_dir": str(session_dir),
                        "sources": self._sources_dict(),
                        "seekdb_enabled": bool(self.config.seekdb.url),
                    },
                    metadata={"trace_id": practice_id},
                )
            )

        for adapter in self._adapters:
            try:
                adapter.start(self._session)
            except Exception as e:
                logger.error("Failed to start adapter %s: %s", adapter.source_name, e)

        if self.config.sources.runtime:
            self._emit(
                PracticeEventEnvelope(
                    practice_id=practice_id,
                    robot_id=self.config.robot_id,
                    source="runtime",
                    event_type="runtime.start",
                    timestamp_ns=start_ns,
                    timestamp_utc=start_utc,
                    trace_id=practice_id,
                    payload={"phase": "start", "sources": self._sources_dict()},
                    tags=["runtime", "start"],
                )
            )

        if self.config.publish_to_event_bus:
            self._event_bus.publish(
                Event(
                    topic="practice.session_started",
                    payload={
                        "practice_id": practice_id,
                        "robot_id": self.config.robot_id,
                        "task_id": self.config.task_id,
                        "task_name": self.config.task_name,
                        "skill_id": self.config.skill_id,
                        "session_dir": str(session_dir),
                        "semantic_intent": self.config.task_name or "",
                    },
                    source="practice_coordinator",
                )
            )

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        logger.info("Practice session started: %s", practice_id)

    def _stop_session(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)

        for adapter in self._adapters:
            try:
                adapter.stop()
            except Exception as e:
                logger.error("Failed to stop adapter %s: %s", adapter.source_name, e)

        duration_ms: float | None = None
        if self._session is not None:
            duration_ms = (time.monotonic_ns() - self._session.start_time_ns) / 1_000_000.0

        if self.config.sources.runtime and self._session is not None:
            self._emit(
                PracticeEventEnvelope(
                    practice_id=self._session.practice_id,
                    robot_id=self.config.robot_id,
                    source="runtime",
                    event_type="runtime.stop",
                    trace_id=self._session.practice_id,
                    payload={"phase": "stop", "event_count": self._event_count, "duration_ms": duration_ms},
                    tags=["runtime", "stop"],
                )
            )

        # Determine outcome. A session that wrote no events at all is never
        # considered successful. Runtime lifecycle events alone are not enough
        # when real data sources were requested.
        requested_sources = self._sources_dict()
        requested_non_runtime = {
            k: v for k, v in requested_sources.items() if k != "runtime" and v
        }

        if self._event_count == 0:
            outcome = "FAILED"
            reward = 0.0
            failure_labels = ["zero_events"]
        elif self._failure_labels:
            outcome = "FAILED"
            reward = 0.0
            failure_labels = list(self._failure_labels)
        elif requested_non_runtime and self._source_event_count == 0:
            outcome = "FAILED"
            reward = 0.0
            failure_labels = ["zero_events"]
        else:
            outcome = "SUCCESS"
            reward = 0.0
            failure_labels = []

        self._summary = PracticeSummary(
            practice_id=self._session.practice_id if self._session else "unknown",
            robot_id=self.config.robot_id,
            outcome=outcome,
            reward=reward,
            duration_ms=duration_ms,
            event_count=self._event_count,
            artifact_dir=self._session.session_dir if self._session else None,
            failure_labels=failure_labels,
        )

        if self._recorder is None:
            self.catalog.update_practice(
                self._session.practice_id if self._session else "unknown",
                {
                    "end_time": _utc_now_iso(),
                    "duration_ms": duration_ms,
                    "outcome": outcome,
                    "reward": reward,
                },
            )

            if self._writer is not None:
                self._writer.close()
                self._writer = None

            if self._session is not None:
                self.layout.write_manifest(
                    self._session,
                    summary=self._summary,
                    sources=self._sources_dict(),
                    seekdb_enabled=bool(self.config.seekdb.url),
                )
                self.layout.finalize_session(
                    self._session.practice_id,
                    self._session,
                    self._summary,
                    sources=self._sources_dict(),
                )

            if self.config.publish_to_event_bus and self._session is not None:
                self._event_bus.publish(
                    Event(
                        topic="practice.session_finished",
                        payload={
                            "practice_id": self._session.practice_id,
                            "robot_id": self.config.robot_id,
                            "outcome": outcome,
                            "reward": reward,
                            "duration_ms": duration_ms,
                            "event_count": self._event_count,
                        },
                        source="practice_coordinator",
                    )
                )
        else:
            if self._session is not None:
                self._runtime_bus.publish(
                    RuntimeEvent(
                        type="practice.stop",
                        source="practice_coordinator",
                        robot=self.config.robot_id,
                        payload={
                            "practice_id": self._session.practice_id,
                            "outcome": outcome,
                            "reward": reward,
                            "duration_ms": duration_ms,
                            "event_count": self._event_count,
                            "failure_labels": failure_labels,
                            "sources": self._sources_dict(),
                            "seekdb_enabled": bool(self.config.seekdb.url),
                        },
                        metadata={"trace_id": self._session.practice_id},
                    )
                )
            if self.config.publish_to_event_bus and self._session is not None:
                self._event_bus.publish(
                    Event(
                        topic="practice.session_finished",
                        payload={
                            "practice_id": self._session.practice_id,
                            "robot_id": self.config.robot_id,
                            "outcome": outcome,
                            "reward": reward,
                            "duration_ms": duration_ms,
                            "event_count": self._event_count,
                        },
                        source="practice_coordinator",
                    )
                )

        logger.info("Practice session stopped: %s", self._session.practice_id if self._session else "unknown")

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        interval = 1.0 / self._poll_hz
        while not self._stop_event.is_set():
            self.tick()
            time.sleep(interval)

    def tick(self) -> None:
        """Poll all adapters once and emit captured events."""
        if self._session is None:
            return
        for adapter in self._adapters:
            try:
                for event in adapter.poll():
                    self._emit(event)
            except Exception as e:
                logger.error("Adapter %s poll failed: %s", adapter.source_name, e)

    def emit_event(self, event: PracticeEventEnvelope) -> None:
        """Public entry for external callers to inject events."""
        self._emit(event)

    def record_failure(self, labels: list[str]) -> None:
        """Record failure labels that will force the session outcome to FAILED."""
        with self._lock:
            for label in labels:
                if label not in self._failure_labels:
                    self._failure_labels.append(label)

    def _emit(self, event: PracticeEventEnvelope) -> None:
        with self._lock:
            self._event_count += 1
            event.sequence_id = self._event_count
            if event.event_type not in {"runtime.start", "runtime.stop"}:
                self._source_event_count += 1

        if self._recorder is None:
            # Legacy file-backed path.
            # Local JSONL
            if self._writer is not None:
                try:
                    self._writer.write(event.model_dump(mode="json"))
                except Exception as e:
                    logger.error("Failed to write event to JSONL: %s", e)

            # Local catalog
            try:
                self.catalog.insert_event(
                    {
                        "event_id": event.event_id,
                        "practice_id": event.practice_id,
                        "source": event.source,
                        "event_type": event.event_type,
                        "timestamp_ns": event.timestamp_ns,
                        "timestamp_utc": event.timestamp_utc,
                        "action_id": event.action_id,
                        "task_id": event.task_id,
                        "skill_id": event.skill_id,
                        "payload_ref": json.dumps(event.payload_ref) if event.payload_ref else None,
                        "tags": ",".join(event.tags),
                    }
                )
            except Exception as e:
                logger.error("Failed to insert event into catalog: %s", e)

            # EventBus
            if self.config.publish_to_event_bus:
                try:
                    self._event_bus.publish(
                        Event(
                            topic="practice.event",
                            payload=event.model_dump(mode="json"),
                            source=f"practice_coordinator:{event.source}",
                        )
                    )
                except Exception as e:
                    logger.error("Failed to publish practice.event: %s", e)
        else:
            # Runtime Kernel v2 path: publish a RuntimeEvent for the recorder.
            try:
                self._runtime_bus.publish(
                    RuntimeEvent(
                        type=event.event_type,
                        source=event.source,
                        robot=event.robot_id,
                        payload=event.model_dump(mode="json"),
                        metadata={
                            "trace_id": event.trace_id or event.practice_id,
                            "source": event.source,
                            "task_id": event.task_id,
                            "skill_id": event.skill_id,
                            "action_id": event.action_id,
                            "frame_id": event.frame_id,
                            "tags": event.tags,
                            "quality": event.quality,
                            "payload_ref": event.payload_ref,
                            "parent_event_id": event.parent_event_id,
                            "source_timestamp_ns": event.source_timestamp_ns,
                        },
                    )
                )
            except Exception as e:
                logger.error("Failed to publish runtime event: %s", e)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sources_dict(self) -> dict[str, bool]:
        return {
            "dds": self.config.sources.dds,
            "ros2": self.config.sources.ros2,
            "camera": self.config.sources.camera,
            "agent": self.config.sources.agent,
            "provider": self.config.sources.provider,
            "sandbox": self.config.sources.sandbox,
            "runtime": self.config.sources.runtime,
            "human": self.config.sources.human,
        }

    @property
    def session(self) -> PracticeSession | None:
        return self._session

    @property
    def summary(self) -> PracticeSummary | None:
        return self._summary




def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
