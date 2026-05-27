"""MemoryInterface - Experience Grounding backed by SeekDB.

Replaces the in-memory list with SeekDB persistence.
Subscribes to praxis.recorded events to auto-ingest experiences.

Sprint 5 of DESIGN_SPRINT3_5.
"""

from typing import Optional
import json
import time

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.memory.seekdb_client import SeekDBClient, SeekDBMemoryClient


class MemoryInterface(LifecycleMixin):
    """
    Experience Grounding engine backed by SeekDB.

    Stores PraxisEvents as experiences in the experience_graph table.
    Provides similarity search for finding relevant past experiences.

    EventBus:
        Subscribes: praxis.recorded (to auto-ingest new experiences)
        Publishes:  memory.experience.stored
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: Optional[EventBus] = None,
        seekdb_client: Optional[SeekDBClient] = None,
    ):
        super().__init__()
        self._robot_id = robot_id
        self.event_bus = event_bus
        self._client = seekdb_client or SeekDBMemoryClient()

    def _do_initialize(self) -> None:
        self._client.connect()

        if self.event_bus is not None:
            self.event_bus.subscribe("praxis.recorded", self._on_praxis_recorded)

        print(f"[MemoryInterface] Initialized for {self._robot_id}, "
              f"backend={type(self._client).__name__}")

    def _do_start(self) -> None:
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="memory.status",
                payload={
                    "state": "running",
                    "robot_id": self._robot_id,
                    "experience_count": self._client.count("experience_graph"),
                },
                source="memory_interface",
            ))

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("praxis.recorded", self._on_praxis_recorded)
        self._client.disconnect()

    def _on_praxis_recorded(self, event: Event) -> None:
        """Auto-ingest PraxisEvent as an experience."""
        payload = event.payload
        self.store_experience(
            event_id=payload.get("event_id", ""),
            event_type=payload.get("event_type", "unknown"),
            instruction=payload.get("instruction", ""),
            duration_sec=payload.get("duration_sec", 0.0),
            metadata=payload,
        )

    def store_experience(
        self,
        event_id: str,
        event_type: str,
        instruction: str,
        cot_trace: Optional[list[str]] = None,
        trajectory: Optional[list[list[float]]] = None,
        outcome: str = "success",
        duration_sec: float = 0.0,
        error_details: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new experience in SeekDB."""
        record = {
            "id": event_id,
            "event_type": event_type,
            "robot_id": self._robot_id,
            "timestamp": time.time(),
            "instruction": instruction,
            "cot_trace": cot_trace or [],
            "trajectory": trajectory or [],
            "outcome": outcome,
            "duration_sec": duration_sec,
            "error_details": error_details,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        record_id = self._client.insert("experience_graph", record)

        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="memory.experience.stored",
                payload={"experience_id": record_id, "event_type": event_type},
                source="memory_interface",
            ))

        return record_id

    def find_similar_experiences(
        self,
        instruction: str,
        limit: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find past experiences similar to the given instruction.

        Current implementation: keyword matching.
        Future: vector embeddings for semantic similarity.
        """
        filters = {"robot_id": self._robot_id}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        all_experiences = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=100,
        )

        keywords = set(instruction.lower().split())
        scored = []
        for exp in all_experiences:
            exp_text = (exp.get("instruction", "") + " " +
                       " ".join(exp.get("tags", []))).lower()
            exp_words = set(exp_text.split())
            overlap = len(keywords & exp_words)
            if overlap > 0:
                scored.append((overlap, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def get_experience(self, experience_id: str) -> Optional[dict]:
        """Retrieve a single experience by ID."""
        results = self._client.query(
            "experience_graph",
            filters={"id": experience_id},
            limit=1,
        )
        return results[0] if results else None

    def get_statistics(self) -> dict:
        """Get experience statistics."""
        total = self._client.count("experience_graph")
        successes = self._client.count("experience_graph", {"outcome": "success"})
        failures = self._client.count("experience_graph", {"outcome": "failure"})
        emergencies = self._client.count("experience_graph", {"outcome": "emergency"})

        return {
            "total_experiences": total,
            "success_count": successes,
            "failure_count": failures,
            "emergency_count": emergencies,
            "success_rate": successes / total if total > 0 else 0.0,
        }
