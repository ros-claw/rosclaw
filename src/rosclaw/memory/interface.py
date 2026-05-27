"""MemoryInterface - Experience Grounding backed by SeekDB.

Replaces the in-memory list with SeekDB persistence.
Subscribes to praxis.recorded events to auto-ingest experiences.

Optional integration with powermem.EmbodiedMemory for:
- World object storage (Object Permanence)
- Trajectory memory with DTW similarity
- Cognitive search (semantic + spatial + temporal)
- Scene graph management

Sprint 5 of DESIGN_SPRINT3_5.
"""

from typing import Optional, Any
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

    When ``embodied_memory`` (powermem.EmbodiedMemory) is provided,
    additional capabilities are unlocked:
    - World object CRUD and spatial search
    - Trajectory storage and DTW similarity search
    - Cognitive search (semantic + spatial + temporal)
    - Object permanence (occlusion-aware scene sync)
    - Scene graph and spatial relations

    EventBus:
        Subscribes: praxis.recorded (to auto-ingest new experiences)
        Publishes:  memory.experience.stored
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: Optional[EventBus] = None,
        seekdb_client: Optional[SeekDBClient] = None,
        embodied_memory: Optional[Any] = None,
    ):
        super().__init__()
        self._robot_id = robot_id
        self.event_bus = event_bus
        self._client = seekdb_client or SeekDBMemoryClient()
        self._embodied = embodied_memory

    def _do_initialize(self) -> None:
        self._client.connect()

        if self._embodied is not None and hasattr(self._embodied, "db_conn"):
            print(f"[MemoryInterface] EmbodiedMemory attached: {type(self._embodied).__name__}")

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
                    "embodied_memory": self._embodied is not None,
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

    # ------------------------------------------------------------------
    # SeekDB Experience APIs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # EmbodiedMemory bridge (world objects, trajectories, cognitive search)
    # ------------------------------------------------------------------

    @property
    def has_embodied_memory(self) -> bool:
        """Whether an EmbodiedMemory instance is attached."""
        return self._embodied is not None

    # -- World Objects --

    def add_world_object(self, obj: Any) -> Optional[str]:
        """Add a world object. Returns obj_id or None if no EmbodiedMemory."""
        if self._embodied is None:
            return None
        return self._embodied.add_world_object(obj)

    def get_world_object(self, obj_id: str) -> Optional[Any]:
        """Get a world object by ID."""
        if self._embodied is None:
            return None
        return self._embodied.get_world_object(obj_id)

    def update_world_object_pose(self, obj_id: str, pose: Any, state: Optional[str] = None) -> bool:
        """Update world object pose and optional state."""
        if self._embodied is None:
            return False
        return self._embodied.update_world_object_pose(obj_id, pose, state)

    def search_world_objects(
        self,
        center: Any,
        radius: float,
        scene_id: Optional[str] = None,
    ) -> list[Any]:
        """Search world objects within spatial radius."""
        if self._embodied is None:
            return []
        return self._embodied.search_world_objects(center, radius, scene_id)

    def get_scene_graph(self, scene_id: str) -> tuple[list[Any], list[Any]]:
        """Get scene graph: (objects, relations)."""
        if self._embodied is None:
            return [], []
        return self._embodied.get_scene_graph(scene_id)

    def compute_relations(self, scene_id: str, spatial_tolerance: float = 0.05) -> list[Any]:
        """Compute spatial relations for a scene."""
        if self._embodied is None:
            return []
        return self._embodied.compute_relations(scene_id, spatial_tolerance)

    # -- Object Permanence --

    def sync_scene_objects(
        self,
        scene_id: str,
        detections: list[Any],
        timestamp_sec: float,
        occlusion_radius: float = 0.5,
    ) -> Optional[Any]:
        """
        Sync sensor detections with world model (Object Permanence).

        Returns PermanenceReport or None if no EmbodiedMemory.
        """
        if self._embodied is None:
            return None
        return self._embodied.sync_scene_objects(
            scene_id, detections, timestamp_sec, occlusion_radius
        )

    # -- Trajectories --

    def record_trajectory(self, content: str, waypoints: list[tuple[Any, float]]) -> Optional[int]:
        """Record a trajectory. Returns memory_id or None."""
        if self._embodied is None:
            return None
        return self._embodied.record_trajectory(content, waypoints)

    def search_similar_trajectories(
        self,
        query_waypoints: list[tuple[Any, float]],
        top_k: int = 5,
        max_dtw_distance: Optional[float] = None,
    ) -> list[tuple[Any, float]]:
        """Search for similar trajectories using DTW."""
        if self._embodied is None:
            return []
        return self._embodied.search_similar_trajectories(
            query_waypoints, top_k, max_dtw_distance
        )

    # -- Cognitive Search --

    def cognitive_search(
        self,
        query: str,
        spatial_center: Optional[Any] = None,
        spatial_radius: float = 2.0,
        temporal_interval: Optional[Any] = None,
        limit: int = 10,
    ) -> list[Any]:
        """Cognitive search: semantic + spatial + temporal."""
        if self._embodied is None:
            return []
        kwargs = {"query": query, "limit": limit}
        if spatial_center is not None:
            kwargs["spatial_center"] = spatial_center
            kwargs["spatial_radius"] = spatial_radius
        if temporal_interval is not None:
            kwargs["temporal_interval"] = temporal_interval
        return self._embodied.cognitive_search(**kwargs)

    # -- Meditation (offline abstraction) --

    def run_meditation(self, phases: Optional[list[str]] = None) -> dict:
        """Run meditation pipeline for offline memory abstraction."""
        if self._embodied is None:
            return {"success": False, "error": "EmbodiedMemory not attached"}
        return self._embodied.run_meditation(phases or ["consolidate", "crystallize", "extract"])
