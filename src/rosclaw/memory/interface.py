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

# Conditional import: powermem Protocol types for type-safe proxy methods
try:
    from powermem.embodied.protocols import (
        WorldObjectLike,
        PoseLike,
        Vec3Like,
        TemporalIntervalLike,
        PermanenceReportLike,
        MemoryAtomLike,
    )
    _HAS_POWERMEM_PROTOCOLS = True
except ImportError:
    _HAS_POWERMEM_PROTOCOLS = False
    # Fallback: use Any when powermem not installed
    WorldObjectLike = Any  # type: ignore
    PoseLike = Any  # type: ignore
    Vec3Like = Any  # type: ignore
    TemporalIntervalLike = Any  # type: ignore
    PermanenceReportLike = Any  # type: ignore
    MemoryAtomLike = Any  # type: ignore

# Conditional import: BM25 ranking for semantic search
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


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

    @property
    def seekdb_client(self) -> SeekDBClient:
        """Public accessor for the SeekDB client (used by HOW/KNOW modules)."""
        return self._client

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

        # Periodic capacity check (every ~100 inserts to avoid overhead)
        self._insert_count = getattr(self, "_insert_count", 0) + 1
        if self._insert_count % 100 == 0:
            self.enforce_capacity()

        return record_id

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25/keyword matching.

        Lowercases, splits on whitespace and punctuation, removes short tokens.
        """
        import re
        tokens = re.findall(r'[a-z0-9一-鿿]+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def find_similar_experiences(
        self,
        instruction: str,
        limit: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find past experiences similar to the given instruction.

        Uses BM25Okapi ranking when ``rank_bm25`` is installed,
        falling back to keyword set intersection otherwise.
        """
        filters = {"robot_id": self._robot_id}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        all_experiences = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=200,
        )

        if not all_experiences:
            return []

        query_tokens = self._tokenize(instruction)
        if not query_tokens:
            return []

        if _HAS_BM25:
            return self._bm25_search(all_experiences, query_tokens, limit)
        return self._keyword_search(all_experiences, query_tokens, limit)

    def _bm25_search(
        self,
        experiences: list[dict],
        query_tokens: list[str],
        limit: int,
    ) -> list[dict]:
        """Rank experiences using BM25Okapi.

        Falls back to keyword matching when BM25 produces no positive
        scores (common with small corpora where IDF can go negative).
        """
        corpus = []
        for exp in experiences:
            text = exp.get("instruction", "") + " " + " ".join(exp.get("tags", []))
            corpus.append(self._tokenize(text))

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        # Pair scores with experiences and filter non-positive scores
        scored = [
            (score, exp)
            for score, exp in zip(scores, experiences)
            if score > 0
        ]

        if not scored:
            # BM25 IDF can go negative for terms in >50% of docs (small corpus).
            # Fall back to keyword matching which always works for overlaps.
            return self._keyword_search(experiences, query_tokens, limit)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def _keyword_search(
        self,
        experiences: list[dict],
        query_tokens: list[str],
        limit: int,
    ) -> list[dict]:
        """Fallback: keyword set intersection scoring."""
        keywords = set(query_tokens)
        scored = []
        for exp in experiences:
            text = exp.get("instruction", "") + " " + " ".join(exp.get("tags", []))
            exp_words = set(self._tokenize(text))
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
    # Capacity management (forgetting / eviction)
    # ------------------------------------------------------------------

    DEFAULT_MAX_EXPERIENCES = 10_000
    DEFAULT_MAX_AGE_DAYS = 30

    def delete_experience(self, experience_id: str) -> bool:
        """Delete a single experience by ID."""
        return self._client.delete("experience_graph", experience_id)

    def forget_old_experiences(
        self,
        max_age_days: Optional[int] = None,
        outcome_filter: Optional[str] = None,
    ) -> int:
        max_age_days = self.DEFAULT_MAX_AGE_DAYS if max_age_days is None else max_age_days
        """Delete experiences older than ``max_age_days``.

        Args:
            max_age_days: Maximum age in days. Experiences older than this
                are deleted. Defaults to 30 days.
            outcome_filter: If set, only forget experiences with this outcome
                (e.g. "failure" to clean up old failures first).

        Returns:
            Number of experiences deleted.
        """
        cutoff = time.time() - (max_age_days * 86400)

        # Find old experiences
        filters = {"robot_id": self._robot_id}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        all_experiences = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="timestamp",
            limit=10_000,
        )

        deleted = 0
        for exp in all_experiences:
            ts = exp.get("timestamp", 0)
            if ts < cutoff:
                if self._client.delete("experience_graph", exp.get("id", "")):
                    deleted += 1
            else:
                break  # Sorted by timestamp, no more old ones

        if deleted > 0:
            print(f"[MemoryInterface] Forgot {deleted} experiences "
                  f"(older than {max_age_days} days)")
        return deleted

    def enforce_capacity(self, max_experiences: Optional[int] = None) -> int:
        max_experiences = self.DEFAULT_MAX_EXPERIENCES if max_experiences is None else max_experiences
        """Evict oldest experiences when capacity is exceeded.

        Removes the oldest experiences until the total count is at or below
        ``max_experiences``.

        Returns:
            Number of experiences evicted.
        """
        total = self._client.count("experience_graph",
                                   {"robot_id": self._robot_id})
        if total <= max_experiences:
            return 0

        excess = total - max_experiences
        # Find oldest experiences
        oldest = self._client.query(
            "experience_graph",
            filters={"robot_id": self._robot_id},
            order_by="timestamp",
            limit=excess,
        )

        evicted = 0
        for exp in oldest:
            if self._client.delete("experience_graph", exp.get("id", "")):
                evicted += 1

        if evicted > 0:
            print(f"[MemoryInterface] Evicted {evicted} experiences "
                  f"(capacity: {max_experiences})")
        return evicted

    def get_capacity_info(self) -> dict:
        """Get capacity information for the experience store.

        Returns:
            dict with total, max_experiences, utilization, and oldest/newest
            timestamps.
        """
        total = self._client.count("experience_graph",
                                   {"robot_id": self._robot_id})

        # Find oldest and newest
        oldest_list = self._client.query(
            "experience_graph",
            filters={"robot_id": self._robot_id},
            order_by="timestamp",
            limit=1,
        )
        newest_list = self._client.query(
            "experience_graph",
            filters={"robot_id": self._robot_id},
            order_by="-timestamp",
            limit=1,
        )

        oldest_ts = oldest_list[0].get("timestamp", 0) if oldest_list else 0
        newest_ts = newest_list[0].get("timestamp", 0) if newest_list else 0
        age_days = (newest_ts - oldest_ts) / 86400 if oldest_ts and newest_ts else 0

        return {
            "total_experiences": total,
            "max_experiences": self.DEFAULT_MAX_EXPERIENCES,
            "utilization": total / self.DEFAULT_MAX_EXPERIENCES
                if self.DEFAULT_MAX_EXPERIENCES > 0 else 0.0,
            "oldest_timestamp": oldest_ts,
            "newest_timestamp": newest_ts,
            "age_span_days": round(age_days, 1),
        }

    # ------------------------------------------------------------------
    # EmbodiedMemory bridge (world objects, trajectories, cognitive search)
    # ------------------------------------------------------------------

    @property
    def has_embodied_memory(self) -> bool:
        """Whether an EmbodiedMemory instance is attached."""
        return self._embodied is not None

    # -- World Objects --

    def add_world_object(self, obj: WorldObjectLike) -> Optional[str]:
        """Add a world object. Returns obj_id or None if no EmbodiedMemory."""
        if self._embodied is None:
            return None
        return self._embodied.add_world_object(obj)

    def get_world_object(self, obj_id: str) -> Optional[WorldObjectLike]:
        """Get a world object by ID."""
        if self._embodied is None:
            return None
        return self._embodied.get_world_object(obj_id)

    def update_world_object_pose(
        self, obj_id: str, pose: PoseLike, state: Optional[str] = None
    ) -> bool:
        """Update world object pose and optional state."""
        if self._embodied is None:
            return False
        return self._embodied.update_world_object_pose(obj_id, pose, state)

    def search_world_objects(
        self,
        center: Vec3Like,
        radius: float,
        scene_id: Optional[str] = None,
    ) -> list[WorldObjectLike]:
        """Search world objects within spatial radius."""
        if self._embodied is None:
            return []
        return self._embodied.search_world_objects(center, radius, scene_id)

    def get_scene_graph(
        self, scene_id: str
    ) -> tuple[list[WorldObjectLike], list[Any]]:
        """Get scene graph: (objects, relations)."""
        if self._embodied is None:
            return [], []
        return self._embodied.get_scene_graph(scene_id)

    def compute_relations(
        self, scene_id: str, spatial_tolerance: float = 0.05
    ) -> list[Any]:
        """Compute spatial relations for a scene."""
        if self._embodied is None:
            return []
        return self._embodied.compute_relations(scene_id, spatial_tolerance)

    # -- Object Permanence --

    def sync_scene_objects(
        self,
        scene_id: str,
        detections: list[WorldObjectLike],
        timestamp_sec: float,
        occlusion_radius: float = 0.5,
    ) -> Optional[PermanenceReportLike]:
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

    def record_trajectory(
        self, content: str, waypoints: list[tuple[Vec3Like, float]]
    ) -> Optional[int]:
        """Record a trajectory. Returns memory_id or None."""
        if self._embodied is None:
            return None
        return self._embodied.record_trajectory(content, waypoints)

    def search_similar_trajectories(
        self,
        query_waypoints: list[tuple[Vec3Like, float]],
        top_k: int = 5,
        max_dtw_distance: Optional[float] = None,
    ) -> list[tuple[MemoryAtomLike, float]]:
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
        spatial_center: Optional[Vec3Like] = None,
        spatial_radius: float = 2.0,
        temporal_interval: Optional[TemporalIntervalLike] = None,
        limit: int = 10,
    ) -> list[MemoryAtomLike]:
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
