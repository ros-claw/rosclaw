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
from rosclaw.memory.types import PraxisEvent, FailureMemory, ArtifactRef

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
            # Sprint 8 — Knowledge Plane event subscriptions
            self.event_bus.subscribe("rosclaw.practice.event.created", self._on_practice_event_created)
            self.event_bus.subscribe("rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed)
            self.event_bus.subscribe("rosclaw.sandbox.episode.succeeded", self._on_sandbox_episode_succeeded)
            self.event_bus.subscribe("rosclaw.how.recovery_hint.generated", self._on_recovery_hint_generated)
            self.event_bus.subscribe("firewall.action_blocked", self._on_firewall_action_blocked)

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
            self.event_bus.unsubscribe("rosclaw.practice.event.created", self._on_practice_event_created)
            self.event_bus.unsubscribe("rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed)
            self.event_bus.unsubscribe("rosclaw.sandbox.episode.succeeded", self._on_sandbox_episode_succeeded)
            self.event_bus.unsubscribe("rosclaw.how.recovery_hint.generated", self._on_recovery_hint_generated)
            self.event_bus.unsubscribe("firewall.action_blocked", self._on_firewall_action_blocked)
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

    # -- Sprint 8 event handlers --

    def _on_practice_event_created(self, event: Event) -> None:
        """Auto-ingest practice events into praxis_events table."""
        payload = event.payload
        self.write_praxis_event(PraxisEvent(
            event_id=payload.get("event_id", ""),
            robot_id=payload.get("robot_id", self._robot_id),
            event_type=payload.get("event_type", "practice"),
            episode_id=payload.get("episode_id"),
            task_id=payload.get("task_id"),
            payload=payload,
        ))

    def _on_sandbox_episode_failed(self, event: Event) -> None:
        """Auto-ingest sandbox failures into failures table."""
        payload = event.payload
        self.write_failure_memory(FailureMemory(
            failure_id=payload.get("failure_id", ""),
            robot_id=payload.get("robot_id", self._robot_id),
            episode_id=payload.get("episode_id"),
            task_id=payload.get("task_id"),
            failure_type=payload.get("failure_type", "unknown"),
            root_cause=payload.get("root_cause", ""),
            recovery_hint=payload.get("recovery_hint", ""),
            sandbox_intervened=payload.get("sandbox_intervened", False),
            category=payload.get("category", ""),
        ))

    def _on_sandbox_episode_succeeded(self, event: Event) -> None:
        """Auto-ingest success patterns from sandbox episodes."""
        payload = event.payload
        record = {
            "id": payload.get("pattern_id", ""),
            "skill_id": payload.get("skill_id", ""),
            "robot_id": payload.get("robot_id", self._robot_id),
            "context_hash": payload.get("context_hash", ""),
            "success_count": payload.get("success_count", 1),
            "avg_duration_sec": payload.get("avg_duration_sec", 0.0),
            "metadata": payload,
        }
        self._client.insert("success_patterns", record)

    def _on_recovery_hint_generated(self, event: Event) -> None:
        """Associate recovery hint with the related failure record."""
        payload = event.payload
        failure_id = payload.get("failure_id")
        hint = payload.get("hint", "")
        if failure_id:
            self._client.update("failures", failure_id, {"recovery_hint": hint})

    def _on_firewall_action_blocked(self, event: Event) -> None:
        """Auto-ingest firewall-blocked actions into failures table."""
        payload = event.payload
        self.write_failure_memory(FailureMemory(
            failure_id=payload.get("episode_id", ""),
            robot_id=self._robot_id,
            episode_id=payload.get("episode_id"),
            failure_type="firewall_blocked",
            root_cause=payload.get("reason", "firewall"),
            recovery_hint="Check joint limits, workspace boundaries, and trajectory safety.",
            sandbox_intervened=True,
            category="safety",
        ))

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
            self.event_bus.publish(Event(
                topic="rosclaw.memory.write.completed",
                payload={"table": "experience_graph", "record_id": record_id, "robot_id": self._robot_id},
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

    def find_analogy(self, error_log: str, limit: int = 3) -> Optional[dict]:
        """Find similar past failures and their recovery actions as analogy.

        Searches the experience_graph for failures with similar error_details
        or tags, returns the recovery hint from the most similar one.

        This is the canonical API used by HOW.knowledge_fallback() when
        no heuristic rule matches the failure.

        Returns:
            {"id": str, "action_suggestion": str, "similarity_score": float}
            or None if no similar failure found.
        """
        filters = {"robot_id": self._robot_id, "outcome": "failure"}

        all_failures = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=100,
        )

        if not all_failures:
            return None

        query_tokens = self._tokenize(error_log)
        if not query_tokens:
            return None

        # Search in error_details and tags (not just instruction)
        keywords = set(query_tokens)
        scored = []
        for exp in all_failures:
            error_text = exp.get("error_details", "")
            tags_text = " ".join(exp.get("tags", []))
            text = error_text + " " + tags_text
            exp_tokens = set(self._tokenize(text))
            overlap = len(keywords & exp_tokens)
            if overlap > 0:
                scored.append((overlap, exp))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        metadata = best.get("metadata", {})

        return {
            "id": best.get("id", ""),
            "action_suggestion": metadata.get("recovery_hint", ""),
            "similarity_score": scored[0][0] / len(query_tokens) if query_tokens else 0.0,
            "source_experience": best.get("id", ""),
        }

    def get_experience(self, experience_id: str) -> Optional[dict]:
        """Retrieve a single experience by ID."""
        results = self._client.query(
            "experience_graph",
            filters={"id": experience_id},
            limit=1,
        )
        return results[0] if results else None

    def write_praxis_event(self, event: dict[str, Any]) -> str:
        """Write a praxis event to the experience store (convenience wrapper).

        This is the canonical API used by Runtime.execute() to persist
        completed actions as experiences.
        """
        return self.store_experience(
            event_id=event.get("event_id", ""),
            event_type=event.get("event_type", "praxis"),
            instruction=event.get("instruction", ""),
            cot_trace=event.get("cot_trace"),
            trajectory=event.get("trajectory"),
            outcome=event.get("outcome", "success"),
            duration_sec=event.get("duration_sec", 0.0),
            error_details=event.get("error_details"),
            tags=event.get("tags"),
            metadata=event.get("metadata"),
        )

    def write_failure_memory(self, failure: dict[str, Any]) -> str:
        """Write a failure event to the experience store (convenience wrapper).

        Used by Runtime when sandbox_check blocks or execution fails.
        """
        return self.store_experience(
            event_id=failure.get("failure_id", failure.get("event_id", "")),
            event_type="failure",
            instruction=failure.get("instruction", ""),
            cot_trace=failure.get("cot_trace"),
            trajectory=failure.get("trajectory"),
            outcome="failure",
            duration_sec=failure.get("duration_sec", 0.0),
            error_details=failure.get("error", failure.get("reason", "")),
            tags=["failure", failure.get("failure_type", "unknown")],
            metadata=failure.get("context", {}),
        )

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

    def write(self, key: str, value: dict[str, Any]) -> str:
        """Store an experience by key — convenience alias for store_experience.

        Matches the simple key-value API expected by MCP/demo consumers.
        """
        return self.store_experience(
            event_id=key,
            event_type=value.get("event_type", "experience"),
            instruction=value.get("instruction", key),
            outcome=value.get("outcome", "unknown"),
            duration_sec=value.get("duration_sec", 0.0),
            tags=value.get("tags", []),
            metadata=value,
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search experiences by query text — convenience alias for find_similar_experiences.

        Matches the simple search API expected by MCP/demo consumers.
        """
        return self.find_similar_experiences(instruction=query, limit=limit)

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
    # KNOW / HOW wrappers — Knowledge Graph and Heuristic Rules
    # ------------------------------------------------------------------

    def query_knowledge_graph(
        self,
        entity_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query the knowledge_graph table.

        Args:
            entity_id: Filter by subject (e.g. robot_id).
            predicate: Filter by predicate (e.g. "has_capability").
            object_value: Filter by object.
            limit: Maximum records to return.

        Returns:
            List of knowledge graph records.
        """
        filters: dict[str, Any] = {}
        if entity_id:
            filters["subject"] = entity_id
        if predicate:
            filters["predicate"] = predicate
        if object_value:
            filters["object"] = object_value
        return self._client.query("knowledge_graph", filters=filters, limit=limit)

    def get_heuristic_rules(
        self,
        condition: Optional[str] = None,
        min_priority: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Query heuristic rules from SeekDB.

        Args:
            condition: Optional substring match on condition text.
            min_priority: Minimum priority threshold (default 0).
            limit: Maximum rules to return.

        Returns:
            List of heuristic rule records.
        """
        rows = self._client.query(
            "heuristic_rules",
            filters={} if condition is None else {"condition": condition},
            order_by="-priority",
            limit=limit,
        )
        return [r for r in rows if int(r.get("priority", 0)) >= min_priority]

    # ------------------------------------------------------------------
    # Sprint 8 — Knowledge Plane core APIs
    # ------------------------------------------------------------------

    def write_praxis_event(self, event: PraxisEvent | dict) -> str:
        """Write a PraxisEvent to the SeekDB praxis_events table.

        Accepts either a PraxisEvent instance or a plain dict for
        backward compatibility with integration tests.
        """
        if isinstance(event, dict):
            record = dict(event)
            record.setdefault("robot_id", self._robot_id)
        else:
            record = event.to_seekdb_record()
        return self._client.insert("praxis_events", record)

    def write_failure_memory(self, failure: FailureMemory | dict) -> str:
        """Write a FailureMemory to the SeekDB failures table.

        Accepts either a FailureMemory instance or a plain dict for
        backward compatibility with integration tests.
        """
        if isinstance(failure, dict):
            record = dict(failure)
            record.setdefault("robot_id", self._robot_id)
        else:
            record = failure.to_seekdb_record()
        return self._client.insert("failures", record)

    def retrieve_similar_episode(
        self,
        task_id: Optional[str] = None,
        robot_id: Optional[str] = None,
        n: int = 5,
    ) -> list[dict]:
        """Retrieve similar historical episodes.

        Filters by task_id and/or robot_id, ordered by most recent.
        """
        filters: dict[str, Any] = {}
        if task_id:
            filters["task_id"] = task_id
        if robot_id:
            filters["robot_id"] = robot_id
        return self._client.query(
            "episodes",
            filters=filters,
            order_by="-started_at",
            limit=n,
        )

    def explain_last_failure(
        self,
        task_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Explain the most recent failure for a task.

        Returns the latest failure record including root_cause,
        recovery_hint, and sandbox_intervened flag.
        """
        filters: dict[str, Any] = {}
        if task_id:
            filters["task_id"] = task_id
        results = self._client.query(
            "failures",
            filters=filters,
            order_by="-timestamp",
            limit=1,
        )
        if not results:
            return None
        return dict(results[0])

    def retrieve_robot_capability(
        self,
        robot_id: Optional[str] = None,
    ) -> list[dict]:
        """Query capabilities for a robot from the knowledge_graph."""
        rid = robot_id or self._robot_id
        return self.query_knowledge_graph(
            entity_id=rid,
            predicate="has_capability",
            limit=100,
        )

    def retrieve_skill_success_pattern(
        self,
        skill_name: str,
        robot_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Retrieve success pattern for a skill + robot combination."""
        filters: dict[str, Any] = {"skill_id": skill_name}
        if robot_id:
            filters["robot_id"] = robot_id
        results = self._client.query(
            "success_patterns",
            filters=filters,
            limit=1,
        )
        return dict(results[0]) if results else None

    def retrieve_safety_case(
        self,
        robot_id: Optional[str] = None,
        constraint_type: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve safety-related heuristic rules for a robot.

        Looks up heuristic rules whose condition mentions safety
        constraints or the given constraint_type.
        """
        filters: dict[str, Any] = {}
        if constraint_type:
            filters["condition"] = constraint_type
        rows = self._client.query(
            "heuristic_rules",
            filters=filters,
            order_by="-priority",
            limit=50,
        )
        return [r for r in rows if int(r.get("priority", 0)) >= 0]

    # -- Artifact handling --

    def store_artifact(self, artifact: ArtifactRef) -> str:
        """Store an artifact reference in SeekDB.

        Large files (MCAP, video, replay) live in the local object
        store at ``./.rosclaw/artifacts/``.  SeekDB only keeps the URI.
        """
        record = artifact.to_seekdb_record()
        return self._client.insert("artifacts", record)

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRef]:
        """Retrieve a single artifact reference by ID."""
        rows = self._client.query(
            "artifacts",
            filters={"id": artifact_id},
            limit=1,
        )
        if not rows:
            return None
        return ArtifactRef(
            artifact_id=rows[0]["id"],
            artifact_type=rows[0].get("artifact_type", ""),
            uri=rows[0].get("uri", ""),
            episode_id=rows[0].get("episode_id"),
            size_bytes=rows[0].get("size_bytes"),
            created_at=rows[0].get("created_at", 0.0),
            metadata=rows[0].get("metadata") or {},
        )

    def find_artifacts_by_episode(
        self,
        episode_id: str,
        artifact_type: Optional[str] = None,
    ) -> list[ArtifactRef]:
        """Find all artifacts linked to an episode."""
        filters: dict[str, Any] = {"episode_id": episode_id}
        if artifact_type:
            filters["artifact_type"] = artifact_type
        rows = self._client.query("artifacts", filters=filters, limit=100)
        return [
            ArtifactRef(
                artifact_id=r["id"],
                artifact_type=r.get("artifact_type", ""),
                uri=r.get("uri", ""),
                episode_id=r.get("episode_id"),
                size_bytes=r.get("size_bytes"),
                created_at=r.get("created_at", 0.0),
                metadata=r.get("metadata") or {},
            )
            for r in rows
        ]

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


# ---------------------------------------------------------------------------
# KNOW / HOW wrapper classes
# ---------------------------------------------------------------------------


class KnowledgeGraphWrapper:
    """Convenience wrapper for SeekDB knowledge_graph table.

    Provides typed access to triples (subject-predicate-object) without
    requiring callers to know the SeekDB schema details.
    """

    def __init__(self, client: SeekDBClient):
        self._client = client
        self._table = "knowledge_graph"

    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query triples by any combination of subject/predicate/object."""
        filters: dict[str, Any] = {}
        if subject:
            filters["subject"] = subject
        if predicate:
            filters["predicate"] = predicate
        if obj:
            filters["object"] = obj
        return self._client.query(self._table, filters=filters, limit=limit)

    def add_triple(
        self,
        triple_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> str:
        """Insert a new triple. Returns the triple id."""
        record = {
            "id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time(),
        }
        return self._client.insert(self._table, record)

    def get_capabilities(self, robot_id: str) -> list[dict]:
        """Return capabilities for a robot as {capability, confidence, source}."""
        rows = self._client.query(
            self._table,
            filters={"subject": robot_id, "predicate": "has_capability"},
            order_by="-confidence",
            limit=100,
        )
        return [
            {
                "capability": r.get("object", ""),
                "confidence": r.get("confidence", 1.0),
                "source": r.get("source", ""),
            }
            for r in rows
        ]

    def count(self) -> int:
        """Total number of triples in the knowledge graph."""
        return self._client.count(self._table)


class HeuristicRuleWrapper:
    """Convenience wrapper for SeekDB heuristic_rules table.

    Provides typed CRUD for heuristic rules used by the HOW recovery engine.
    """

    def __init__(self, client: SeekDBClient):
        self._client = client
        self._table = "heuristic_rules"

    def list_rules(
        self,
        condition_filter: Optional[str] = None,
        min_priority: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Return active rules ordered by priority descending."""
        filters = {}
        if condition_filter:
            filters["condition"] = condition_filter
        rows = self._client.query(
            self._table,
            filters=filters,
            order_by="-priority",
            limit=limit,
        )
        return [r for r in rows if int(r.get("priority", 0)) >= min_priority]

    def add_rule(
        self,
        rule_id: str,
        condition: str,
        action: str,
        priority: int = 0,
    ) -> str:
        """Insert a new heuristic rule. Returns rule_id."""
        record = {
            "id": rule_id,
            "condition": condition,
            "action": action,
            "priority": priority,
            "success_count": 0,
            "failure_count": 0,
            "last_triggered": None,
        }
        return self._client.insert(self._table, record)

    def get_rule(self, rule_id: str) -> Optional[dict]:
        """Retrieve a single rule by id."""
        rows = self._client.query(self._table, filters={"id": rule_id}, limit=1)
        return dict(rows[0]) if rows else None

    def update_rule(self, rule_id: str, **fields: Any) -> bool:
        """Update rule fields. Allowed: condition, action, priority."""
        allowed = {"condition", "action", "priority"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        return self._client.update(self._table, rule_id, updates)

    def record_trigger(self, rule_id: str, success: bool = True) -> bool:
        """Increment success_count or failure_count and update last_triggered."""
        rule = self.get_rule(rule_id)
        if not rule:
            return False
        key = "success_count" if success else "failure_count"
        new_count = int(rule.get(key, 0)) + 1
        return self._client.update(
            self._table,
            rule_id,
            {key: new_count, "last_triggered": time.time()},
        )

    def count(self) -> int:
        """Total number of heuristic rules."""
        return self._client.count(self._table)
