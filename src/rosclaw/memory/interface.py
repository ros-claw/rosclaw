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

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.memory.seekdb_client import SeekDBClient, SeekDBMemoryClient
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent

logger = logging.getLogger("rosclaw.memory.interface")

# Conditional import: powermem Protocol types for type-safe proxy methods
try:
    from powermem.embodied.protocols import (
        MemoryAtomLike,
        PermanenceReportLike,
        PoseLike,
        TemporalIntervalLike,
        Vec3Like,
        WorldObjectLike,
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

# Conditional import: sklearn TF-IDF for lightweight semantic similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


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
        event_bus: EventBus | None = None,
        seekdb_client: SeekDBClient | None = None,
        embodied_memory: Any | None = None,
    ):
        super().__init__()
        self._robot_id = robot_id
        self.event_bus = event_bus
        self._client = seekdb_client or SeekDBMemoryClient()
        self._embodied = embodied_memory
        self._sense_runtime: Any | None = None
        self._memory_writer_adapter: Any | None = None
        # Semantic-search cache: invalidated on any experience mutation
        self._search_cache_lock = threading.Lock()
        self._search_cache: dict[str, Any] = {}
        self._cache_version = 0
        # Background preloader: rebuilds index during idle time
        self._preload_thread: threading.Thread | None = None
        self._preload_stop = threading.Event()
        # Query-result cache: same query within 5s returns cached results
        self._query_result_cache: dict[str, tuple[list[dict], float]] = {}
        self._query_cache_ttl_sec = 5.0

    def _preload_worker(self) -> None:
        """Daemon thread that rebuilds the semantic index when stale."""
        while not self._preload_stop.is_set():
            # Wait up to 2 seconds or until stop is requested
            if self._preload_stop.wait(timeout=2.0):
                break
            # Check if cache is stale
            with self._search_cache_lock:
                cache = self._search_cache
                if cache.get("version") == self._cache_version:
                    continue  # Cache is warm
                # Rebuild in background
                try:
                    filters = {"robot_id": self._robot_id}
                    all_experiences = self._client.query(
                        "experience_graph",
                        filters=filters,
                        order_by="-timestamp",
                        limit=200,
                    )
                    if all_experiences:
                        self._search_cache = self._build_search_cache(
                            all_experiences, filters
                        )
                        logger.debug("Preloaded semantic index (%d experiences)",
                                     len(all_experiences))
                except Exception as exc:
                    logger.debug("Preload failed: %s", exc)

    def _do_initialize(self) -> None:
        self._client.connect()

        if self._embodied is not None and hasattr(self._embodied, "db_conn"):
            logger.info("EmbodiedMemory attached: %s", type(self._embodied).__name__)

        if self.event_bus is not None:
            self.event_bus.subscribe("praxis.recorded", self._on_praxis_recorded)
            # Sprint 8 — Knowledge Plane event subscriptions
            self.event_bus.subscribe("rosclaw.practice.event.created", self._on_practice_event_created)
            self.event_bus.subscribe("rosclaw.sandbox.episode.failed", self._on_sandbox_episode_failed)
            self.event_bus.subscribe("rosclaw.sandbox.episode.succeeded", self._on_sandbox_episode_succeeded)
            self.event_bus.subscribe("rosclaw.how.recovery_hint.generated", self._on_recovery_hint_generated)
            self.event_bus.subscribe("firewall.action_blocked", self._on_firewall_action_blocked)

        logger.info("Initialized for %s, backend=%s", self._robot_id, type(self._client).__name__)

    def _do_start(self) -> None:
        # Start background preloader to eliminate cold-start latency
        if self._preload_thread is None or not self._preload_thread.is_alive():
            self._preload_stop.clear()
            self._preload_thread = threading.Thread(
                target=self._preload_worker, daemon=True, name="memory-preload"
            )
            self._preload_thread.start()

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
        # Signal and join background preloader
        if self._preload_thread is not None and self._preload_thread.is_alive():
            self._preload_stop.set()
            self._preload_thread.join(timeout=1.0)
            self._preload_thread = None

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

    def set_sense_runtime(self, sense_runtime: Any | None) -> None:
        """Late-bind the SenseRuntime after it has been initialized.

        MemoryInterface is created before SenseRuntime in Runtime, so the
        sense reference must be injected via this setter rather than the
        constructor.
        """
        self._sense_runtime = sense_runtime
        if sense_runtime is not None:
            try:
                from rosclaw.sense.adapters.memory_writer import MemoryWriterAdapter
                self._memory_writer_adapter = MemoryWriterAdapter(sense_runtime)
            except Exception:
                logger.warning("Failed to initialize MemoryWriterAdapter", exc_info=True)
        else:
            self._memory_writer_adapter = None

    def _enrich_record_with_body_sense(self, record: dict[str, Any]) -> dict[str, Any]:
        """Attach body-sense evidence to a memory record when available."""
        if self._memory_writer_adapter is None:
            return record
        try:
            return self._memory_writer_adapter.apply(record)
        except Exception:
            logger.warning("MemoryWriterAdapter failed; storing record unchanged", exc_info=True)
            return record

    def _on_praxis_recorded(self, event: Event) -> None:
        """Auto-ingest PraxisEvent as an experience."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        instruction = payload.get("instruction", "")
        if not instruction:
            instruction = (
                payload.get("skill_name", "")
                or payload.get("agent_instruction", "")  # noqa: W503
                or (payload.get("episode_metadata", {}) or {}).get("agent_instruction", "")  # noqa: W503
                or "unnamed_task"  # noqa: W503
            )
        self.store_experience(
            event_id=str(payload.get("event_id") or payload.get("episode_id") or ""),
            event_type=payload.get("event_type", "unknown"),
            instruction=instruction,
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
        cot_trace: list[str] | None = None,
        trajectory: list[list[float]] | None = None,
        outcome: str = "success",
        duration_sec: float = 0.0,
        error_details: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
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
        self._invalidate_search_cache()

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

    def _invalidate_search_cache(self) -> None:
        """Bump cache version so the next semantic query rebuilds the index."""
        with self._search_cache_lock:
            self._cache_version += 1
            self._search_cache.clear()
            self._query_result_cache.clear()

    def _build_search_cache(
        self,
        experiences: list[dict],
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        """Pre-build TF-IDF matrix and BM25 index from experiences."""
        cache: dict[str, Any] = {
            "version": self._cache_version,
            "filters": dict(filters),
            "experiences": experiences,
        }

        # Build texts and corpus
        texts: list[str] = []
        corpus: list[list[str]] = []
        for exp in experiences:
            instruction = exp.get("instruction", "")
            tags = " ".join(exp.get("tags", []))
            error = exp.get("error_details", "")
            text = " ".join(p for p in [instruction, tags, error] if p)
            texts.append(text)
            corpus.append(self._tokenize(text))

        cache["corpus"] = corpus

        # TF-IDF
        if _HAS_SKLEARN and texts and any(texts):
            try:
                vectorizer = TfidfVectorizer(
                    tokenizer=self._tokenize,
                    token_pattern=None,
                    lowercase=True,
                    stop_words="english",
                    min_df=1,
                    max_df=1.0,
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                cache["vectorizer"] = vectorizer
                cache["tfidf_matrix"] = tfidf_matrix
            except Exception:
                cache["vectorizer"] = None
                cache["tfidf_matrix"] = None

        # BM25
        if _HAS_BM25 and corpus:
            try:
                cache["bm25"] = BM25Okapi(corpus)
            except Exception:
                cache["bm25"] = None

        return cache

    def _semantic_search_cached(
        self,
        query: str,
        limit: int,
        cache: dict[str, Any],
    ) -> list[dict]:
        """Rank experiences using cached TF-IDF + cosine similarity."""
        vectorizer = cache.get("vectorizer")
        tfidf_matrix = cache.get("tfidf_matrix")
        experiences = cache["experiences"]

        if vectorizer is None or tfidf_matrix is None:
            return self._keyword_search(
                experiences, self._tokenize(query), limit
            )

        try:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        except Exception:
            return self._keyword_search(
                experiences, self._tokenize(query), limit
            )

        scored = [
            (sim, exp)
            for sim, exp in zip(similarities, experiences, strict=False)
            if sim > 0
        ]
        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def _bm25_search_cached(
        self,
        query_tokens: list[str],
        limit: int,
        cache: dict[str, Any],
    ) -> list[dict]:
        """Rank experiences using cached BM25Okapi."""
        bm25 = cache.get("bm25")
        experiences = cache["experiences"]

        if bm25 is None:
            return self._keyword_search(experiences, query_tokens, limit)

        scores = bm25.get_scores(query_tokens)
        scored = [
            (score, exp)
            for score, exp in zip(scores, experiences, strict=False)
            if score > 0
        ]

        if not scored:
            return self._keyword_search(experiences, query_tokens, limit)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def find_similar_experiences(
        self,
        instruction: str,
        limit: int = 5,
        outcome_filter: str | None = None,
    ) -> list[dict]:
        """
        Find past experiences similar to the given instruction.

        Search priority (best available wins):
          1. Query-result cache (same query within 5s)
          2. TF-IDF + cosine similarity (semantic, via sklearn) — cached
          3. BM25Okapi ranking (statistical, via rank_bm25) — cached
          4. Keyword set intersection (fallback, always works)

        Tiered retrieval: if the first 200 experiences don't yield enough
        matches, automatically expands to 400 then 600 records.
        """
        filters = {"robot_id": self._robot_id}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        # 1) Check query-result cache
        cache_key = f"{instruction}|{outcome_filter}|{limit}"
        now = time.time()
        cached_result = self._query_result_cache.get(cache_key)
        if cached_result is not None:
            results, ts = cached_result
            if now - ts < self._query_cache_ttl_sec:
                return list(results)

        query_tokens = self._tokenize(instruction)
        if not query_tokens:
            return []

        # 2) Tiered retrieval: fast path 200, expand to 600 only if needed
        all_experiences = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=200,
        )

        if not all_experiences:
            return []

        # Build cache once (max 600 experiences to avoid repeated rebuilds)
        with self._search_cache_lock:
            cache = self._search_cache
            if (
                cache.get("version") != self._cache_version
                or cache.get("filters") != filters
                or len(cache.get("experiences", [])) < 200
            ):
                cache = self._build_search_cache(all_experiences, filters)
                self._search_cache = cache

        # Try semantic search on first 200
        if _HAS_SKLEARN:
            results = self._semantic_search_cached(instruction, limit, cache)
        elif _HAS_BM25:
            results = self._bm25_search_cached(query_tokens, limit, cache)
        else:
            results = self._keyword_search(all_experiences, query_tokens, limit)

        if len(results) >= limit:
            self._query_result_cache[cache_key] = (results, now)
            return results

        # Expand to 600 if first 200 didn't yield enough matches
        expanded = self._client.query(
            "experience_graph",
            filters=filters,
            order_by="-timestamp",
            limit=600,
        )
        if len(expanded) > len(all_experiences):
            all_experiences = expanded
            with self._search_cache_lock:
                cache = self._build_search_cache(all_experiences, filters)
                self._search_cache = cache
            if _HAS_SKLEARN:
                results = self._semantic_search_cached(instruction, limit, cache)
            elif _HAS_BM25:
                results = self._bm25_search_cached(query_tokens, limit, cache)
            else:
                results = self._keyword_search(all_experiences, query_tokens, limit)

        self._query_result_cache[cache_key] = (results, now)
        return results

    def _semantic_search(
        self,
        experiences: list[dict],
        query: str,
        limit: int,
    ) -> list[dict]:
        """Rank experiences using TF-IDF + cosine similarity.

        This provides semantic-level matching (e.g. "grasp cup" will match
        "pick up the red mug") without requiring heavy embeddings.
        """
        texts: list[str] = []
        for exp in experiences:
            parts = [
                exp.get("instruction", ""),
                " ".join(exp.get("tags", [])),
                exp.get("error_details", ""),
            ]
            texts.append(" ".join(p for p in parts if p))

        if not any(texts):
            return []

        try:
            vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize,
                lowercase=True,
                stop_words="english",
                min_df=1,
                max_df=1.0,
            )
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            similarities = cosine_similarity(
                tfidf_matrix[-1:], tfidf_matrix[:-1]
            )[0]
        except Exception:
            # TF-IDF can fail on very small corpora — fall back to keywords
            return self._keyword_search(experiences, self._tokenize(query), limit)

        scored = [
            (sim, exp)
            for sim, exp in zip(similarities, experiences, strict=False)
            if sim > 0
        ]
        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

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

        scored = [
            (score, exp)
            for score, exp in zip(scores, experiences, strict=False)
            if score > 0
        ]

        if not scored:
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

    def find_analogy(self, error_log: str, limit: int = 3) -> dict | None:
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

    def get_experience(self, experience_id: str) -> dict | None:
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
        result = self._client.delete("experience_graph", experience_id)
        if result:
            self._invalidate_search_cache()
        return result

    def forget_old_experiences(
        self,
        max_age_days: int | None = None,
        outcome_filter: str | None = None,
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
            logger.info("Forgot %d experiences (older than %s days)", deleted, max_age_days)
            self._invalidate_search_cache()
        return deleted

    def enforce_capacity(self, max_experiences: int | None = None) -> int:
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
            logger.info("Evicted %d experiences (capacity: %s)", evicted, max_experiences)
            self._invalidate_search_cache()
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
        entity_id: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
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
        condition: str | None = None,
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
        record = self._enrich_record_with_body_sense(record)
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
        record = self._enrich_record_with_body_sense(record)
        return self._client.insert("failures", record)

    def ingest_episode(
        self,
        episode_id: str,
        data_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Ingest a recorded practice episode into memory.

        Reads the episode.json, raw events, and provider result from disk and
        stores a summary experience plus artifact records in SeekDB.
        """
        from rosclaw.practice.storage.layout import PracticeLayout

        root = Path(data_root or "/data/rosclaw/practice")
        layout = PracticeLayout(root)
        session_dir = layout.session_dir(episode_id)
        if not session_dir.exists():
            return {"status": "error", "reason": f"session not found: {session_dir}"}

        episode_path = session_dir / "episode.json"
        events_path = layout.events_jsonl_path(episode_id)
        provider_path = session_dir / "provider" / "provider_result.json"

        episode: dict[str, Any] = {}
        if episode_path.exists():
            try:
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"status": "error", "reason": f"failed to parse episode.json: {exc}"}

        events: list[dict[str, Any]] = []
        if events_path.exists():
            try:
                with open(events_path, encoding="utf-8") as f:
                    events = [json.loads(line) for line in f if line.strip()]
            except Exception as exc:
                return {"status": "error", "reason": f"failed to parse events.jsonl: {exc}"}

        provider_data: dict[str, Any] | None = None
        if provider_path.exists():
            try:
                provider_data = json.loads(provider_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse provider result for %s: %s", episode_id, exc)

        task = episode.get("task", {})
        instruction = (
            f"Practice episode {episode_id}: "
            f"{task.get('task_id') or task.get('task_name') or 'unknown'}"
        )
        outcome = str(episode.get("outcome", "unknown")).lower()
        duration_ms = episode.get("duration_ms") or 0
        duration_sec = duration_ms / 1000.0 if isinstance(duration_ms, (int, float)) else 0.0

        record_id = self.store_experience(
            event_id=episode_id,
            event_type="practice_episode",
            instruction=instruction,
            outcome=outcome,
            duration_sec=duration_sec,
            tags=[
                "practice",
                str(episode.get("robot_type") or "unknown"),
                str(episode.get("robot_id") or "unknown"),
            ],
            metadata={
                "episode": episode,
                "events": events,
                "provider": provider_data,
            },
        )

        # Index the RGB frame artifact for retrieval.
        for ev in events:
            payload = ev.get("payload", {})
            rgb_ref = payload.get("rgb_ref")
            if rgb_ref:
                artifact_id = f"{episode_id}_{ev.get('event_type', 'frame')}"
                self._client.insert(
                    "artifacts",
                    {
                        "id": artifact_id,
                        "episode_id": episode_id,
                        "artifact_type": "rgb_frame",
                        "uri": str(rgb_ref),
                        "created_at": time.time(),
                        "metadata": ev,
                    },
                )

        return {
            "status": "success",
            "experience_id": record_id,
            "event_count": len(events),
            "outcome": outcome,
        }

    def retrieve_similar_episode(
        self,
        task_id: str | None = None,
        robot_id: str | None = None,
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
        task_id: str | None = None,
    ) -> dict | None:
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
        robot_id: str | None = None,
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
        robot_id: str | None = None,
    ) -> dict | None:
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
        robot_id: str | None = None,
        constraint_type: str | None = None,
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

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
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
        artifact_type: str | None = None,
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

    def add_world_object(self, obj: WorldObjectLike) -> str | None:
        """Add a world object. Returns obj_id or None if no EmbodiedMemory."""
        if self._embodied is None:
            return None
        return self._embodied.add_world_object(obj)

    def get_world_object(self, obj_id: str) -> WorldObjectLike | None:
        """Get a world object by ID."""
        if self._embodied is None:
            return None
        return self._embodied.get_world_object(obj_id)

    def update_world_object_pose(
        self, obj_id: str, pose: PoseLike, state: str | None = None
    ) -> bool:
        """Update world object pose and optional state."""
        if self._embodied is None:
            return False
        return self._embodied.update_world_object_pose(obj_id, pose, state)

    def search_world_objects(
        self,
        center: Vec3Like,
        radius: float,
        scene_id: str | None = None,
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
    ) -> PermanenceReportLike | None:
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
    ) -> int | None:
        """Record a trajectory. Returns memory_id or None."""
        if self._embodied is None:
            return None
        return self._embodied.record_trajectory(content, waypoints)

    def search_similar_trajectories(
        self,
        query_waypoints: list[tuple[Vec3Like, float]],
        top_k: int = 5,
        max_dtw_distance: float | None = None,
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
        spatial_center: Vec3Like | None = None,
        spatial_radius: float = 2.0,
        temporal_interval: TemporalIntervalLike | None = None,
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

    def run_meditation(self, phases: list[str] | None = None) -> dict:
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
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
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
        condition_filter: str | None = None,
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

    def get_rule(self, rule_id: str) -> dict | None:
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
