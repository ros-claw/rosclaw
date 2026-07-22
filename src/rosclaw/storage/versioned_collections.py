"""Versioned SeekDB collections + projection registry (数据库优化v3 §8.3).

Existing collections are never modified in place.  A new multilingual
index is built as a NEW physical collection with manual embeddings
(``embedding_function=None``), registered in ``projection_registry``,
verified, shadow-queried, benchmarked, and only then atomically
ACTIVATED.  The old collection is kept (status OLD) so rollback is a
registry flip, not a rebuild.

Switch flow::

    create -> backfill -> verify -> shadow query -> benchmark
    -> activate (ACTIVE) -> keep old (OLD) -> observe -> (optional) drop
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from rosclaw.embedding.protocol import EmbeddingProfile, EmbeddingProvider

REGISTRY_TABLE = "projection_registry"

# Registry row states (数据库优化v3 §8.3)
BUILDING = "BUILDING"
READY = "READY"
ACTIVE = "ACTIVE"
OLD = "OLD"
FAILED = "FAILED"


def physical_name(logical_name: str, profile: EmbeddingProfile, analyzer: str) -> str:
    """Deterministic physical collection name (never shared across
    models or dimensions — §17.5)."""
    return f"{logical_name}__{profile.profile_id}__{analyzer}"


def corpus_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: str(r.get("id"))):
        digest.update(str(record.get("id")).encode())
        digest.update(str(record.get("document") or record.get("title") or "").encode())
    return digest.hexdigest()


def _exact_row_multiplier(row: dict[str, Any], exact: dict[str, list[str]]) -> float:
    """Dict-row mirror of the §6.3 exact-entity multiplier (memory
    metadata rows carry body_id/joint_name/failure_type/gesture_name at
    top level after §2.2 flattening)."""
    if not exact:
        return 1.0
    mult = 1.0
    joints = exact.get("joints") or []
    if len(joints) == 1 and row.get("memory_type") == "failure":
        wanted = joints[0]
        joint = row.get("joint_name")
        if joint == wanted:
            mult *= 1.5
        elif joint is not None and joint != wanted:
            mult *= 0.25
    failure_types = exact.get("failure_types") or []
    if len(failure_types) == 1 and row.get("memory_type") == "failure":
        wanted = failure_types[0]
        ftype = row.get("failure_type")
        if ftype == wanted:
            mult *= 1.4
        elif ftype is not None and ftype != wanted:
            mult *= 0.4
    hands = exact.get("hands") or []
    if len(hands) == 1:
        wanted_body = f"rh56_{hands[0]}_01"
        body = row.get("body_id")
        if body == wanted_body:
            mult *= 1.3
        elif body in ("rh56_left_01", "rh56_right_01") and body != wanted_body:
            mult *= 0.3
    return mult


class VersionedCollectionManager:
    """Builds and switches versioned multilingual collections on a
    SeekDBNativeStore (embedded or server)."""

    def __init__(self, store: Any, provider: EmbeddingProvider) -> None:
        self._store = store
        self._provider = provider

    # ------------------------------------------------------------------
    # Registry rows
    # ------------------------------------------------------------------
    def _register(self, row: dict[str, Any]) -> None:
        self._store.insert(REGISTRY_TABLE, row)

    def _rows(self, logical_name: str) -> list[dict[str, Any]]:
        return self._store.query(REGISTRY_TABLE, filters={"logical_name": logical_name}, limit=100)

    def registry(self, logical_name: str) -> list[dict[str, Any]]:
        return self._rows(logical_name)

    def active(self, logical_name: str) -> dict[str, Any] | None:
        for row in self._rows(logical_name):
            if row.get("status") == ACTIVE:
                return row
        return None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        logical_name: str,
        records: list[dict[str, Any]],
        *,
        analyzer: str = "ngram",
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Create the physical collection and backfill it with manual
        embeddings.  Records need ``id`` plus text fields (title/document)."""
        profile = self._provider.profile
        name = physical_name(logical_name, profile, analyzer)
        client = self._store._client
        if client is None:
            raise RuntimeError("store is not connected")
        import pyseekdb

        fulltext = pyseekdb.FulltextIndexConfig(analyzer=analyzer)
        # NOTE: pyseekdb's newer Schema path still forces the built-in
        # 384-dim default embedder (dimension validation fails against a
        # manual-embedding profile); the legacy Configuration form +
        # embedding_function=None is the working no-embedder shape on
        # this pyseekdb version (probed on the real engine).
        config = pyseekdb.Configuration(
            hnsw=pyseekdb.HNSWConfiguration(
                dimension=profile.dimension,
                distance=profile.distance,
            ),
            fulltext_config=fulltext,
        )
        if client.has_collection(name):
            client.delete_collection(name)
        collection = client.create_collection(
            name=name,
            configuration=config,
            embedding_function=None,
        )
        row = {
            "id": f"proj_{logical_name}_{profile.profile_id}_{analyzer}",
            "logical_name": logical_name,
            "physical_collection": name,
            "embedding_profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "dimension": profile.dimension,
            "analyzer": analyzer,
            "corpus_hash": corpus_hash(records),
            "record_count": 0,
            "status": BUILDING,
            "created_at": time.time(),
            "activated_at": None,
        }
        self._register(row)
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                ids = [str(r["id"]) for r in batch]
                documents = [str(r.get("document") or r.get("title") or r["id"]) for r in batch]
                embeddings = self._provider.encode_documents(documents)
                metadatas = [self._store._metadata(r) for r in batch]
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            collection.refresh_index()
            row["record_count"] = len(records)
            row["status"] = READY
            self._register(row)
        except Exception:
            row["status"] = FAILED
            self._register(row)
            raise
        return row

    # ------------------------------------------------------------------
    # Verify / shadow / activate / rollback
    # ------------------------------------------------------------------
    def verify(self, logical_name: str, *, analyzer: str) -> dict[str, Any]:
        profile = self._provider.profile
        name = physical_name(logical_name, profile, analyzer)
        rows = [r for r in self._rows(logical_name) if r.get("physical_collection") == name]
        if not rows:
            return {"ok": False, "reason": "no registry row"}
        row = rows[-1]
        actual = self._store.count(name)
        ok = actual == int(row.get("record_count") or 0)
        return {
            "ok": ok,
            "expected": row.get("record_count"),
            "actual": actual,
            "dimension": self._store.embedding_info(name).get("dimension"),
            "profile_dimension": profile.dimension,
        }

    def shadow_query(
        self,
        logical_name: str,
        query_text: str,
        *,
        analyzer: str,
        filters: dict | None = None,
        limit: int = 5,
        candidate_window: int = 20,
        exact_boost: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the NOT-yet-active collection with the profile's query
        embedding (query side instruction included — §7.2).

        ``exact_boost`` applies the §6.3 exact-entity reorder to the
        returned rows: a query naming 左手/left hand never surfaces a
        right-body memory on top just because the text is similar
        (middle vs thumb_rot, left vs right are hard entities, not
        synonyms)."""
        vector = self._provider.encode_queries([query_text])[0]
        name = physical_name(logical_name, self._provider.profile, analyzer)
        from rosclaw.memory.v2.document import extract_exact_terms

        exact = extract_exact_terms(query_text) if exact_boost else {}
        hands = (exact.get("hands") or []) if exact else []
        hard_filters = dict(filters or {})
        constrained = False
        if exact_boost and len(hands) == 1 and "body_id" not in hard_filters:
            # v3 §9 flow: Metadata HARD filter first when the query names
            # an unambiguous body — otherwise the 20-row shortlist can be
            # fully starved of the requested body (measured: 左手 query
            # returned 20/20 right-body rows on the qwen3 index).
            hard_filters["body_id"] = f"rh56_{hands[0]}_01"
            constrained = True
        rows = self._store.hybrid_search(
            name,
            query_text,
            filters=hard_filters or None,
            limit=max(limit, candidate_window),
            candidate_window=candidate_window,
            query_embedding=vector,
        )
        if constrained and not rows:
            # Honest fallback: no memory for that body at all — retry
            # unfiltered and let the demote reorder speak.
            rows = self._store.hybrid_search(
                name,
                query_text,
                filters=filters,
                limit=max(limit, candidate_window),
                candidate_window=candidate_window,
                query_embedding=vector,
            )
        if not exact_boost or not exact:
            return rows[:limit]
        ranked = sorted(
            enumerate(rows),
            key=lambda pair: (_exact_row_multiplier(pair[1], exact), -pair[0]),
            reverse=True,
        )
        return [row for _, row in ranked[:limit]]

    def activate(self, logical_name: str, *, analyzer: str) -> dict[str, Any]:
        """Atomically flip: current ACTIVE -> OLD, target -> ACTIVE."""
        profile = self._provider.profile
        name = physical_name(logical_name, profile, analyzer)
        now = time.time()
        target: dict[str, Any] | None = None
        for row in self._rows(logical_name):
            updated = dict(row)
            if row.get("status") == ACTIVE:
                updated["status"] = OLD
                self._register(updated)
            if row.get("physical_collection") == name and row.get("status") in (READY, OLD):
                target = dict(row)
        if target is None:
            raise RuntimeError(f"no READY/OLD build of {name} to activate (run build+verify first)")
        target["status"] = ACTIVE
        target["activated_at"] = now
        self._register(target)
        return target

    def rollback(self, logical_name: str) -> dict[str, Any]:
        """Flip back: current ACTIVE -> OLD, newest OLD -> ACTIVE."""
        rows = self._rows(logical_name)
        current = next((r for r in rows if r.get("status") == ACTIVE), None)
        olds = [r for r in rows if r.get("status") == OLD]
        if not olds:
            raise RuntimeError(f"no OLD build of {logical_name} to roll back to")
        target = max(olds, key=lambda r: float(r.get("activated_at") or 0.0))
        if current is not None:
            demoted = dict(current)
            demoted["status"] = OLD
            self._register(demoted)
        restored = dict(target)
        restored["status"] = ACTIVE
        restored["activated_at"] = time.time()
        self._register(restored)
        return restored

    # ------------------------------------------------------------------
    # Truth reporting (数据库优化v3 §13 truth tests)
    # ------------------------------------------------------------------
    def describe(self, logical_name: str) -> dict[str, Any]:
        """Everything a CLI must be able to state about the index:
        backend, collection, model, revision, dimension, analyzer,
        vector source, score semantics, reranker, fallback state."""
        active = self.active(logical_name)
        profile = self._provider.profile
        return {
            "backend": type(self._store).__name__,
            "logical_name": logical_name,
            "active_collection": (active or {}).get("physical_collection"),
            "embedding": {
                "profile_id": profile.profile_id,
                "model_id": profile.model_id,
                "model_revision": profile.model_revision,
                "dimension": profile.dimension,
                "normalize": profile.normalize,
                "distance": profile.distance,
                "query_instruction": bool(profile.query_instruction),
            },
            "analyzer": (active or {}).get("analyzer"),
            "vector_source": "manual_query_embedding" if active else None,
            "score_semantics": (
                "cosine on manual query/document embeddings + BM25, fused by "
                "RRF (rank_constant=60); exact-entity multipliers applied "
                "post-fusion; scores are NOT interchangeable similarities"
            ),
            "reranker": None,
            "fallback_state": "bm25+metadata when embedding provider unavailable",
            "registry": [
                {
                    "physical_collection": r.get("physical_collection"),
                    "status": r.get("status"),
                    "record_count": r.get("record_count"),
                    "activated_at": r.get("activated_at"),
                }
                for r in self._rows(logical_name)
            ],
        }
