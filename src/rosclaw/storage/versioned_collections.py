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
import logging
import time
import uuid
from typing import Any

from rosclaw.embedding.errors import EmbeddingUnavailableError
from rosclaw.embedding.protocol import EmbeddingProfile, EmbeddingProvider

REGISTRY_TABLE = "projection_registry"
_REGISTRY_LIMIT = 10_000
_POINTER_KIND = "active_pointer"
logger = logging.getLogger("rosclaw.storage.versioned_collections")

# Registry row states (数据库优化v3 §8.3)
BUILDING = "BUILDING"
READY = "READY"
ACTIVE = "ACTIVE"
OLD = "OLD"
FAILED = "FAILED"


def physical_name(
    logical_name: str,
    profile: EmbeddingProfile,
    analyzer: str,
    generation: str | None = None,
) -> str:
    """Deterministic physical collection name (never shared across
    models or dimensions — §17.5).

    A build adds a unique generation suffix so rebuilding the same profile
    can never delete or mutate an ACTIVE collection in place.
    """
    base = f"{logical_name}__{profile.profile_id}__{analyzer}"
    return f"{base}__g{generation}" if generation else base


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
        rows = self._store.query(
            REGISTRY_TABLE,
            filters={"logical_name": logical_name},
            limit=_REGISTRY_LIMIT,
        )
        builds = [row for row in rows if row.get("row_kind") != _POINTER_KIND]
        return sorted(
            builds,
            key=lambda row: (float(row.get("created_at") or 0.0), str(row.get("id") or "")),
        )

    @staticmethod
    def _pointer_id(logical_name: str) -> str:
        digest = hashlib.sha256(logical_name.encode()).hexdigest()[:24]
        return f"projection_active_{digest}"

    def _pointer(self, logical_name: str) -> dict[str, Any] | None:
        rows = self._store.query(
            REGISTRY_TABLE,
            filters={"id": self._pointer_id(logical_name)},
            limit=1,
        )
        return rows[0] if rows else None

    def _row_by_collection(
        self, logical_name: str, physical_collection: str | None
    ) -> dict[str, Any] | None:
        if not physical_collection:
            return None
        return next(
            (
                row
                for row in self._rows(logical_name)
                if row.get("physical_collection") == physical_collection
            ),
            None,
        )

    def registry(self, logical_name: str) -> list[dict[str, Any]]:
        pointer = self._pointer(logical_name)
        active_name = (pointer or {}).get("active_physical_collection")
        previous_name = (pointer or {}).get("previous_physical_collection")
        effective: list[dict[str, Any]] = []
        for raw in self._rows(logical_name):
            row = dict(raw)
            name = row.get("physical_collection")
            if name == active_name:
                row["status"] = ACTIVE
                row["activated_at"] = (pointer or {}).get("active_activated_at")
            elif row.get("status") not in (BUILDING, FAILED):
                row["status"] = OLD if name == previous_name or row.get("activated_at") else READY
                if name == previous_name:
                    row["activated_at"] = (pointer or {}).get("previous_activated_at")
            effective.append(row)
        return effective

    def active(self, logical_name: str) -> dict[str, Any] | None:
        pointer = self._pointer(logical_name)
        if pointer is not None:
            row = self._row_by_collection(
                logical_name,
                pointer.get("active_physical_collection"),
            )
            if row is None:
                return None
            active = dict(row)
            active["status"] = ACTIVE
            active["activated_at"] = pointer.get("active_activated_at")
            return active

        # Backward compatibility for registries written before the canonical
        # pointer existed.  The next activation/rollback migrates them.
        legacy = [row for row in self._rows(logical_name) if row.get("status") == ACTIVE]
        return (
            max(legacy, key=lambda row: float(row.get("activated_at") or 0.0)) if legacy else None
        )

    def _select_build(
        self,
        logical_name: str,
        *,
        analyzer: str,
        allowed_statuses: set[str],
    ) -> dict[str, Any] | None:
        profile_id = self._provider.profile.profile_id
        rows = [
            row
            for row in self.registry(logical_name)
            if row.get("embedding_profile_id") == profile_id
            and row.get("analyzer") == analyzer
            and row.get("status") in allowed_statuses
        ]
        return max(rows, key=lambda row: float(row.get("created_at") or 0.0)) if rows else None

    def _mark_build_status(
        self,
        row: dict[str, Any] | None,
        status: str,
        activated_at: float | None,
    ) -> None:
        """Persist audit status after the canonical pointer has switched.

        The pointer is authoritative.  A failed secondary audit write is
        logged but cannot make the already-completed switch fail or disappear.
        """
        if row is None:
            return
        updated = dict(row)
        updated["status"] = status
        updated["activated_at"] = activated_at
        try:
            self._register(updated)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "active pointer switched but registry audit status update failed for %s: %s",
                row.get("physical_collection"),
                exc,
            )

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
        if not records:
            raise ValueError("refusing to build an empty versioned collection")
        profile = self._provider.profile
        build_id = uuid.uuid4().hex[:10]
        name = physical_name(logical_name, profile, analyzer, build_id)
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
        row = {
            "id": f"projection_build_{uuid.uuid4().hex}",
            "row_kind": "build",
            "build_id": build_id,
            "logical_name": logical_name,
            "physical_collection": name,
            "embedding_profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "dimension": profile.dimension,
            "normalize": profile.normalize,
            "distance": profile.distance,
            "query_instruction_enabled": bool(profile.query_instruction),
            "document_instruction_enabled": bool(profile.document_instruction),
            "provider_type": profile.provider_type,
            "analyzer": analyzer,
            "corpus_hash": corpus_hash(records),
            "record_count": 0,
            "status": BUILDING,
            "created_at": time.time(),
            "activated_at": None,
        }
        self._register(row)
        try:
            if client.has_collection(name):  # UUID collision or stale external object
                raise RuntimeError(f"refusing to overwrite existing physical collection {name}")
            collection = client.create_collection(
                name=name,
                configuration=config,
                embedding_function=None,
            )
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
            try:
                self._register(row)
            except Exception as audit_exc:  # noqa: BLE001
                logger.warning(
                    "versioned collection build failed and FAILED audit write also failed for %s: %s",
                    name,
                    audit_exc,
                )
            raise
        return row

    # ------------------------------------------------------------------
    # Verify / shadow / activate / rollback
    # ------------------------------------------------------------------
    def verify(self, logical_name: str, *, analyzer: str) -> dict[str, Any]:
        row = self._select_build(
            logical_name,
            analyzer=analyzer,
            allowed_statuses={READY, ACTIVE, OLD},
        )
        if row is None:
            return {"ok": False, "reason": "no registry row"}
        return self._verify_row(row)

    def _verify_row(
        self,
        row: dict[str, Any],
        *,
        require_provider_match: bool = True,
    ) -> dict[str, Any]:
        profile = self._provider.profile
        name = str(row["physical_collection"])
        client = self._store._client
        if client is None:
            raise RuntimeError("store is not connected")
        if not client.has_collection(name):
            return {
                "ok": False,
                "reason": "physical collection missing",
                "expected": row.get("record_count"),
                "actual": None,
                "dimension": None,
                "profile_dimension": profile.dimension,
                "physical_collection": name,
            }
        actual = self._store.count(name)
        actual_dimension = self._store.embedding_info(name).get("dimension")
        expected_dimension = int(row.get("dimension") or 0)
        provider_matches = (
            row.get("embedding_profile_id") == profile.profile_id
            and expected_dimension == profile.dimension
        )
        ok = (
            actual == int(row.get("record_count") or 0)
            and actual_dimension == expected_dimension
            and (provider_matches or not require_provider_match)
        )
        return {
            "ok": ok,
            "expected": row.get("record_count"),
            "actual": actual,
            "dimension": actual_dimension,
            "profile_dimension": profile.dimension,
            "provider_matches": provider_matches,
            "physical_collection": name,
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
        build = self._select_build(
            logical_name,
            analyzer=analyzer,
            allowed_statuses={READY, ACTIVE, OLD},
        )
        if build is None:
            raise RuntimeError(
                f"no READY/ACTIVE/OLD build for {logical_name!r}, "
                f"profile={self._provider.profile.profile_id!r}, analyzer={analyzer!r}"
            )
        name = str(build["physical_collection"])
        client = self._store._client
        if client is None:
            raise RuntimeError("store is not connected")
        if not client.has_collection(name):
            raise RuntimeError(f"registered physical collection is missing: {name}")
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
        vector: list[float] | None = None
        try:
            vector = self._provider.encode_queries([query_text])[0]
            rows = self._store.hybrid_search(
                name,
                query_text,
                filters=hard_filters or None,
                limit=max(limit, candidate_window),
                candidate_window=candidate_window,
                query_embedding=vector,
            )
        except EmbeddingUnavailableError:
            rows = self._store.fulltext_search(
                name,
                query_text,
                filters=hard_filters or None,
                limit=max(limit, candidate_window),
            )
        if constrained and not rows:
            # Honest fallback: no memory for that body at all — retry
            # unfiltered and let the demote reorder speak.
            if vector is None:
                rows = self._store.fulltext_search(
                    name,
                    query_text,
                    filters=filters,
                    limit=max(limit, candidate_window),
                )
            else:
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
        """Atomically switch the canonical active pointer to a verified build."""
        target = self._select_build(
            logical_name,
            analyzer=analyzer,
            allowed_statuses={READY},
        )
        if target is None:
            current = self.active(logical_name)
            if (
                current is not None
                and current.get("embedding_profile_id") == self._provider.profile.profile_id
                and current.get("analyzer") == analyzer
            ):
                target = current
        if target is None:
            target = self._select_build(
                logical_name,
                analyzer=analyzer,
                allowed_statuses={OLD},
            )
        if target is None:
            raise RuntimeError(
                f"no READY/OLD build for profile={self._provider.profile.profile_id}, "
                f"analyzer={analyzer} to activate (run build+verify first)"
            )
        if target.get("status") == ACTIVE:
            return target
        verification = self._verify_row(target)
        if not verification["ok"]:
            raise RuntimeError(f"refusing to activate unverified build: {verification}")

        now = time.time()
        current = self.active(logical_name)
        previous_name = (current or {}).get("physical_collection")
        previous_activated_at = (current or {}).get("activated_at")
        pointer = {
            "id": self._pointer_id(logical_name),
            "row_kind": _POINTER_KIND,
            "logical_name": logical_name,
            "active_physical_collection": target["physical_collection"],
            "active_activated_at": now,
            "previous_physical_collection": previous_name,
            "previous_activated_at": previous_activated_at,
            "updated_at": now,
        }
        self._register(pointer)
        self._mark_build_status(current, OLD, previous_activated_at)
        self._mark_build_status(target, ACTIVE, now)
        activated = dict(target)
        activated["status"] = ACTIVE
        activated["activated_at"] = now
        return activated

    def rollback(self, logical_name: str) -> dict[str, Any]:
        """Atomically swap the active and previous generation pointers."""
        pointer = self._pointer(logical_name)
        current = self.active(logical_name)
        if pointer is not None:
            previous_name = pointer.get("previous_physical_collection")
            target = self._row_by_collection(logical_name, previous_name)
        else:
            olds = [row for row in self.registry(logical_name) if row.get("status") == OLD]
            target = (
                max(olds, key=lambda row: float(row.get("activated_at") or 0.0)) if olds else None
            )
        if target is None:
            raise RuntimeError(f"no OLD build of {logical_name} to roll back to")
        verification = self._verify_row(target, require_provider_match=False)
        if not verification["ok"]:
            raise RuntimeError(f"refusing to roll back to unverified build: {verification}")
        now = time.time()
        self._register(
            {
                "id": self._pointer_id(logical_name),
                "row_kind": _POINTER_KIND,
                "logical_name": logical_name,
                "active_physical_collection": target["physical_collection"],
                "active_activated_at": now,
                "previous_physical_collection": (current or {}).get("physical_collection"),
                "previous_activated_at": (current or {}).get("activated_at"),
                "updated_at": now,
            }
        )
        self._mark_build_status(current, OLD, (current or {}).get("activated_at"))
        self._mark_build_status(target, ACTIVE, now)
        restored = dict(target)
        restored["status"] = ACTIVE
        restored["activated_at"] = now
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
        active_profile_id = (active or {}).get("embedding_profile_id")
        provider_matches_active = active is None or active_profile_id == profile.profile_id

        def active_value(key: str, provider_value: Any) -> Any:
            value = (active or {}).get(key)
            if value is not None:
                return value
            return provider_value if provider_matches_active else None

        return {
            "backend": type(self._store).__name__,
            "logical_name": logical_name,
            "active_collection": (active or {}).get("physical_collection"),
            "embedding": {
                "profile_id": active_profile_id or profile.profile_id,
                "model_id": active_value("model_id", profile.model_id),
                "model_revision": active_value("model_revision", profile.model_revision),
                "dimension": active_value("dimension", profile.dimension),
                "normalize": active_value("normalize", profile.normalize),
                "distance": active_value("distance", profile.distance),
                "query_instruction": active_value(
                    "query_instruction_enabled", bool(profile.query_instruction)
                ),
                "provider_type": active_value("provider_type", profile.provider_type),
            },
            "requested_provider_profile_id": profile.profile_id,
            "provider_matches_active": provider_matches_active,
            "runtime_query_integration": "shadow_only_not_general_memory_query",
            "analyzer": (active or {}).get("analyzer"),
            "vector_source": "manual_query_embedding" if active else None,
            "score_semantics": (
                "cosine on manual query/document embeddings + BM25; server uses native "
                "RRF and embedded pyseekdb 1.3.0 uses deterministic client RRF over two "
                "engine-filtered legs (rank_constant=60); exact-entity multipliers apply "
                "post-fusion; scores are NOT interchangeable similarities"
            ),
            "reranker": None,
            "fallback_state": "bm25+metadata when embedding provider unavailable",
            "registry": [
                {
                    "physical_collection": r.get("physical_collection"),
                    "embedding_profile_id": r.get("embedding_profile_id"),
                    "analyzer": r.get("analyzer"),
                    "status": r.get("status"),
                    "record_count": r.get("record_count"),
                    "activated_at": r.get("activated_at"),
                }
                for r in self.registry(logical_name)
            ],
        }
