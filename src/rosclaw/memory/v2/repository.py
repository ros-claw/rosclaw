"""Memory 2.0 repository — typed memory CRUD with evidence and idempotent writes."""

from __future__ import annotations

import logging
import time
from typing import Any

from rosclaw.memory.v2.models import (
    MemoryEvidence,
    MemoryItem,
    MemoryStatus,
)

logger = logging.getLogger("rosclaw.memory.v2.repository")

ITEMS_TABLE = "memory_items"
EVIDENCE_TABLE = "memory_evidence"


class MemoryRepository:
    """Stores :class:`MemoryItem` rows plus their :class:`MemoryEvidence` rows.

    Writes are idempotent on ``memory_id`` (upsert) and deduplicated on
    ``content_hash`` — re-distilling the same practice session never creates
    duplicate memories.  Every stored memory carries at least one evidence
    row so it can be traced back to practice events, episodes, MCAP,
    telemetry windows, human feedback, or critic results.
    """

    def __init__(self, client: Any):
        self._client = client

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def store(self, item: MemoryItem, evidence: list[MemoryEvidence] | None = None) -> str:
        """Persist *item* and its evidence; returns the memory_id.

        Idempotency rules:

        * Same ``memory_id`` — upsert (refresh ``updated_at`` and fields).
        * Same ``content_hash`` already active — no new row; the existing
          memory_id is returned so re-distillation stays a no-op.
        """
        existing = self.find_by_content_hash(item.content_hash, robot_id=item.robot_id)
        if existing is not None and existing.memory_id != item.memory_id:
            logger.debug(
                "Dedup by content_hash: %s already stored as %s",
                item.memory_id,
                existing.memory_id,
            )
            return existing.memory_id

        item.updated_at = time.time()
        self._client.insert(ITEMS_TABLE, item.to_record())
        rows = evidence if evidence is not None else self._evidence_from_refs(item)
        for ev in rows:
            ev.memory_id = item.memory_id
            self._client.insert(EVIDENCE_TABLE, ev.to_record())
        return item.memory_id

    def merge_into(self, target_id: str, item: MemoryItem) -> bool:
        """Fold *item* into the existing memory *target_id* (MERGE decision).

        Appends evidence refs, keeps the higher confidence/importance, and
        refreshes ``updated_at``.  The target's document/title are preserved —
        a merge never silently rewrites the curated record.
        """
        target = self.get(target_id)
        if target is None:
            return False
        merged_refs = sorted(set(target.evidence_refs) | set(item.evidence_refs))
        merged_artifacts = sorted(set(target.artifact_refs) | set(item.artifact_refs))
        merged_tags = sorted(set(target.tags) | set(item.tags))
        self._client.update(
            ITEMS_TABLE,
            target_id,
            {
                "evidence_refs": _json_list(merged_refs),
                "artifact_refs": _json_list(merged_artifacts),
                "tags": _json_list(merged_tags),
                "confidence": max(target.confidence, item.confidence),
                "importance": max(target.importance, item.importance),
                "updated_at": time.time(),
            },
        )
        for ev in self._evidence_from_refs(item):
            ev.memory_id = target_id
            self._client.insert(EVIDENCE_TABLE, ev.to_record())
        return True

    def supersede(self, old_id: str, new_item: MemoryItem) -> str:
        """Mark *old_id* superseded and store *new_item* (UPDATE decision).

        The old memory's evidence chain is carried forward into the new item
        so the active record always holds the union of all evidence seen for
        this content — otherwise redistilling an older session would look
        like it brings "new" evidence and trigger endless update cycles.
        """
        old = self.get(old_id)
        if old is not None:
            new_item.evidence_refs = sorted(set(new_item.evidence_refs) | set(old.evidence_refs))
            new_item.artifact_refs = sorted(set(new_item.artifact_refs) | set(old.artifact_refs))
        self.mark_status(old_id, MemoryStatus.SUPERSEDED.value)
        new_item.metadata = {**new_item.metadata, "supersedes": old_id}
        return self.store(new_item)

    def mark_status(self, memory_id: str, status: str) -> bool:
        return bool(
            self._client.update(
                ITEMS_TABLE, memory_id, {"status": status, "updated_at": time.time()}
            )
        )

    def pin(self, memory_id: str, pinned: bool = True) -> bool:
        """Safety/human-approved pinning: pinned memories never decay or expire."""
        return bool(
            self._client.update(
                ITEMS_TABLE,
                memory_id,
                {"pinned": 1 if pinned else 0, "updated_at": time.time()},
            )
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> MemoryItem | None:
        rows = self._client.query(ITEMS_TABLE, filters={"id": memory_id}, limit=1)
        return MemoryItem.from_record(rows[0]) if rows else None

    def find_by_content_hash(
        self, content_hash: str, *, robot_id: str | None = None
    ) -> MemoryItem | None:
        filters: dict[str, Any] = {
            "content_hash": content_hash,
            "status": MemoryStatus.ACTIVE.value,
        }
        if robot_id is not None:
            filters["robot_id"] = robot_id
        rows = self._client.query(ITEMS_TABLE, filters=filters, limit=1)
        return MemoryItem.from_record(rows[0]) if rows else None

    def query(
        self,
        filters: dict[str, Any] | None = None,
        *,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[MemoryItem]:
        filters = dict(filters or {})
        if not include_inactive:
            filters.setdefault("status", MemoryStatus.ACTIVE.value)
        rows = self._client.query(ITEMS_TABLE, filters=filters, limit=limit)
        return [MemoryItem.from_record(row) for row in rows]

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return self._client.count(ITEMS_TABLE, filters)

    def evidence_for(self, memory_id: str) -> list[MemoryEvidence]:
        rows = self._client.query(EVIDENCE_TABLE, filters={"memory_id": memory_id}, limit=1000)
        return [MemoryEvidence.from_record(row) for row in rows]

    def trace(self, memory_id: str) -> dict[str, Any]:
        """Return a memory with its full evidence chain for verification."""
        item = self.get(memory_id)
        if item is None:
            return {"found": False, "memory_id": memory_id}
        evidence = self.evidence_for(memory_id)
        return {
            "found": True,
            "memory": item.to_record(),
            "evidence": [ev.to_record() for ev in evidence],
            "evidence_count": len(evidence),
            "traceable": len(evidence) > 0,
        }

    # ------------------------------------------------------------------
    # Migration from experience_graph (§5.9)
    # ------------------------------------------------------------------

    def migrate_experience_graph(self, *, limit: int = 10_000) -> dict[str, int]:
        """Migrate legacy ``experience_graph`` rows into typed memory items.

        Idempotent via content-hash dedup: rerunning the migration never
        duplicates.  Outcome mapping: failure → failure memory, everything
        else → episodic memory.
        """
        from rosclaw.memory.v2.models import MemoryType

        rows = self._client.query("experience_graph", limit=limit)
        stats = {"scanned": len(rows), "migrated": 0, "deduplicated": 0, "skipped": 0}
        for row in rows:
            try:
                outcome = (row.get("outcome") or "").lower()
                memory_type = (
                    MemoryType.FAILURE.value
                    if outcome in {"failure", "emergency"}
                    else MemoryType.EPISODIC.value
                )
                instruction = row.get("instruction") or row.get("event_type") or "experience"
                document_parts = [instruction]
                if row.get("error_details"):
                    document_parts.append(f"error: {row['error_details']}")
                item = MemoryItem(
                    memory_type=memory_type,
                    robot_id=row.get("robot_id", "unknown"),
                    title=instruction[:120],
                    document="\n".join(document_parts),
                    outcome=outcome or None,
                    event_time=float(row.get("timestamp") or time.time()),
                    evidence_refs=[row.get("id", "")] if row.get("id") else [],
                    tags=["migrated", "experience_graph"],
                    metadata={"legacy_id": row.get("id"), "event_type": row.get("event_type")},
                )
                existing = self.find_by_content_hash(item.content_hash, robot_id=item.robot_id)
                if existing is not None:
                    stats["deduplicated"] += 1
                    continue
                self.store(item)
                stats["migrated"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to migrate experience %s: %s", row.get("id"), exc)
                stats["skipped"] += 1
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evidence_from_refs(self, item: MemoryItem) -> list[MemoryEvidence]:
        """Build default practice-event evidence rows from item refs."""
        evidence: list[MemoryEvidence] = []
        for ref in item.evidence_refs:
            if not ref:
                continue
            evidence.append(
                MemoryEvidence(
                    memory_id=item.memory_id,
                    evidence_type="practice_event",
                    source_event_id=ref,
                    confidence=item.confidence,
                )
            )
        for uri in item.artifact_refs:
            if not uri:
                continue
            evidence.append(
                MemoryEvidence(
                    memory_id=item.memory_id,
                    evidence_type="mcap" if uri.endswith(".mcap") else "frame",
                    artifact_uri=uri,
                    confidence=item.confidence,
                )
            )
        return evidence


def _json_list(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)
