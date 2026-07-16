"""Memory 2.0 consolidation — dedup, decay, supersession, TTL, pinning (§5.7)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.memory.v2.models import MemoryItem, MemoryStatus

logger = logging.getLogger("rosclaw.memory.v2.consolidate")


@dataclass
class ConsolidateResult:
    scanned: int = 0
    expired: int = 0
    superseded: int = 0
    decayed: int = 0
    pinned_kept: int = 0
    duplicates_marked: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """Background consolidation over the memory store.

    * **TTL** — memories past ``expires_at`` become ``expired`` (never pinned).
    * **Supersession** — exact ``content_hash`` duplicates keep the newest
      (highest ``event_time``); older rows become ``superseded``.
    * **Confidence decay** — importance decays exponentially with age;
      pinned and safety memories are exempt.
    """

    def __init__(
        self,
        repository: Any,
        *,
        decay_half_life_days: float = 30.0,
        min_importance: float = 0.05,
    ):
        self._repo = repository
        self._half_life_s = decay_half_life_days * 86400.0
        self._min_importance = min_importance

    def consolidate(self, *, robot_id: str | None = None, limit: int = 5000) -> ConsolidateResult:
        result = ConsolidateResult()
        filters: dict[str, Any] = {}
        if robot_id:
            filters["robot_id"] = robot_id
        items = self._repo.query(filters, include_inactive=False, limit=limit)
        result.scanned = len(items)

        now = time.time()
        by_hash: dict[str, list[MemoryItem]] = {}
        for item in items:
            # 1. TTL expiry (pinned memories are exempt).
            if item.expires_at and item.expires_at <= now and not item.pinned:
                self._repo.mark_status(item.memory_id, MemoryStatus.EXPIRED.value)
                result.expired += 1
                continue
            if item.pinned:
                result.pinned_kept += 1
            by_hash.setdefault(item.content_hash, []).append(item)

        # 2. Supersession within exact content-hash groups.
        for group in by_hash.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda item: item.event_time, reverse=True)
            for stale in group[1:]:
                self._repo.mark_status(stale.memory_id, MemoryStatus.SUPERSEDED.value)
                result.superseded += 1
                result.duplicates_marked.append(stale.memory_id)

        # 3. Confidence decay for non-pinned, non-safety memories.
        if self._half_life_s > 0:
            for item in items:
                if item.pinned or item.status != MemoryStatus.ACTIVE.value:
                    continue
                if item.metadata.get("safety") or "safety" in item.tags:
                    continue
                age_s = max(now - item.event_time, 0.0)
                if age_s < self._half_life_s:
                    continue
                decay = 0.5 ** (age_s / self._half_life_s)
                new_importance = max(self._min_importance, item.importance * decay)
                if new_importance < item.importance - 1e-6:
                    self._repo._client.update(
                        "memory_items",
                        item.memory_id,
                        {"importance": new_importance, "updated_at": now},
                    )
                    result.decayed += 1
        return result
