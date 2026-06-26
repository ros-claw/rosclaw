"""Fleet compatibility cache.

Cache key:
    (workspace_id, body_instance_id, effective_body_hash, skill_manifest_hash)

Invalidation conditions:
- body effective hash changed
- skill manifest changed
- registry active body changed
- SENSE_BODY_UPDATED event
- provider health safety-affecting event
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.body.fleet import FleetCompatibilityAggregator, discover_skill_manifests
from rosclaw.body.schema import FleetCompatibilityReport, SkillManifest


@dataclass
class _CacheEntry:
    report: FleetCompatibilityReport
    created_at: float
    access_count: int = 0


class FleetCompatibilityCache:
    """In-memory cache for fleet compatibility reports.

    The cache is intentionally simple (dict-backed) for P0/P0.5. In P1 it can
    be backed by a persistent store or shared across processes.
    """

    def __init__(self, workspace: Path | str, ttl_sec: float = 300.0):
        self.workspace = Path(workspace)
        self.ttl_sec = ttl_sec
        self._cache: dict[str, _CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0

    def _skill_manifest_hash(self, manifests: list[SkillManifest]) -> str:
        canonical = json.dumps(
            sorted([m.to_dict() for m in manifests], key=lambda d: json.dumps(d, sort_keys=True)),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_key(
        self,
        body_ids: frozenset[str],
        effective_hashes: dict[str, str],
        manifest_hash: str,
    ) -> str:
        data = {
            "workspace": str(self.workspace),
            "bodies": {bid: effective_hashes.get(bid, "") for bid in sorted(body_ids)},
            "manifest_hash": manifest_hash,
        }
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _current_state(self, skill_manifests: list[SkillManifest] | None = None) -> tuple[frozenset[str], dict[str, str], str]:
        from rosclaw.body.registry import BodyRegistryManager

        manager = BodyRegistryManager(self.workspace)
        entries = manager.list_bodies()
        body_ids = frozenset(e.body_id for e in entries)
        effective_hashes: dict[str, str] = {}
        for entry in entries:
            try:
                from rosclaw.body.resolver import BodyResolver

                resolver = BodyResolver(self.workspace, body_id=entry.body_id)
                effective = resolver.get_effective_body(recompile_if_stale=False)
                effective_hashes[entry.body_id] = effective.effective_body_hash
            except Exception:
                effective_hashes[entry.body_id] = ""

        manifests = skill_manifests or discover_skill_manifests(self.workspace)
        manifest_hash = self._skill_manifest_hash(manifests)
        return body_ids, effective_hashes, manifest_hash

    def get(
        self,
        skill_manifests: list[SkillManifest] | None = None,
    ) -> FleetCompatibilityReport | None:
        """Return cached report if valid; otherwise None."""
        body_ids, effective_hashes, manifest_hash = self._current_state(skill_manifests)
        key = self._cache_key(body_ids, effective_hashes, manifest_hash)
        entry = self._cache.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        if time.time() - entry.created_at > self.ttl_sec:
            self._miss_count += 1
            del self._cache[key]
            return None
        entry.access_count += 1
        self._hit_count += 1
        return entry.report

    def set(
        self,
        report: FleetCompatibilityReport,
        skill_manifests: list[SkillManifest] | None = None,
    ) -> str:
        """Store a report in the cache and return the cache key."""
        body_ids, effective_hashes, manifest_hash = self._current_state(skill_manifests)
        key = self._cache_key(body_ids, effective_hashes, manifest_hash)
        self._cache[key] = _CacheEntry(report=report, created_at=time.time())
        return key

    def compute_and_cache(
        self,
        skill_manifests: list[SkillManifest] | None = None,
    ) -> FleetCompatibilityReport:
        """Compute fleet compatibility and cache the result."""
        aggregator = FleetCompatibilityAggregator(self.workspace)
        manifests = skill_manifests or discover_skill_manifests(self.workspace)
        report = aggregator.aggregate(manifests)
        self.set(report, manifests)
        return report

    def get_or_compute(
        self,
        skill_manifests: list[SkillManifest] | None = None,
    ) -> FleetCompatibilityReport:
        """Return cached report if valid, else compute and cache."""
        cached = self.get(skill_manifests)
        if cached is not None:
            return cached
        return self.compute_and_cache(skill_manifests)

    def invalidate(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def invalidate_for_body(self, body_instance_id: str) -> int:
        """Remove entries that include the given body."""
        removed = 0
        for key in list(self._cache.keys()):
            entry = self._cache[key]
            if body_instance_id in (entry.report.per_body or {}):
                del self._cache[key]
                removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_rate(),
        }

    def _hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def on_body_changed(self, body_instance_id: str) -> None:
        """Invalidate cache entries affected by a body change."""
        self.invalidate_for_body(body_instance_id)

    def on_skill_manifest_changed(self) -> None:
        """Invalidate all entries when skill manifests change."""
        self.invalidate()

    def on_active_body_switched(self) -> None:
        """Invalidate cache when active body pointer changes."""
        self.invalidate()

    def on_sense_body_updated(self) -> None:
        """Invalidate cache on SENSE_BODY_UPDATED event."""
        self.invalidate()

    def on_provider_health_safety_event(self) -> None:
        """Invalidate cache on safety-affecting provider health event."""
        self.invalidate()
