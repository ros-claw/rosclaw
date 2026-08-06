"""Shared world model: freshness-aware delta merge (PR-TF-073 core).

Rules (总纲 §10.5, §10.8):
- every object state carries time/frame/source/confidence/revision and an
  explicit measurement/inference class;
- merge policy ``latest_valid``: newest non-tombstone observation wins,
  but only if it is fresh (observed_at within max_age_ms of merge time);
- clock skew beyond tolerance rejects fusion (never pretend timestamps
  are comparable);
- stale world state degrades behavior — it is reported, never silently
  treated as current.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rosclaw.contracts.common import ValidationError
from rosclaw.contracts.team.world import ObjectState, SharedWorldDeltaV1

DEFAULT_CLOCK_TOLERANCE_MS = 500


class StaleWorldError(ValidationError):
    """World data older than its max_age must not be fused or used."""


class ClockSkewError(ValidationError):
    """Source clock disagrees with ours beyond tolerance."""


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class WorldModel:
    def __init__(self, *, clock_tolerance_ms: int = DEFAULT_CLOCK_TOLERANCE_MS) -> None:
        # object_id -> (observed_at, ObjectState): latest-valid by observation
        # time, not arrival order (a late old observation must not clobber a
        # newer one — 总纲 §10.5).
        self._objects: dict[str, tuple[datetime, ObjectState]] = {}
        self._sources: dict[str, datetime] = {}
        self._world_revision = 0
        self._tolerance_ms = clock_tolerance_ms

    @property
    def world_revision(self) -> int:
        return self._world_revision

    def merge_delta(self, delta: SharedWorldDeltaV1, *, now: datetime) -> list[str]:
        """Fuse a delta. Returns warnings; raises on clock skew/stale data."""
        observed = _parse(delta.observed_at)
        skew_ms = abs((now - observed).total_seconds()) * 1000.0
        if skew_ms > self._tolerance_ms + delta.max_age_ms:
            raise ClockSkewError(
                f"delta from {delta.source_member} observed {skew_ms:.0f} ms away "
                f"(tolerance {self._tolerance_ms} + max_age {delta.max_age_ms})"
            )
        warnings: list[str] = []
        changed = False
        for obj in delta.objects:
            if obj.tombstone:
                if self._objects.pop(obj.object_id, None) is not None:
                    changed = True
                continue
            existing = self._objects.get(obj.object_id)
            if existing is not None and observed < existing[0]:
                warnings.append(
                    f"object {obj.object_id!r}: stale observation ignored "
                    f"(existing from {existing[0].isoformat()})"
                )
                continue
            if existing is None or existing[1] != obj:
                changed = True
            self._objects[obj.object_id] = (observed, obj)
        self._sources[delta.source_member] = observed
        if changed:
            self._world_revision = max(self._world_revision + 1, delta.world_revision)
        return warnings

    def fresh_objects(self, *, now: datetime, max_age_ms: int) -> dict[str, ObjectState]:
        """Objects whose observation is still usable. Callers must not act
        on anything absent from this view."""
        fresh: dict[str, ObjectState] = {}
        for object_id, (observed, obj) in self._objects.items():
            age_ms = (now - observed).total_seconds() * 1000.0
            if age_ms <= max_age_ms:
                fresh[object_id] = obj
        return fresh

    def staleness_ms(self, *, now: datetime) -> float | None:
        if not self._sources:
            return None
        newest = max(self._sources.values())
        return (now - newest).total_seconds() * 1000.0
