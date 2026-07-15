"""Clock mapping for Gate B.1 asynchronous sensor synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It applies identity/offset/affine mappings to bring heterogeneous
source clocks onto a canonical episode-time clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.source_stream_schema import (
    ClockMapping,
    SourceStreamBundle,
)

# Default clock assumed for aligned-frames episodes (no raw source streams).
DEFAULT_TARGET_CLOCK = "episode_time"


@dataclass
class ClockMappingResult:
    """Result of normalizing one bundle's source timestamps."""

    target_clock: str
    mappings: dict[str, ClockMapping]
    unmapped_clocks: set[str]
    allow_unmapped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_clock": self.target_clock,
            "mappings": {k: v.to_dict() for k, v in self.mappings.items()},
            "unmapped_clocks": sorted(self.unmapped_clocks),
            "allow_unmapped": self.allow_unmapped,
        }


def _identity_mapping(source_clock: str, target_clock: str) -> ClockMapping:
    return ClockMapping(
        source_clock=source_clock,
        target_clock=target_clock,
        model="identity",
        scale=1.0,
        offset_ns=0,
        authoritative=True,
    )


def build_clock_mappings(
    bundle: SourceStreamBundle,
    *,
    target_clock: str = DEFAULT_TARGET_CLOCK,
    allow_unmapped: bool = False,
) -> ClockMappingResult:
    """Resolve a mapping for every clock domain present in ``bundle``.

    For a single-clock bundle, an identity mapping is implied.  For mixed
    clocks, each source clock must have a mapping in ``clock_sync.json`` or
    ``allow_unmapped`` must be true.
    """
    source_clocks = {
        sample.clock_domain
        for stream in bundle.streams.values()
        for sample in stream.samples
    }
    # The target clock itself does not need mapping.
    source_clocks.discard(target_clock)

    # Build a lookup from the explicit clock_sync bundle.
    explicit: dict[str, ClockMapping] = {
        m.source_clock: m
        for m in bundle.clock_sync.mappings
        if m.target_clock == target_clock
    }

    mappings: dict[str, ClockMapping] = {}
    unmapped: set[str] = set()

    if not source_clocks:
        # Nothing to map; return an empty result.
        return ClockMappingResult(
            target_clock=target_clock,
            mappings=mappings,
            unmapped_clocks=unmapped,
            allow_unmapped=allow_unmapped,
        )

    if len(source_clocks) == 1:
        only_clock = next(iter(source_clocks))
        if only_clock in explicit:
            mappings[only_clock] = explicit[only_clock]
        else:
            # Single non-target clock: imply identity mapping.
            mappings[only_clock] = _identity_mapping(only_clock, target_clock)
        return ClockMappingResult(
            target_clock=target_clock,
            mappings=mappings,
            unmapped_clocks=unmapped,
            allow_unmapped=allow_unmapped,
        )

    # Multiple source clocks: explicit mappings required for all non-target clocks.
    for clock in source_clocks:
        if clock in explicit:
            mappings[clock] = explicit[clock]
        elif allow_unmapped:
            mappings[clock] = _identity_mapping(clock, target_clock)
            unmapped.add(clock)
        else:
            raise NormalizationError(
                "clock_mapping_missing",
                f"Missing clock mapping for '{clock}' -> '{target_clock}'. "
                "Add it to clock_sync.json or pass --allow-unmapped-clock.",
            )

    return ClockMappingResult(
        target_clock=target_clock,
        mappings=mappings,
        unmapped_clocks=unmapped,
        allow_unmapped=allow_unmapped,
    )


def apply_clock_mapping(
    source_timestamp_ns: int,
    mapping: ClockMapping,
) -> int:
    """Map a single source timestamp to the target clock domain."""
    return mapping.apply(source_timestamp_ns)


def normalize_bundle_timestamps(
    bundle: SourceStreamBundle,
    mapping_result: ClockMappingResult,
) -> dict[str, list[tuple[int, int]]]:
    """Return per-stream (source_ts_ns, target_ts_ns) pairs for every sample.

    Invalid samples are still included in the returned pairs so that callers
    can reason about coverage; they should skip invalid values during
    resampling.
    """
    timeline: dict[str, list[tuple[int, int]]] = {}
    for key, stream in bundle.streams.items():
        mapping = mapping_result.mappings.get(stream.samples[0].clock_domain) if stream.samples else None
        pairs: list[tuple[int, int]] = []
        for sample in stream.samples:
            if mapping is None:
                target_ts = sample.source_timestamp_ns
            else:
                target_ts = apply_clock_mapping(sample.source_timestamp_ns, mapping)
            pairs.append((sample.source_timestamp_ns, target_ts))
        timeline[key] = pairs
    return timeline


__all__ = [
    "ClockMappingResult",
    "apply_clock_mapping",
    "build_clock_mappings",
    "normalize_bundle_timestamps",
]
