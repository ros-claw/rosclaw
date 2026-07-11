"""Report-block builders and sidecar writers for Gate B.1 synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It produces the ``synchronization`` and ``missingness`` blocks of the
v1.2 dataset export report plus ``meta/rosclaw`` sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult
from rosclaw.integrations.lerobot.sync_config import SyncConfig
from rosclaw.integrations.lerobot.sync_quality import QualityResult
from rosclaw.integrations.lerobot.sync_stats import SyncFeatureStats
from rosclaw.integrations.lerobot.synchronize import SynchronizationResult


def build_synchronization_report_block(
    sync_result: SynchronizationResult,
    mapping_result: ClockMappingResult,
    quality_result: QualityResult | None,
    *,
    input_timing_mode: str = "source_streams",
    quality_profile: str = "balanced",
) -> dict[str, Any]:
    """Build the ``synchronization`` block for a v1.2 export report."""
    block: dict[str, Any] = {
        "input_mode": input_timing_mode,
        "level": "validated" if input_timing_mode == "source_streams" else "assumed_aligned",
        "resampled": input_timing_mode == "source_streams",
        "target_fps": sync_result.timeline.fps,
        "canonical_frames": sync_result.timeline.num_frames,
        "written_frames": len(sync_result.kept_frame_indices),
        "dropped_frames": len(sync_result.dropped_frame_indices),
        "clock_domains": sorted(
            {mapping.source_clock for mapping in mapping_result.mappings.values()}
        ),
        "clock_mappings_valid": not bool(mapping_result.unmapped_clocks),
        "unmapped_clock_domains": sorted(mapping_result.unmapped_clocks),
        "quality_profile": quality_profile,
        "quality_passed": quality_result.passed if quality_result else None,
        "warnings": quality_result.warnings if quality_result else [],
    }
    return block


def build_missingness_report_block(policy: str) -> dict[str, Any]:
    """Build the ``missingness`` block for a v1.2 export report."""
    return {
        "policy": policy,
        "unknown_float_encoding": "NaN",
        "unknown_bool_encoding": -1,
        "unknown_category_encoding": 0,
    }


def write_clock_mappings_sidecar(
    mapping_result: ClockMappingResult,
    output_path: Path | str,
) -> Path:
    """Write ``meta/rosclaw/clock_mappings.json``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rosclaw.clock_mappings.v1",
        "target_clock": mapping_result.target_clock,
        "mappings": [m.to_dict() for m in mapping_result.mappings.values()],
        "unmapped_clocks": sorted(mapping_result.unmapped_clocks),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_sync_config_sidecar(
    sync_config: SyncConfig,
    output_path: Path | str,
) -> Path:
    """Write ``meta/rosclaw/sync_config.json``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rosclaw.sync_config.v1",
        **sync_config.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_sync_stats_rows(stats: list[SyncFeatureStats]) -> list[dict[str, Any]]:
    """Return sync statistics as plain dictionaries."""
    return [s.to_dict() for s in stats]


__all__ = [
    "build_missingness_report_block",
    "build_synchronization_report_block",
    "build_sync_stats_rows",
    "write_clock_mappings_sidecar",
    "write_sync_config_sidecar",
]
