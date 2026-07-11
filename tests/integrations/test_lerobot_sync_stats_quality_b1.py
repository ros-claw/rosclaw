"""Tests for Gate B.1 sync stats and quality gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.clock_mapping import build_clock_mappings
from rosclaw.integrations.lerobot.source_stream_reader import read_source_bundle
from rosclaw.integrations.lerobot.source_stream_schema import (
    ClockSyncBundle,
    SourceSample,
    SourceStream,
    SourceStreamBundle,
)
from rosclaw.integrations.lerobot.sync_config import default_sync_config
from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality
from rosclaw.integrations.lerobot.sync_stats import (
    compute_sync_stats,
    write_sync_stats_parquet,
)
from rosclaw.integrations.lerobot.synchronize import synchronize_bundle


ASYNC_FIXTURE = Path(__file__).parent.parent.parent / "examples" / "practice" / "async_physical_episode"


def test_sync_stats_fields_and_values() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features,
        bundle.streams,
        config,
        mapping,
        result.timeline,
        episode_index=0,
        dropped_frames=len(result.dropped_frame_indices),
    )

    stats_by_key = {s.feature_key: s for s in stats}
    assert "observation.state" in stats_by_key
    assert "action" in stats_by_key
    assert "observation.motor_current" in stats_by_key
    assert "observation.force_torque" in stats_by_key
    assert "observation.contact" in stats_by_key
    assert "observation.images.front" in stats_by_key

    state = stats_by_key["observation.state"]
    assert state.source_samples == 101
    assert state.target_frames == 11
    assert state.valid_frames == 11
    assert state.coverage_ratio == pytest.approx(1.0)
    assert state.sync_method == "linear"
    assert state.mean_skew_ms is not None

    current = stats_by_key["observation.motor_current"]
    assert current.aggregated_frames == 11
    assert current.mean_samples_per_window == pytest.approx(5.0, abs=0.5)

    contact = stats_by_key["observation.contact"]
    assert contact.valid_frames == 3
    assert contact.missing_frames == 8
    assert contact.active_output_frames == 1


def test_sync_stats_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features,
        bundle.streams,
        config,
        mapping,
        result.timeline,
    )

    parquet_path = tmp_path / "sync_stats.parquet"
    written = write_sync_stats_parquet(stats, parquet_path)
    assert written.exists()

    import pandas as pd

    df = pd.read_parquet(written)
    assert "feature_key" in df.columns
    assert "coverage_ratio" in df.columns
    assert len(df) == len(stats)


def test_quality_balanced_passes_with_warnings() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features,
        bundle.streams,
        config,
        mapping,
        result.timeline,
    )

    quality_config = default_quality_config("balanced")
    quality = evaluate_quality(stats, mapping, quality_config)

    assert quality.passed is True
    assert quality.action == "warn"
    assert any("observation.contact" in w for w in quality.warnings)


def test_quality_strict_fails_required_coverage() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=500_000_000, clock_domain="monotonic", value=[0.5]),
                ],
            ),
            "action": SourceStream(
                key="action",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=1_000_000_000, clock_domain="monotonic", value=[1.0]),
                ],
            ),
        },
    )
    mapping = build_clock_mappings(bundle)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features,
        bundle.streams,
        config,
        mapping,
        result.timeline,
    )

    quality_config = default_quality_config("balanced")
    quality = evaluate_quality(stats, mapping, quality_config)

    assert quality.passed is False
    assert quality.action == "fail"
    assert any("observation.state" in e for e in quality.errors)


def test_quality_unmapped_clock_fails() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=1_000_000_000, clock_domain="monotonic", value=[1.0]),
                ],
            ),
            "action": SourceStream(
                key="action",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=1_000_000_000, clock_domain="monotonic", value=[1.0]),
                ],
            ),
            "observation.images.front": SourceStream(
                key="observation.images.front",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="camera_device", image_path="f.png"),
                ],
                stream_type="image",
            ),
        },
        clock_sync=ClockSyncBundle(),
    )
    mapping = build_clock_mappings(bundle, allow_unmapped=True)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan", mapping_result=mapping)
    stats = compute_sync_stats(
        result.features,
        bundle.streams,
        config,
        mapping,
        result.timeline,
    )

    quality_config = default_quality_config("balanced")
    quality = evaluate_quality(stats, mapping, quality_config)

    assert quality.passed is False
    assert quality.action == "fail"
    assert "camera_device" in str(quality.errors)
