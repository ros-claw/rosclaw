"""Tests for Gate B.1 source streams, clock mapping, and canonical timeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.canonical_timeline import (
    build_canonical_timeline,
)
from rosclaw.integrations.lerobot.clock_mapping import (
    build_clock_mappings,
    normalize_bundle_timestamps,
)
from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.source_stream_reader import (
    detect_input_timing_mode,
    read_source_bundle,
)
from rosclaw.integrations.lerobot.source_stream_schema import (
    ClockMapping,
    ClockSyncBundle,
    SourceSample,
    SourceStream,
    SourceStreamBundle,
)

ASYNC_FIXTURE = Path(__file__).parent.parent.parent / "examples" / "practice" / "async_physical_episode"


def test_detect_input_timing_mode() -> None:
    assert detect_input_timing_mode(ASYNC_FIXTURE) == "source_streams"
    minimal = ASYNC_FIXTURE.parent / "minimal_lerobot_episode"
    assert detect_input_timing_mode(minimal) == "aligned_frames"


def test_read_source_bundle_keys_and_counts() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    keys = bundle.stream_keys()
    assert "observation.state" in keys
    assert "action" in keys
    assert "observation.motor_current" in keys
    assert "observation.force_torque" in keys
    assert "observation.contact" in keys
    assert "observation.images.front" in keys

    assert len(bundle.streams["observation.state"].samples) == 101
    assert len(bundle.streams["action"].samples) == 11
    assert len(bundle.streams["observation.motor_current"].samples) == 51
    assert len(bundle.streams["observation.force_torque"].samples) == 201
    assert len(bundle.streams["observation.contact"].samples) == 3
    assert len(bundle.streams["observation.images.front"].samples) == 31


def test_read_source_bundle_stream_types() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    assert bundle.streams["observation.state"].stream_type == "continuous"
    assert bundle.streams["action"].stream_type == "continuous"
    assert bundle.streams["observation.contact"].stream_type == "discrete"
    assert bundle.streams["observation.images.front"].stream_type == "image"


def test_read_source_bundle_infers_shapes() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    assert bundle.streams["observation.state"].shape == [3]
    assert bundle.streams["action"].shape == [1]
    assert bundle.streams["observation.motor_current"].shape == [2]
    assert bundle.streams["observation.force_torque"].shape == [6]


def test_read_source_bundle_image_paths() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    front = bundle.streams["observation.images.front"]
    assert front.samples[0].image_path == "frames/front/000000.png"
    assert front.samples[-1].image_path == "frames/front/000030.png"


def test_identity_clock_mapping_for_single_clock() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    result = build_clock_mappings(bundle)
    assert result.target_clock == "episode_time"
    assert "monotonic" in result.mappings
    assert result.mappings["monotonic"].model == "identity"
    assert result.unmapped_clocks == set()


def test_offset_clock_mapping() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=1_000_000_000, clock_domain="device_time", value=[0.0]),
                ],
            ),
        },
        clock_sync=ClockSyncBundle(
            target_clock="episode_time",
            mappings=[
                ClockMapping(
                    source_clock="device_time",
                    target_clock="episode_time",
                    model="offset",
                    offset_ns=-1_000_000_000,
                ),
            ],
        ),
    )
    result = build_clock_mappings(bundle)
    assert result.mappings["device_time"].model == "offset"
    normalized = normalize_bundle_timestamps(bundle, result)
    assert normalized["observation.state"][0][1] == 0


def test_affine_clock_mapping() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=1_000_000_000, clock_domain="camera_device", value=[0.0]),
                ],
            ),
        },
        clock_sync=ClockSyncBundle(
            target_clock="episode_time",
            mappings=[
                ClockMapping(
                    source_clock="camera_device",
                    target_clock="episode_time",
                    model="affine",
                    scale=2.0,
                    offset_ns=500_000_000,
                ),
            ],
        ),
    )
    result = build_clock_mappings(bundle)
    assert result.mappings["camera_device"].model == "affine"
    normalized = normalize_bundle_timestamps(bundle, result)
    # 2.0 * 1e9 + 5e8 = 2.5e9
    assert normalized["observation.state"][0][1] == 2_500_000_000


def test_mixed_clock_without_mapping_fails() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                ],
            ),
            "observation.images.front": SourceStream(
                key="observation.images.front",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="camera_device", image_path="f.png"),
                ],
            ),
        },
    )
    with pytest.raises(NormalizationError) as exc_info:
        build_clock_mappings(bundle)
    assert exc_info.value.code == "clock_mapping_missing"


def test_mixed_clock_with_allow_unmapped() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                ],
            ),
            "observation.images.front": SourceStream(
                key="observation.images.front",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="camera_device", image_path="f.png"),
                ],
            ),
        },
    )
    result = build_clock_mappings(bundle, allow_unmapped=True)
    assert "camera_device" in result.unmapped_clocks
    assert result.mappings["camera_device"].model == "identity"


def test_canonical_timeline_frame_count() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping_result = build_clock_mappings(bundle)
    timeline = build_canonical_timeline(bundle, mapping_result, fps=10.0)
    assert timeline.num_frames == 11
    assert timeline.start_sec == pytest.approx(0.0)
    assert timeline.end_sec == pytest.approx(1.0)
    assert timeline.timestamps_sec[0] == pytest.approx(0.0)
    assert timeline.timestamps_sec[-1] == pytest.approx(1.0)


def test_explicit_episode_bounds_override() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping_result = build_clock_mappings(bundle)
    timeline = build_canonical_timeline(
        bundle, mapping_result, fps=10.0, start_sec=0.2, end_sec=0.5
    )
    assert timeline.num_frames == 4
    assert timeline.start_sec == pytest.approx(0.2)
    assert timeline.end_sec == pytest.approx(0.5)


def test_required_stream_intersection_missing_action() -> None:
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                ],
            ),
        },
    )
    result = build_clock_mappings(bundle)
    with pytest.raises(NormalizationError) as exc_info:
        build_canonical_timeline(bundle, result, fps=10.0)
    assert exc_info.value.code == "required_stream_missing_for_timeline"


def test_timeline_floating_point_boundary() -> None:
    # Slightly over 1.0 due to floating point should not create an extra frame.
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=1_000_000_001, clock_domain="monotonic", value=[1.0]),
                ],
            ),
            "action": SourceStream(
                key="action",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=1_000_000_001, clock_domain="monotonic", value=[1.0]),
                ],
            ),
        },
    )
    result = build_clock_mappings(bundle)
    timeline = build_canonical_timeline(bundle, result, fps=10.0)
    assert timeline.num_frames == 11
    assert timeline.timestamps_sec[-1] == pytest.approx(1.0)
