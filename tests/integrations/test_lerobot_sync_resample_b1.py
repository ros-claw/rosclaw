"""Tests for Gate B.1 resampling, missingness policies, and synchronization."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.canonical_timeline import build_canonical_timeline
from rosclaw.integrations.lerobot.clock_mapping import build_clock_mappings
from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.source_stream_reader import read_source_bundle
from rosclaw.integrations.lerobot.source_stream_schema import (
    SourceSample,
    SourceStream,
    SourceStreamBundle,
)
from rosclaw.integrations.lerobot.sync_config import default_sync_config
from rosclaw.integrations.lerobot.sync_provenance import (
    PROVENANCE_AGGREGATED,
    PROVENANCE_EXACT,
    PROVENANCE_FILLED_NAN,
    PROVENANCE_HELD,
    PROVENANCE_INTERPOLATED,
    VALIDITY_INVALID,
    VALIDITY_VALID,
)
from rosclaw.integrations.lerobot.synchronize import synchronize_bundle

ASYNC_FIXTURE = Path(__file__).parent.parent.parent / "examples" / "practice" / "async_physical_episode"


def _small_bundle() -> SourceStreamBundle:
    """Return a bundle where state has a 500 ms gap to exercise missingness."""
    return SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0, 0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=500_000_000, clock_domain="monotonic", value=[0.5, 1.0]),
                ],
            ),
            "action": SourceStream(
                key="action",
                samples=[
                    SourceSample(sequence=0, source_timestamp_ns=0, clock_domain="monotonic", value=[0.0]),
                    SourceSample(sequence=1, source_timestamp_ns=500_000_000, clock_domain="monotonic", value=[1.0]),
                ],
            ),
        },
    )


def test_synchronize_bundle_state_linear() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=30.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    assert result.timeline.num_frames == 31
    assert result.timeline.start_sec == pytest.approx(0.0)
    assert result.timeline.end_sec == pytest.approx(1.0)

    state = result.features["observation.state"]
    # Exact sample at 0.10 s (100 Hz grid).
    assert state.values[3] == pytest.approx([0.1, 0.2, 0.3])
    assert state.provenance[3] == PROVENANCE_EXACT
    # Interpolated sample at 0.0333 s.
    assert state.values[1] == pytest.approx([0.0333, 0.0667, 0.1], abs=1e-4)
    assert state.provenance[1] == PROVENANCE_INTERPOLATED


def test_synchronize_bundle_action_previous() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=25.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    action = result.features["action"]
    # Held from the 0.0 s action sample.
    assert action.values[1] == [0.0]
    assert action.provenance[1] == PROVENANCE_HELD
    assert action.source_skew_ms[1] == pytest.approx(40.0)
    # Exact match at 0.2 s (floor(10 * 0.2) == 2).
    assert action.values[5] == [2.0]
    assert action.provenance[5] == PROVENANCE_EXACT


def test_synchronize_bundle_motor_current_interval_mean() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    current = result.features["observation.motor_current"]
    assert current.provenance[0] == PROVENANCE_AGGREGATED
    # Window around t=0 includes samples at 0.00, 0.02, 0.04 s.
    assert current.values[0] == pytest.approx([1.02, 2.02])
    assert current.peak_values is not None
    assert current.peak_values[0] == pytest.approx([1.04, 2.04])


def test_synchronize_bundle_force_torque_peak_abs() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    ft = result.features["observation.force_torque"]
    assert ft.provenance[1] == PROVENANCE_AGGREGATED
    assert ft.peak_values is not None
    # 200 Hz samples in [0.05, 0.15) peak absolute Fx is just below 0.15.
    assert ft.peak_values[1][0] == pytest.approx(0.15, abs=0.01)


def test_synchronize_bundle_contact_interval_any() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    contact = result.features["observation.contact"]
    assert contact.values[0] == [0]  # false at t=0
    assert contact.values[5] == [1]  # true around t=0.5
    assert contact.values[6] == [0]  # false at t=0.55
    assert contact.valid[1] == VALIDITY_INVALID  # no event around t=0.1


def test_synchronize_bundle_image_nearest() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="nan")

    images = result.features["observation.images.front"]
    assert images.values[0] == "frames/front/000000.png"
    assert images.values[1] == "frames/front/000003.png"
    assert images.provenance[1] == PROVENANCE_EXACT
    assert images.valid[1] == VALIDITY_VALID


def test_missing_policy_error_raises() -> None:
    bundle = _small_bundle()
    config = default_sync_config(fps=10.0)
    config.policies["observation.state"] = config.policy_for("observation.state")
    config.policies["observation.state"].max_gap_ms = 100.0

    with pytest.raises(NormalizationError) as exc_info:
        synchronize_bundle(bundle, config, missing_policy="error")
    assert exc_info.value.code == "sync_required_feature_missing"


def test_missing_policy_drop_frame_atomic() -> None:
    bundle = _small_bundle()
    config = default_sync_config(fps=10.0)
    config.policies["observation.state"].max_gap_ms = 100.0

    result = synchronize_bundle(bundle, config, missing_policy="drop-frame")

    # Frames at 0.0 and 0.5 are valid; interior frames are dropped atomically.
    assert result.timeline.num_frames == 6
    assert result.kept_frame_indices == [0, 5]
    assert result.dropped_frame_indices == [1, 2, 3, 4]

    state = result.features["observation.state"]
    assert len(state.values) == 2
    assert state.values[0] == [0.0, 0.0]
    assert state.values[1] == [0.5, 1.0]

    action = result.features["action"]
    assert len(action.values) == 2
    assert action.values[0] == [0.0]
    assert action.values[1] == [1.0]


def test_missing_policy_nan_fills_float_arrays() -> None:
    bundle = _small_bundle()
    config = default_sync_config(fps=10.0)
    config.policies["observation.state"].max_gap_ms = 100.0

    result = synchronize_bundle(bundle, config, missing_policy="nan")

    state = result.features["observation.state"]
    assert state.values[0] == [0.0, 0.0]
    assert all(math.isnan(v) for v in state.values[2])
    assert state.provenance[2] == PROVENANCE_FILLED_NAN
    assert state.valid[2] == VALIDITY_INVALID


def test_missing_policy_fill_last_holds_previous() -> None:
    bundle = _small_bundle()
    config = default_sync_config(fps=10.0)
    # action previous with a generous max_age holds all frames anyway.
    result = synchronize_bundle(bundle, config, missing_policy="fill-last")

    action = result.features["action"]
    assert action.values[2] == [0.0]
    assert action.provenance[2] == PROVENANCE_HELD
    assert action.valid[2] == VALIDITY_VALID


def test_synchronize_bundle_with_explicit_timeline() -> None:
    bundle = read_source_bundle(ASYNC_FIXTURE)
    mapping = build_clock_mappings(bundle)
    timeline = build_canonical_timeline(
        bundle, mapping, fps=10.0, start_sec=0.2, end_sec=0.5
    )
    config = default_sync_config(fps=10.0)

    result = synchronize_bundle(bundle, config, missing_policy="nan", timeline=timeline)

    assert result.timeline.num_frames == 4
    assert result.timeline.timestamps_sec[0] == pytest.approx(0.2)
    assert len(result.features["observation.state"].values) == 4


def test_synchronize_bundle_image_missing_drop_frame() -> None:
    """If an image stream is shorter than required streams, drop-frame removes those frames."""
    bundle = SourceStreamBundle(
        streams={
            "observation.state": SourceStream(
                key="observation.state",
                samples=[
                    SourceSample(sequence=i, source_timestamp_ns=i * 100_000_000, clock_domain="monotonic", value=[float(i)])
                    for i in range(11)
                ],
            ),
            "action": SourceStream(
                key="action",
                samples=[
                    SourceSample(sequence=i, source_timestamp_ns=i * 100_000_000, clock_domain="monotonic", value=[float(i)])
                    for i in range(11)
                ],
            ),
            "observation.images.front": SourceStream(
                key="observation.images.front",
                samples=[
                    SourceSample(sequence=i, source_timestamp_ns=i * 100_000_000, clock_domain="monotonic", image_path=f"frames/front/{i:06d}.png")
                    for i in range(5)
                ],
                stream_type="image",
            ),
        },
    )
    config = default_sync_config(fps=10.0)
    result = synchronize_bundle(bundle, config, missing_policy="drop-frame")

    # Frames 5..10 have no nearby image and must be dropped.
    assert result.dropped_frame_indices == [5, 6, 7, 8, 9, 10]
    assert len(result.features["observation.images.front"].values) == 5
