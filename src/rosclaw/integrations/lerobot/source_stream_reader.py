"""Reader for Gate B.1 source-stream bundles.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It loads the raw multi-rate JSONL streams and ``clock_sync.json``
produced by ``rosclaw.practice.source_bundle.v1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizationError
from rosclaw.integrations.lerobot.source_stream_schema import (
    ClockSyncBundle,
    SourceSample,
    SourceStream,
    SourceStreamBundle,
)

# Maps a JSONL filename (without extension) to the canonical feature key and
# stream classification.  Camera streams are detected by the "camera_" prefix.
_STREAM_FILENAME_TO_KEY: dict[str, tuple[str, str]] = {
    "observation_state": ("observation.state", "continuous"),
    "action": ("action", "continuous"),
    "motor_current": ("observation.motor_current", "continuous"),
    "joint_temperature": ("observation.joint_temperature", "continuous"),
    "joint_velocity": ("observation.joint_velocity", "continuous"),
    "joint_effort": ("observation.joint_effort", "continuous"),
    "force_torque": ("observation.force_torque", "continuous"),
    "contact": ("observation.contact", "discrete"),
    "sandbox": ("rosclaw.sandbox", "event"),
    "failure": ("rosclaw.failure", "event"),
    "intervention": ("rosclaw.intervention", "event"),
}


def _resolve_stream_key(stem: str) -> tuple[str, str]:
    """Return (feature_key, stream_type) for a JSONL filename stem."""
    if stem in _STREAM_FILENAME_TO_KEY:
        return _STREAM_FILENAME_TO_KEY[stem]
    if stem.startswith("camera_"):
        camera_name = stem[len("camera_"):]
        return (f"observation.images.{camera_name}", "image")
    # Unknown streams are treated as continuous values and use the stem as key.
    return (stem, "continuous")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping empty lines."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NormalizationError(
                    "source_stream_jsonl_invalid",
                    f"Invalid JSON in {path} at line {line_no}: {exc}",
                ) from exc
            if not isinstance(record, dict):
                raise NormalizationError(
                    "source_stream_record_invalid",
                    f"Each line of {path} must be an object (line {line_no}).",
                )
            records.append(record)
    return records


def _build_stream(key: str, stream_type: str, records: list[dict[str, Any]]) -> SourceStream:
    """Convert JSONL records into a SourceStream."""
    samples: list[SourceSample] = []
    prev_ts: int | None = None
    for idx, record in enumerate(records):
        ts = record.get("source_timestamp_ns")
        if ts is None:
            raise NormalizationError(
                "source_timestamp_missing",
                f"Missing source_timestamp_ns in stream '{key}' at index {idx}.",
            )
        ts = int(ts)
        if prev_ts is not None and ts < prev_ts:
            raise NormalizationError(
                "source_timestamp_not_monotonic",
                f"Timestamps must be non-decreasing in stream '{key}' "
                f"(index {idx}: {ts} < {prev_ts}).",
            )
        prev_ts = ts

        value = record.get("value")
        image_path = record.get("path")
        event = record.get("event")

        samples.append(
            SourceSample(
                sequence=int(record.get("sequence", idx)),
                source_timestamp_ns=ts,
                clock_domain=record.get("clock_domain", "unknown"),
                valid=bool(record.get("valid", True)),
                value=value,
                image_path=image_path,
                event=event,
                metadata=dict(record.get("metadata", {})),
            )
        )

    dtype = "float32"
    shape: list[int] = []
    units: str | None = None
    names: list[str] | None = None

    # Infer dtype/shape from the first sample that carries a value.
    for sample in samples:
        if sample.value is None:
            continue
        if isinstance(sample.value, bool):
            dtype = "bool"
            shape = []
        elif isinstance(sample.value, list):
            dtype = "float32"
            shape = [len(sample.value)]
        elif isinstance(sample.value, int):
            dtype = "int32"
            shape = []
        elif isinstance(sample.value, float):
            dtype = "float32"
            shape = []
        break

    return SourceStream(
        key=key,
        samples=samples,
        dtype=dtype,
        shape=shape,
        units=units,
        names=names,
        stream_type=stream_type,  # type: ignore[arg-type]
        )


def read_clock_sync(base_dir: Path | str) -> ClockSyncBundle:
    """Load ``clock_sync.json`` if present, otherwise return an empty bundle."""
    base_dir = Path(base_dir)
    path = base_dir / "clock_sync.json"
    if not path.exists():
        return ClockSyncBundle()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NormalizationError(
            "clock_sync_invalid",
            f"Invalid JSON in {path}: {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise NormalizationError(
            "clock_sync_invalid",
            f"{path} must contain a JSON object.",
        )
    return ClockSyncBundle.from_dict(data)


def read_source_bundle(
    base_dir: Path | str,
    episode_id: str = "",
) -> SourceStreamBundle:
    """Load a source-stream bundle from ``base_dir``.

    Expects ``streams/*.jsonl`` plus an optional ``clock_sync.json``.
    If no ``streams`` directory exists, raises ``NormalizationError``.
    """
    base_dir = Path(base_dir)
    streams_dir = base_dir / "streams"
    if not streams_dir.exists():
        raise NormalizationError(
            "source_streams_missing",
            f"No streams directory found in {base_dir}.",
        )

    streams: dict[str, SourceStream] = {}
    for jsonl_path in sorted(streams_dir.glob("*.jsonl")):
        key, stream_type = _resolve_stream_key(jsonl_path.stem)
        records = _read_jsonl(jsonl_path)
        streams[key] = _build_stream(key, stream_type, records)

    if not streams:
        raise NormalizationError(
            "source_streams_empty",
            f"No stream JSONL files found in {streams_dir}.",
        )

    clock_sync = read_clock_sync(base_dir)

    return SourceStreamBundle(
        episode_id=episode_id or base_dir.name,
        base_dir=str(base_dir.resolve()),
        streams=streams,
        clock_sync=clock_sync,
        metadata={},
        input_timing_mode="source_streams",
    )


def detect_input_timing_mode(base_dir: Path | str) -> str:
    """Return ``source_streams`` if a ``streams`` directory exists, else ``aligned_frames``."""
    base_dir = Path(base_dir)
    if (base_dir / "streams").is_dir() and any((base_dir / "streams").glob("*.jsonl")):
        return "source_streams"
    return "aligned_frames"


__all__ = [
    "detect_input_timing_mode",
    "read_clock_sync",
    "read_source_bundle",
]
