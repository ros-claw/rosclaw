"""Source-stream schema for Gate B.1 asynchronous sensor synchronization.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It describes the raw, multi-rate input layer that sits *before*
the canonical LeRobotDataset timeline is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SOURCE_BUNDLE_SCHEMA_VERSION = "rosclaw.practice.source_bundle.v1"
CLOCK_SYNC_SCHEMA_VERSION = "rosclaw.clock_sync.v1"

ClockDomain = Literal[
    "episode_time",
    "monotonic",
    "ros_time",
    "device_time",
    "camera_device",
    "wall_time",
    "unknown",
]

ClockMappingModel = Literal["identity", "offset", "affine"]


@dataclass
class ClockMapping:
    """Affine mapping from one clock domain to another."""

    source_clock: str
    target_clock: str
    model: ClockMappingModel = "identity"
    scale: float = 1.0
    offset_ns: int = 0
    anchor_count: int | None = None
    fit_rmse_ns: float | None = None
    authoritative: bool = True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_clock": self.source_clock,
            "target_clock": self.target_clock,
            "model": self.model,
            "scale": self.scale,
            "offset_ns": self.offset_ns,
            "authoritative": self.authoritative,
        }
        if self.anchor_count is not None:
            out["anchor_count"] = self.anchor_count
        if self.fit_rmse_ns is not None:
            out["fit_rmse_ns"] = self.fit_rmse_ns
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClockMapping:
        return cls(
            source_clock=data.get("source_clock", "unknown"),
            target_clock=data.get("target_clock", "episode_time"),
            model=data.get("model", "identity"),  # type: ignore[arg-type]
            scale=float(data.get("scale", 1.0)),
            offset_ns=int(data.get("offset_ns", 0)),
            anchor_count=data.get("anchor_count"),
            fit_rmse_ns=data.get("fit_rmse_ns"),
            authoritative=bool(data.get("authoritative", True)),
        )

    def apply(self, source_timestamp_ns: int) -> int:
        """Return the target timestamp in nanoseconds."""
        # Use float for scale, then round back to integer nanoseconds.
        scaled = self.scale * float(source_timestamp_ns)
        return int(round(scaled + float(self.offset_ns)))


@dataclass
class ClockSyncBundle:
    """Collection of mappings used to bring heterogeneous clocks together."""

    schema_version: str = CLOCK_SYNC_SCHEMA_VERSION
    target_clock: str = "episode_time"
    mappings: list[ClockMapping] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_clock": self.target_clock,
            "mappings": [m.to_dict() for m in self.mappings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClockSyncBundle:
        return cls(
            schema_version=data.get("schema_version", CLOCK_SYNC_SCHEMA_VERSION),
            target_clock=data.get("target_clock", "episode_time"),
            mappings=[
                ClockMapping.from_dict(m) for m in data.get("mappings", [])
            ],
        )


@dataclass
class SourceSample:
    """A single raw sample from a source stream."""

    sequence: int
    source_timestamp_ns: int
    clock_domain: str
    valid: bool = True
    value: Any | None = None
    image_path: str | None = None
    event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sequence": self.sequence,
            "source_timestamp_ns": self.source_timestamp_ns,
            "clock_domain": self.clock_domain,
            "valid": self.valid,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.image_path is not None:
            out["path"] = self.image_path
        if self.event is not None:
            out["event"] = self.event
        if self.metadata:
            out["metadata"] = self.metadata
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSample:
        return cls(
            sequence=int(data.get("sequence", 0)),
            source_timestamp_ns=int(data["source_timestamp_ns"]),
            clock_domain=data.get("clock_domain", "unknown"),
            valid=bool(data.get("valid", True)),
            value=data.get("value"),
            image_path=data.get("path"),
            event=data.get("event"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SourceStream:
    """A named stream of source samples with a feature key and shape metadata."""

    key: str
    samples: list[SourceSample] = field(default_factory=list)
    dtype: str = "float32"
    shape: list[int] = field(default_factory=list)
    units: str | None = None
    names: list[str] | None = None
    stream_type: Literal["continuous", "discrete", "event", "image"] = "continuous"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "samples": [s.to_dict() for s in self.samples],
            "dtype": self.dtype,
            "shape": self.shape,
            "stream_type": self.stream_type,
        }
        if self.units is not None:
            out["units"] = self.units
        if self.names is not None:
            out["names"] = self.names
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceStream:
        return cls(
            key=data.get("key", ""),
            samples=[SourceSample.from_dict(s) for s in data.get("samples", [])],
            dtype=data.get("dtype", "float32"),
            shape=list(data.get("shape", [])),
            units=data.get("units"),
            names=list(data.get("names", [])) if data.get("names") else None,
            stream_type=data.get("stream_type", "continuous"),  # type: ignore[arg-type]
        )


@dataclass
class SourceStreamBundle:
    """All source streams and clock mappings for one episode."""

    schema_version: str = SOURCE_BUNDLE_SCHEMA_VERSION
    episode_id: str = ""
    base_dir: str = ""
    streams: dict[str, SourceStream] = field(default_factory=dict)
    clock_sync: ClockSyncBundle = field(default_factory=ClockSyncBundle)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_timing_mode: Literal["aligned_frames", "source_streams"] = "source_streams"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "base_dir": self.base_dir,
            "streams": {k: v.to_dict() for k, v in self.streams.items()},
            "clock_sync": self.clock_sync.to_dict(),
            "metadata": self.metadata,
            "input_timing_mode": self.input_timing_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceStreamBundle:
        raw_streams = data.get("streams", {})
        return cls(
            schema_version=data.get("schema_version", SOURCE_BUNDLE_SCHEMA_VERSION),
            episode_id=data.get("episode_id", ""),
            base_dir=data.get("base_dir", ""),
            streams={k: SourceStream.from_dict(v) for k, v in raw_streams.items()},
            clock_sync=ClockSyncBundle.from_dict(data.get("clock_sync", {})),
            metadata=dict(data.get("metadata", {})),
            input_timing_mode=data.get("input_timing_mode", "source_streams"),  # type: ignore[arg-type]
        )

    def stream_keys(self) -> list[str]:
        return sorted(self.streams.keys())

    def required_stream_keys(self) -> list[str]:
        """Streams that must be present for a meaningful episode."""
        return sorted(k for k in self.streams if k in {"observation.state", "action"})


__all__ = [
    "CLOCK_SYNC_SCHEMA_VERSION",
    "ClockDomain",
    "ClockMapping",
    "ClockMappingModel",
    "ClockSyncBundle",
    "SOURCE_BUNDLE_SCHEMA_VERSION",
    "SourceSample",
    "SourceStream",
    "SourceStreamBundle",
]
