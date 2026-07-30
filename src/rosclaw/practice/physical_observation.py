"""Unified physical observation contract (Physical Evolution Lab §5.2).

``PhysicalObservationBundle`` is the single contract every synchronized
multi-modal observation must satisfy: one camera frame aligned to the
nearest left/right hand states, with explicit identity, per-device
timestamps, a clock-sync report, and content-addressed snapshot hashes.

Design rules (v3 §5.1/§16):

* Identity is EXPLICIT — never associate records by directory name or
  timestamp string.
* Every device keeps its own timestamp; the clock-sync report discloses
  the maximum skew instead of pretending the clocks agree.
* ``validate()`` returns violations (gate-friendly); it never raises on
  bad data — bad data is a finding, not an exception.
* Bundle records are immutable evidence once written (to_record → store;
  from_record is the only way back).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "rosclaw.physical_observation.v1"


def canonical_hash(payload: dict[str, Any], *, prefix: str) -> str:
    """Content-addressed snapshot hash (deterministic, JSON canonical)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class EvidenceIdentity:
    """The unified ID spine (v3 §5.1).  Optional at the contract level —
    but every missing id is a validation violation, not a silent gap."""

    experiment_id: str | None = None
    campaign_id: str | None = None
    block_id: str | None = None
    session_id: str | None = None
    episode_id: str | None = None
    round_id: str | None = None
    observation_id: str | None = None

    def missing(self) -> list[str]:
        return [
            name
            for name in ("experiment_id", "session_id", "episode_id", "observation_id")
            if not getattr(self, name)
        ]


@dataclass(frozen=True)
class ClockSyncReport:
    """Disclosed clock skew between devices (v3 §5.2 time block)."""

    samples: int
    max_skew_ms: float
    mean_skew_ms: float
    reference: str  # which clock is the alignment reference, e.g. "host_monotonic_ns"

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "max_skew_ms": round(self.max_skew_ms, 3),
            "mean_skew_ms": round(self.mean_skew_ms, 3),
            "reference": self.reference,
        }

    @classmethod
    def from_pairs(cls, pairs: list[tuple[float, float]], *, reference: str) -> ClockSyncReport:
        """pairs = (device_ts_s, host_ts_s) — skew = |device - host| in ms."""
        if not pairs:
            return cls(
                samples=0, max_skew_ms=float("nan"), mean_skew_ms=float("nan"), reference=reference
            )
        skews = [abs(d - h) * 1000.0 for d, h in pairs]
        return cls(
            samples=len(skews),
            max_skew_ms=max(skews),
            mean_skew_ms=sum(skews) / len(skews),
            reference=reference,
        )


@dataclass(frozen=True)
class HandObservation:
    body_id: str
    body_snapshot_hash: str
    position: dict[str, float]
    force: dict[str, float] | None
    current_ma: dict[str, float] | None
    temperature_c: dict[str, float] | None
    status: str | None
    transport_latency_ms: float | None
    timestamp_ns: int | None


@dataclass(frozen=True)
class CameraObservation:
    camera_id: str
    camera_snapshot_hash: str
    color_artifact: str | None
    depth_artifact: str | None
    frame_age_ms: float | None
    rgb_depth_skew_ms: float | None
    depth_valid_ratio: float | None
    exposure: float | None
    sensor_health: str
    device_timestamp_s: float | None


@dataclass(frozen=True)
class PhysicalObservationBundle:
    """One synchronized observation (camera frame × nearest hand states)."""

    identity: EvidenceIdentity
    host_monotonic_ns: int
    clock_sync: ClockSyncReport
    camera: CameraObservation
    left_hand: HandObservation | None
    right_hand: HandObservation | None
    derived: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    action_id: str | None = None
    receipt_id: str | None = None
    policy_hash: str | None = None
    candidate_hash: str | None = None
    regime_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> list[str]:
        """Contract violations (gate-friendly: bad data is a finding)."""
        violations: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            violations.append(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        for name in self.identity.missing():
            violations.append(f"identity.{name} missing")
        if self.host_monotonic_ns <= 0:
            violations.append("host_monotonic_ns must be positive")
        if self.clock_sync.samples == 0:
            violations.append("clock_sync has no samples")
        cam = self.camera
        if not cam.camera_id:
            violations.append("camera.camera_id missing")
        if cam.frame_age_ms is not None and cam.frame_age_ms < 0:
            violations.append("camera.frame_age_ms negative")
        if cam.rgb_depth_skew_ms is not None and cam.rgb_depth_skew_ms < 0:
            violations.append("camera.rgb_depth_skew_ms negative")
        if cam.depth_valid_ratio is not None and not 0.0 <= cam.depth_valid_ratio <= 1.0:
            violations.append("camera.depth_valid_ratio outside [0,1]")
        if not (self.left_hand or self.right_hand):
            violations.append("at least one hand observation required")
        for side, hand in (("left_hand", self.left_hand), ("right_hand", self.right_hand)):
            if hand is None:
                continue
            if not hand.body_id:
                violations.append(f"{side}.body_id missing")
            if not hand.body_snapshot_hash:
                violations.append(f"{side}.body_snapshot_hash missing")
            for label, mapping in (
                ("position", hand.position),
                ("force", hand.force),
                ("current_ma", hand.current_ma),
                ("temperature_c", hand.temperature_c),
            ):
                if mapping is None:
                    continue
                for joint, value in mapping.items():
                    if not math.isfinite(value):
                        violations.append(f"{side}.{label}.{joint} not finite: {value}")
        derived = self.derived or {}
        conf = derived.get("gesture_confidence")
        if conf is not None and not 0.0 <= float(conf) <= 1.0:
            violations.append("derived.gesture_confidence outside [0,1]")
        contact = derived.get("contact_probability")
        if contact is not None and not 0.0 <= float(contact) <= 1.0:
            violations.append("derived.contact_probability outside [0,1]")
        return violations

    def to_record(self) -> dict[str, Any]:
        def _hand(h: HandObservation | None) -> dict[str, Any] | None:
            if h is None:
                return None
            return {
                "body_id": h.body_id,
                "body_snapshot_hash": h.body_snapshot_hash,
                "position": h.position,
                "force": h.force,
                "current_ma": h.current_ma,
                "temperature_c": h.temperature_c,
                "status": h.status,
                "transport_latency_ms": h.transport_latency_ms,
                "timestamp_ns": h.timestamp_ns,
            }

        return {
            "schema_version": self.schema_version,
            "identity": {
                "experiment_id": self.identity.experiment_id,
                "campaign_id": self.identity.campaign_id,
                "block_id": self.identity.block_id,
                "session_id": self.identity.session_id,
                "episode_id": self.identity.episode_id,
                "round_id": self.identity.round_id,
                "observation_id": self.identity.observation_id,
            },
            "time": {
                "host_monotonic_ns": self.host_monotonic_ns,
                "camera_device_timestamp": self.camera.device_timestamp_s,
                "left_hand_timestamp_ns": self.left_hand.timestamp_ns if self.left_hand else None,
                "right_hand_timestamp_ns": self.right_hand.timestamp_ns
                if self.right_hand
                else None,
                "clock_sync_report": self.clock_sync.to_dict(),
                "maximum_skew_ms": self.clock_sync.max_skew_ms,
            },
            "camera": {
                "camera_id": self.camera.camera_id,
                "camera_snapshot_hash": self.camera.camera_snapshot_hash,
                "color_artifact": self.camera.color_artifact,
                "depth_artifact": self.camera.depth_artifact,
                "frame_age_ms": self.camera.frame_age_ms,
                "rgb_depth_skew_ms": self.camera.rgb_depth_skew_ms,
                "depth_valid_ratio": self.camera.depth_valid_ratio,
                "exposure": self.camera.exposure,
                "sensor_health": self.camera.sensor_health,
            },
            "left_hand": _hand(self.left_hand),
            "right_hand": _hand(self.right_hand),
            "derived": self.derived,
            "trace_id": self.trace_id,
            "action_id": self.action_id,
            "receipt_id": self.receipt_id,
            "policy_hash": self.policy_hash,
            "candidate_hash": self.candidate_hash,
            "regime_id": self.regime_id,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PhysicalObservationBundle:
        ident = record.get("identity") or {}
        time_block = record.get("time") or {}
        cam = record.get("camera") or {}
        sync = time_block.get("clock_sync_report") or {}

        def _hand(raw: dict[str, Any] | None) -> HandObservation | None:
            if raw is None:
                return None
            return HandObservation(
                body_id=str(raw.get("body_id") or ""),
                body_snapshot_hash=str(raw.get("body_snapshot_hash") or ""),
                position=dict(raw.get("position") or {}),
                force=raw.get("force"),
                current_ma=raw.get("current_ma"),
                temperature_c=raw.get("temperature_c"),
                status=raw.get("status"),
                transport_latency_ms=raw.get("transport_latency_ms"),
                timestamp_ns=raw.get("timestamp_ns"),
            )

        return cls(
            identity=EvidenceIdentity(
                experiment_id=ident.get("experiment_id"),
                campaign_id=ident.get("campaign_id"),
                block_id=ident.get("block_id"),
                session_id=ident.get("session_id"),
                episode_id=ident.get("episode_id"),
                round_id=ident.get("round_id"),
                observation_id=ident.get("observation_id"),
            ),
            host_monotonic_ns=int(time_block.get("host_monotonic_ns") or 0),
            clock_sync=ClockSyncReport(
                samples=int(sync.get("samples") or 0),
                max_skew_ms=float(sync.get("max_skew_ms") or float("nan")),
                mean_skew_ms=float(sync.get("mean_skew_ms") or float("nan")),
                reference=str(sync.get("reference") or "host_monotonic_ns"),
            ),
            camera=CameraObservation(
                camera_id=str(cam.get("camera_id") or ""),
                camera_snapshot_hash=str(cam.get("camera_snapshot_hash") or ""),
                color_artifact=cam.get("color_artifact"),
                depth_artifact=cam.get("depth_artifact"),
                frame_age_ms=cam.get("frame_age_ms"),
                rgb_depth_skew_ms=cam.get("rgb_depth_skew_ms"),
                depth_valid_ratio=cam.get("depth_valid_ratio"),
                exposure=cam.get("exposure"),
                sensor_health=str(cam.get("sensor_health") or "unknown"),
                device_timestamp_s=time_block.get("camera_device_timestamp"),
            ),
            left_hand=_hand(record.get("left_hand")),
            right_hand=_hand(record.get("right_hand")),
            derived=dict(record.get("derived") or {}),
            trace_id=record.get("trace_id"),
            action_id=record.get("action_id"),
            receipt_id=record.get("receipt_id"),
            policy_hash=record.get("policy_hash"),
            candidate_hash=record.get("candidate_hash"),
            regime_id=record.get("regime_id"),
            schema_version=str(record.get("schema_version") or ""),
        )
