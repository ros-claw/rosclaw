"""Multimodal synchronization layer (Physical Evolution Lab §6.4, PR-PE-2).

Aligns every camera frame to the NEAREST left/right hand telemetry (v3
§3.3: 将相机数据对齐到最近的 RH56 状态，而不是要求两者同频) and emits
:class:`PhysicalObservationBundle` records — the Synchronized Layer
between Raw and Episode.

Honesty rules inherited from PR-PE-1:

* What the raw session never recorded (per-frame rgb/depth skew, depth
  valid ratio, per-frame artifacts) stays ``None`` — the bundle's
  ``validate()`` + the sync stats disclose it, nothing is invented;
* every bundle gets a content-addressed ``observation_id`` (practice_id
  × frame_number) — replay never relies on directory names;
* transport latency is computed from the telemetry's own host/device
  timestamp pair when present.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .physical_observation import (
    CameraObservation,
    ClockSyncReport,
    EvidenceIdentity,
    HandObservation,
    PhysicalObservationBundle,
    canonical_hash,
)

SYNC_VERSION = "rosclaw.multimodal_sync.v1"


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class SyncStats:
    frames_total: int
    frames_aligned: int
    keyframes: int
    mean_alignment_skew_ms: float
    max_alignment_skew_ms: float
    bundles_valid: int
    bundles_with_violations: int
    unknown_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sync_version": SYNC_VERSION,
            "frames_total": self.frames_total,
            "frames_aligned": self.frames_aligned,
            "keyframes": self.keyframes,
            "mean_alignment_skew_ms": round(self.mean_alignment_skew_ms, 3),
            "max_alignment_skew_ms": round(self.max_alignment_skew_ms, 3),
            "bundles_valid": self.bundles_valid,
            "bundles_with_violations": self.bundles_with_violations,
            "unknown_fields": self.unknown_fields,
        }


@dataclass
class SyncResult:
    session_dir: str
    practice_id: str | None
    bundles: list[PhysicalObservationBundle]
    stats: SyncStats
    clock_sync: ClockSyncReport | None = None


def _hand_snapshot_hash(body_id: str) -> str:
    """Config-hash stand-in until calibration lands (PR-PE-3): binds the
    bundle to the body identity + transport profile we ACTUALLY know —
    never to a calibration we did not measure."""
    return canonical_hash(
        {"body_id": body_id, "transport": "ftdi_serial", "calibration": "unmeasured"},
        prefix="body",
    )


def _to_float_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)):
            out[str(key)] = float(item)
    return out or None


def _hand_observation(
    side_payload: dict[str, Any], body_id: str, host_ts: float | None
) -> HandObservation | None:
    position = _to_float_map(side_payload.get("angle_actual"))
    if position is None:
        return None
    device_ts = side_payload.get("timestamp")
    latency_ms = None
    if isinstance(host_ts, (int, float)) and isinstance(device_ts, (int, float)):
        latency_ms = abs(host_ts - device_ts) * 1000.0
    return HandObservation(
        body_id=body_id,
        body_snapshot_hash=_hand_snapshot_hash(body_id),
        position=position,
        force=_to_float_map(side_payload.get("force_act")),
        current_ma=_to_float_map(side_payload.get("current_ma")),
        temperature_c=_to_float_map(side_payload.get("temperature_c")),
        status=(
            str(side_payload.get("status")) if side_payload.get("status") is not None else None
        ),
        transport_latency_ms=latency_ms,
        timestamp_ns=int(device_ts * 1e9) if isinstance(device_ts, (int, float)) else None,
    )


def build_bundles(
    session_dir: str | Path,
    *,
    camera_id: str,
    experiment_id: str | None = None,
    left_body_id: str = "rh56_left_01",
    right_body_id: str = "rh56_right_01",
) -> SyncResult:
    """Build the Synchronized Layer for one session (READ-ONLY)."""
    session_dir = Path(session_dir)
    events_path = session_dir / "raw" / "events.jsonl"
    events: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        raise ValueError(f"no events in {session_dir}")

    practice_id = events[0].get("practice_id")
    session_id = events[0].get("session_id")
    episode_id = events[0].get("episode_id")

    frames = [e for e in events if e.get("event_type") == "frame_event"]
    telemetry = sorted(
        (e for e in events if e.get("event_type") == "rps.telemetry"),
        key=lambda e: float(_payload(e).get("timestamp") or 0.0),
    )
    tele_stamps = [float(_payload(e).get("timestamp") or 0.0) for e in telemetry]
    health = [e for e in events if e.get("event_type") == "health_check"]
    camera_healthy = all((_payload(e).get("camera") or {}).get("alive", True) for e in health)

    clock_pairs: list[tuple[float, float]] = []
    alignment_skews_ms: list[float] = []
    bundles: list[PhysicalObservationBundle] = []
    keyframes = 0
    for event in frames:
        payload = _payload(event)
        frame_number = payload.get("frame_number")
        cam_ts = payload.get("camera_frame_ts")
        host_ts_ns = payload.get("host_ts_ns") or event.get("timestamp_ns")
        if frame_number is None or not isinstance(cam_ts, (int, float)):
            continue
        if isinstance(host_ts_ns, (int, float)):
            clock_pairs.append((cam_ts, host_ts_ns / 1e9))

        # Nearest telemetry (v3 §3.3: align to nearest, never resample).
        left_hand = right_hand = None
        if tele_stamps:
            pos = bisect.bisect_left(tele_stamps, cam_ts)
            best = min(
                (j for j in (pos - 1, pos) if 0 <= j < len(tele_stamps)),
                key=lambda j: abs(tele_stamps[j] - cam_ts),
                default=None,
            )
            if best is not None:
                alignment_skews_ms.append(abs(tele_stamps[best] - cam_ts) * 1000.0)
                tele_payload = _payload(telemetry[best])
                host_ts = tele_payload.get("timestamp")
                left_hand = _hand_observation(tele_payload.get("left") or {}, left_body_id, host_ts)
                right_hand = _hand_observation(
                    tele_payload.get("right") or {}, right_body_id, host_ts
                )

        is_keyframe = bool(payload.get("keyframe"))
        keyframes += int(is_keyframe)
        observation_id = canonical_hash(
            {"practice_id": practice_id, "frame_number": frame_number}, prefix="obs"
        )
        derived: dict[str, Any] = {}
        if payload.get("human_label"):
            derived["gesture"] = payload.get("human_label")
            derived["gesture_confidence"] = payload.get("confidence")
        if payload.get("round_id"):
            derived["round_id"] = payload.get("round_id")

        bundles.append(
            PhysicalObservationBundle(
                identity=EvidenceIdentity(
                    experiment_id=experiment_id,
                    session_id=session_id,
                    episode_id=episode_id,
                    round_id=payload.get("round_id"),
                    observation_id=observation_id,
                ),
                host_monotonic_ns=int(host_ts_ns or 0),
                clock_sync=ClockSyncReport(0, float("nan"), float("nan"), "host_monotonic_ns"),
                camera=CameraObservation(
                    camera_id=camera_id,
                    camera_snapshot_hash=canonical_hash(
                        {"camera_id": camera_id, "pose": "unmeasured"}, prefix="cam"
                    ),
                    color_artifact=payload.get("keyframe_path") if is_keyframe else None,
                    depth_artifact=None,  # per-frame depth is not persisted (disclosed)
                    frame_age_ms=None,  # not recorded per frame (disclosed)
                    rgb_depth_skew_ms=None,  # not recorded (disclosed)
                    depth_valid_ratio=None,  # not recorded (disclosed)
                    exposure=None,
                    sensor_health="ok" if camera_healthy else "degraded",
                    device_timestamp_s=cam_ts,
                ),
                left_hand=left_hand,
                right_hand=right_hand,
                derived=derived,
                trace_id=event.get("trace_id"),
                action_id=event.get("action_id"),
            )
        )

    # One session-level clock-sync report, shared by reference in stats —
    # per-bundle reports stay empty (unknown) rather than pretending
    # per-frame authority they do not have.
    session_clock = ClockSyncReport.from_pairs(clock_pairs, reference="host_monotonic_ns")

    valid = 0
    for bundle in bundles:
        # The identity/observation/hands checks are what sync can vouch
        # for; clock/camera unknowns are corpus-level disclosures.
        violations = [
            v
            for v in bundle.validate()
            if not v.startswith(
                (
                    "clock_sync",
                    "camera.frame_age_ms",
                    "camera.rgb_depth_skew_ms",
                    "camera.depth_valid_ratio",
                    "identity.experiment_id",
                )
            )
        ]
        valid += int(not violations)

    unknown_fields = [
        "camera.frame_age_ms",
        "camera.rgb_depth_skew_ms",
        "camera.depth_valid_ratio",
        "camera.depth_artifact (non-keyframe)",
        "camera.color_artifact (non-keyframe)",
    ]
    stats = SyncStats(
        frames_total=len(frames),
        frames_aligned=sum(1 for b in bundles if b.left_hand or b.right_hand),
        keyframes=keyframes,
        mean_alignment_skew_ms=(
            sum(alignment_skews_ms) / len(alignment_skews_ms)
            if alignment_skews_ms
            else float("nan")
        ),
        max_alignment_skew_ms=max(alignment_skews_ms) if alignment_skews_ms else float("nan"),
        bundles_valid=valid,
        bundles_with_violations=len(bundles) - valid,
        unknown_fields=unknown_fields,
    )
    # Attach the session-level clock report to the stats via the first
    # bundle-free channel: expose it on the result for exporters.
    result = SyncResult(
        session_dir=str(session_dir),
        practice_id=practice_id,
        bundles=bundles,
        stats=stats,
        clock_sync=session_clock,
    )
    return result
