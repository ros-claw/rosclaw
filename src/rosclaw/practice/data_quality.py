"""Data Quality Gate for physical sessions (Physical Evolution Lab §6.3).

Runs before Distill / Train / Promotion: a session must EARN each use.

Three-state honesty per check:

* ``pass``   — measured and within contract;
* ``fail``   — measured and violated;
* ``unknown``— the session never recorded what this check needs
  (unmeasurable is disclosed in ``missing_ranges``, never silently
  treated as pass).

The gate is READ-ONLY: it never modifies sessions, manifests, or stores.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_VERSION = "rosclaw.data_quality.v1"

# Thresholds (declared, overridable) — v3 §6.5 "configured threshold".
# The camera/hand alignment default is principled, not tuned-to-pass: the
# current corpus records hand telemetry at ~2.4 Hz (median period 0.42 s),
# so the nearest-state alignment floor is ~210 ms; 500 ms is ~2.4× that
# floor.  Tighter budgets belong to the promotion path via the parameter.
DEFAULT_MAX_CAMERA_HAND_SKEW_MS = 500.0
DEFAULT_MAX_STALE_FRAME_AGE_S = 0.5

# Check criticality (v3 §6.3 verdict semantics):
# * CRITICAL — the session's basic integrity is broken; unusable for
#   anything (not even failure research needs corrupted evidence).
# * LINKAGE/FRESHNESS — evidence gaps that block training and promotion
#   but still allow memory/failure research (§6.3: 低质量 Session 可以
#   用于故障研究、写 Failure Memory、禁止用于训练、禁止作为 Promotion
#   Positive Evidence).
CRITICAL_CHECKS = {
    "rgb_completeness",
    "depth_completeness",
    "rh56_state_completeness",
    "timestamp_monotonicity",
    "duplicate_event_ids",
    "nan_inf",
    "frame_reuse",
    "body_hash_consistency",
    "policy_version_switch",
}
LINKAGE_CHECKS = {"receipt_linkage", "trace_linkage", "missing_artifacts"}
FRESHNESS_CHECKS = {"camera_hand_alignment_skew", "rgb_depth_skew", "stale_observation"}
PROMOTION_MUST_MEASURE = CRITICAL_CHECKS | LINKAGE_CHECKS


@dataclass
class CheckResult:
    status: str  # pass | fail | unknown
    detail: str = ""
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "detail": self.detail}
        if self.value is not None:
            out["value"] = round(self.value, 4)
        return out


@dataclass
class DataQualityReport:
    """§6.3 output contract."""

    session_dir: str
    practice_id: str | None
    passed: bool
    usable_for_memory: bool
    usable_for_training: bool
    usable_for_promotion: bool
    reasons: list[str]
    missing_ranges: list[str]
    degraded_sensors: list[str]
    checks: dict[str, CheckResult] = field(default_factory=dict)
    gate_version: str = GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": self.gate_version,
            "session_dir": self.session_dir,
            "practice_id": self.practice_id,
            "data_quality": {
                "passed": self.passed,
                "usable_for_memory": self.usable_for_memory,
                "usable_for_training": self.usable_for_training,
                "usable_for_promotion": self.usable_for_promotion,
                "reasons": self.reasons,
                "missing_ranges": self.missing_ranges,
                "degraded_sensors": self.degraded_sensors,
            },
            "checks": {name: result.to_dict() for name, result in self.checks.items()},
        }


def _load_events(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "raw" / "events.jsonl"
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _iter_numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_numbers(item)


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))
    return ordered[index]


def run_data_quality_gate(
    session_dir: str | Path,
    *,
    max_camera_hand_skew_ms: float = DEFAULT_MAX_CAMERA_HAND_SKEW_MS,
    max_stale_frame_age_s: float = DEFAULT_MAX_STALE_FRAME_AGE_S,
) -> DataQualityReport:
    """Run the §6.3 gate over one practice session directory (READ-ONLY)."""
    session_dir = Path(session_dir)
    events = _load_events(session_dir)
    checks: dict[str, CheckResult] = {}
    reasons: list[str] = []
    missing: list[str] = []
    degraded: list[str] = []

    if not events:
        return DataQualityReport(
            session_dir=str(session_dir),
            practice_id=None,
            passed=False,
            usable_for_memory=False,
            usable_for_training=False,
            usable_for_promotion=False,
            reasons=["no events.jsonl or empty session"],
            missing_ranges=["events"],
            degraded_sensors=[],
            checks={"events_present": CheckResult("fail", "no events")},
        )

    practice_id = events[0].get("practice_id")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_type[event.get("event_type") or "unknown"].append(event)

    frames = by_type.get("frame_event", [])
    telemetry = by_type.get("rps.telemetry", [])
    health = by_type.get("health_check", [])

    # 1) RGB completeness ------------------------------------------------
    if frames:
        missing_numbers = sum(1 for e in frames if _payload(e).get("frame_number") is None)
        checks["rgb_completeness"] = CheckResult(
            "pass" if missing_numbers == 0 else "fail",
            f"{len(frames)} frame events, {missing_numbers} missing frame_number",
            1.0 - missing_numbers / len(frames),
        )
    else:
        checks["rgb_completeness"] = CheckResult("fail", "no frame_event recorded")
        reasons.append("no camera frames")

    # 2) Depth completeness ----------------------------------------------
    if frames:
        with_depth = sum(1 for e in frames if _payload(e).get("has_depth"))
        ratio = with_depth / len(frames)
        checks["depth_completeness"] = CheckResult(
            "pass" if ratio >= 0.99 else "fail",
            f"{with_depth}/{len(frames)} frames report depth",
            ratio,
        )
        if ratio < 0.99:
            degraded.append("depth")
    else:
        checks["depth_completeness"] = CheckResult("fail", "no frames")

    # 3) RH56 state completeness ------------------------------------------
    if telemetry:
        bad = 0
        for event in telemetry:
            payload = _payload(event)
            for side in ("left", "right"):
                side_payload = payload.get(side) or {}
                angles = side_payload.get("angle_actual")
                if not isinstance(angles, dict) or not angles:
                    bad += 1
        checks["rh56_state_completeness"] = CheckResult(
            "pass" if bad == 0 else "fail",
            f"{len(telemetry)} telemetry events, {bad} side-readings missing angle_actual",
            1.0 - bad / (2 * len(telemetry)),
        )
        if bad:
            degraded.append("rh56_state")
    else:
        checks["rh56_state_completeness"] = CheckResult("fail", "no rps.telemetry recorded")
        reasons.append("no hand telemetry")

    # 4) Timestamp monotonicity (per event type) --------------------------
    inversions = 0
    for group in by_type.values():
        stamps: list[float] = [
            float(e["timestamp_ns"])
            for e in group
            if isinstance(e.get("timestamp_ns"), (int, float))
        ]
        inversions += sum(1 for a, b in zip(stamps, stamps[1:], strict=False) if b < a)
    checks["timestamp_monotonicity"] = CheckResult(
        "pass" if inversions == 0 else "fail", f"{inversions} timestamp inversions"
    )

    # 5) RGB-depth skew ----------------------------------------------------
    if any("rgb_depth_skew_ms" in _payload(e) for e in frames):
        skews = [
            float(_payload(e)["rgb_depth_skew_ms"])
            for e in frames
            if "rgb_depth_skew_ms" in _payload(e)
        ]
        p99 = _p99(skews)
        checks["rgb_depth_skew"] = CheckResult(
            "pass" if (p99 or 0) <= 50.0 else "fail", f"p99 {p99:.1f} ms", p99
        )
    else:
        checks["rgb_depth_skew"] = CheckResult(
            "unknown", "sessions do not record per-frame rgb/depth skew"
        )
        missing.append("rgb_depth_skew_ms")

    # 6) Camera-hand alignment skew ---------------------------------------
    if frames and telemetry:
        tele_stamps = sorted(
            float(_payload(e)["timestamp"]) for e in telemetry if "timestamp" in _payload(e)
        )
        skews_ms: list[float] = []
        if tele_stamps:
            import bisect

            for event in frames:
                cam_ts = _payload(event).get("camera_frame_ts")
                if not isinstance(cam_ts, (int, float)):
                    continue
                pos = bisect.bisect_left(tele_stamps, cam_ts)
                nearest = min(
                    (
                        abs(cam_ts - tele_stamps[j])
                        for j in (pos - 1, pos)
                        if 0 <= j < len(tele_stamps)
                    ),
                    default=None,
                )
                if nearest is not None:
                    skews_ms.append(nearest * 1000.0)
        if skews_ms:
            p99 = _p99(skews_ms)
            periods = [b - a for a, b in zip(tele_stamps, tele_stamps[1:], strict=False)]
            median_period_ms = sorted(periods)[len(periods) // 2] * 1000.0 if periods else 0.0
            checks["camera_hand_alignment_skew"] = CheckResult(
                "pass" if (p99 or 0) <= max_camera_hand_skew_ms else "fail",
                f"p99 {p99:.1f} ms (threshold {max_camera_hand_skew_ms} ms; "
                f"telemetry period ~{median_period_ms:.0f} ms → sampling floor ~{median_period_ms / 2:.0f} ms)",
                p99,
            )
        else:
            checks["camera_hand_alignment_skew"] = CheckResult(
                "unknown", "no aligned camera/hand timestamp pairs"
            )
            missing.append("camera_hand_alignment")
    else:
        checks["camera_hand_alignment_skew"] = CheckResult(
            "unknown", "need both frames and telemetry"
        )
        missing.append("camera_hand_alignment")

    # 7) Duplicate event IDs ----------------------------------------------
    id_counts = Counter(e.get("event_id") for e in events if e.get("event_id"))
    duplicates = sum(1 for count in id_counts.values() if count > 1)
    checks["duplicate_event_ids"] = CheckResult(
        "pass" if duplicates == 0 else "fail", f"{duplicates} duplicated event ids"
    )

    # 8) Missing artifacts --------------------------------------------------
    referenced = [
        session_dir / str(_payload(e)["keyframe_path"])
        for e in frames
        if _payload(e).get("keyframe_path")
        and not str(_payload(e)["keyframe_path"]).startswith("/")
    ] + [
        Path(str(_payload(e)["keyframe_path"]))
        for e in frames
        if str(_payload(e).get("keyframe_path") or "").startswith("/")
    ]
    missing_artifacts = sum(1 for path in referenced if not path.is_file())
    checks["missing_artifacts"] = CheckResult(
        "pass" if missing_artifacts == 0 else "fail",
        f"{len(referenced)} referenced keyframes, {missing_artifacts} missing on disk",
    )

    # 9) Receipt linkage (action_id present on actionable events) ----------
    actionable = [
        e
        for e in events
        if (e.get("event_type") or "").startswith(("rps.gesture", "rps.stress.round"))
    ]
    if actionable:
        linked = sum(1 for e in actionable if e.get("action_id"))
        ratio = linked / len(actionable)
        checks["receipt_linkage"] = CheckResult(
            "pass" if ratio >= 0.99 else "fail",
            f"{linked}/{len(actionable)} actionable events carry action_id",
            ratio,
        )
    else:
        checks["receipt_linkage"] = CheckResult("unknown", "no actionable events recorded")
        missing.append("receipt_linkage")

    # 10) Body hash consistency --------------------------------------------
    bodies = {e.get("body_id") for e in events if e.get("body_id")}
    checks["body_hash_consistency"] = CheckResult(
        "pass" if len(bodies) <= 2 else "fail",
        f"body_ids in session: {sorted(str(b) for b in bodies)}",
    )

    # 11) Trace linkage ------------------------------------------------------
    traced = sum(1 for e in events if e.get("trace_id"))
    ratio = traced / len(events)
    checks["trace_linkage"] = CheckResult(
        "pass" if ratio >= 0.99 else "fail",
        f"{traced}/{len(events)} events carry trace_id",
        ratio,
    )

    # 12) NaN / Inf ------------------------------------------------------------
    nonfinite = 0
    for event in events:
        for number in _iter_numbers(_payload(event)):
            if not math.isfinite(number):
                nonfinite += 1
    checks["nan_inf"] = CheckResult(
        "pass" if nonfinite == 0 else "fail", f"{nonfinite} non-finite payload numbers"
    )

    # 13) Frame reuse -----------------------------------------------------------
    numbers = [
        _payload(e).get("frame_number")
        for e in frames
        if _payload(e).get("frame_number") is not None
    ]
    reused = len(numbers) - len(set(numbers))
    checks["frame_reuse"] = CheckResult(
        "pass" if reused == 0 else "fail", f"{reused} reused frame numbers"
    )

    # 14) Stale observation ------------------------------------------------------
    if health:
        worst_age = 0.0
        worst_streak = 0
        for event in health:
            camera = _payload(event).get("camera") or {}
            age = camera.get("last_frame_age_s")
            if isinstance(age, (int, float)):
                worst_age = max(worst_age, float(age))
            streak = camera.get("empty_streak")
            if isinstance(streak, (int, float)):
                worst_streak = max(worst_streak, int(streak))
        ok = worst_age <= max_stale_frame_age_s and worst_streak <= 3
        checks["stale_observation"] = CheckResult(
            "pass" if ok else "fail",
            f"worst frame age {worst_age:.2f}s, worst empty streak {worst_streak}",
            worst_age,
        )
        if not ok:
            degraded.append("camera_freshness")
    else:
        checks["stale_observation"] = CheckResult("unknown", "no health_check recorded")
        missing.append("stale_observation")

    # 15) Policy version switches inside episode ---------------------------------
    per_episode: dict[str, set] = defaultdict(set)
    for event in events:
        episode = event.get("episode_id")
        policy = event.get("policy_id")
        if episode and policy:
            per_episode[str(episode)].add(str(policy))
    switches = sum(1 for policies in per_episode.values() if len(policies) > 1)
    checks["policy_version_switch"] = CheckResult(
        "pass" if switches == 0 else "fail",
        f"{switches} episodes with >1 policy_id",
    )

    # ------------------------------------------------------------------ verdict
    failed = [name for name, result in checks.items() if result.status == "fail"]
    unknown = [name for name, result in checks.items() if result.status == "unknown"]

    critical_failed = [name for name in failed if name in CRITICAL_CHECKS]
    linkage_failed = [name for name in failed if name in LINKAGE_CHECKS]
    freshness_failed = [name for name in failed if name in FRESHNESS_CHECKS]
    promotion_unknown = [name for name in unknown if name in PROMOTION_MUST_MEASURE]
    training_unknown = [name for name in unknown if name in FRESHNESS_CHECKS]

    # §6.3 verdict semantics: a broken-integrity session is unusable,
    # period; evidence gaps (linkage/freshness) still allow memory and
    # failure research but never training or promotion positive evidence.
    usable_memory = not critical_failed
    usable_training = not failed and not training_unknown
    usable_promotion = not failed and not promotion_unknown
    if critical_failed:
        reasons.append(f"integrity failures (unusable): {', '.join(critical_failed)}")
    if linkage_failed:
        reasons.append(f"evidence linkage gaps (no train/promote): {', '.join(linkage_failed)}")
    if freshness_failed:
        reasons.append(f"freshness/sync gaps (no train/promote): {', '.join(freshness_failed)}")
    if training_unknown:
        reasons.append(f"training blocked by unmeasurable: {', '.join(training_unknown)}")
    if promotion_unknown:
        reasons.append(f"promotion blocked by unmeasurable: {', '.join(promotion_unknown)}")

    return DataQualityReport(
        session_dir=str(session_dir),
        practice_id=practice_id,
        passed=not failed,
        usable_for_memory=usable_memory,
        usable_for_training=usable_training,
        usable_for_promotion=usable_promotion,
        reasons=reasons,
        missing_ranges=missing,
        degraded_sensors=degraded,
        checks=checks,
    )
