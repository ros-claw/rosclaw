"""Extract regime telemetry samples from practice session events (v4 §4.4).

Reads a session's ``raw/events.jsonl`` and builds the round-level
:class:`TelemetrySample` stream the regime builder consumes.  Features the
session never recorded stay ``None`` — the extractor never invents
position-error or time-to-reach values the hardware did not report
(unknown is not wildcard).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import TelemetrySample

ROUND_RESOLVED = "rps.stress.round.resolved"
HEALTH_CHECK = "health_check"


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload or {}


def _event_time(event: dict[str, Any]) -> float:
    ts_ns = event.get("timestamp_ns")
    if isinstance(ts_ns, (int, float)) and ts_ns:
        return float(ts_ns) / 1e9
    started = _payload(event).get("round", {}).get("started_at")
    if isinstance(started, (int, float)):
        return float(started)
    return 0.0


def _hand_temperature(payload: dict[str, Any], hand: str) -> float | None:
    side = payload.get(hand) or {}
    temps = side.get("temperature_c")
    if not isinstance(temps, dict) or not temps:
        return None
    values = [float(v) for v in temps.values() if isinstance(v, (int, float))]
    return max(values) if values else None


def _temperature_near(health_checks: list[tuple[float, float]], timestamp: float) -> float | None:
    if not health_checks:
        return None
    return min(health_checks, key=lambda pair: abs(pair[0] - timestamp))[1]


def load_session_events(session_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(session_dir) / "raw" / "events.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no events.jsonl in session dir: {session_dir}")
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def extract_samples(
    events: list[dict[str, Any]],
    *,
    hand: str = "right",
) -> list[TelemetrySample]:
    """Round-level samples for one hand from RPS practice events."""
    health_checks: list[tuple[float, float]] = []
    for event in events:
        if event.get("event_type") != HEALTH_CHECK:
            continue
        temp = _hand_temperature(_payload(event), hand)
        if temp is not None:
            health_checks.append((_event_time(event), temp))
    health_checks.sort()

    rounds: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != ROUND_RESOLVED:
            continue
        payload = _payload(event)
        round_info = payload.get("round") or {}
        started = round_info.get("started_at")
        if not isinstance(started, (int, float)):
            continue
        rounds.append(
            {
                "event_id": event.get("event_id"),
                "started_at": float(started),
                "ended_at": round_info.get("ended_at"),
                "result": round_info.get("result"),
                "verified": round_info.get("robot_gesture_verified"),
                "failure_reason": round_info.get("robot_gesture_failure_reason"),
            }
        )
    rounds.sort(key=lambda row: row["started_at"])

    samples: list[TelemetrySample] = []
    previous_started: float | None = None
    for row in rounds:
        started = row["started_at"]
        result = str(row.get("result") or "").lower()
        failure = bool(row.get("failure_reason")) or row.get("verified") is False
        interval = started - previous_started if previous_started is not None else None
        samples.append(
            TelemetrySample(
                timestamp=started,
                temperature_c=_temperature_near(health_checks, started),
                invalid=(result == "invalid"),
                failure=failure,
                action_count=1,
                gesture_interval_sec=interval,
                evidence_ref=row.get("event_id"),
            )
        )
        previous_started = started
    return samples


def latest_session_dir(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    sessions = root / "sessions"
    if not sessions.is_dir():
        raise FileNotFoundError(f"no sessions/ under data root: {root}")
    candidates = sorted(
        path
        for path in sessions.iterdir()
        if path.is_dir() and (path / "raw" / "events.jsonl").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no session with raw/events.jsonl under {sessions}")
    return candidates[-1]


def resolve_session_dir(data_root: str | Path, practice_id: str) -> Path:
    root = Path(data_root).expanduser()
    candidate = root / "sessions" / practice_id
    if not candidate.is_dir():
        raise FileNotFoundError(f"practice session not found: {candidate}")
    return candidate
