"""Session envelope backfill tests (distill → RegimeMatcher link).

The gap this covers: distillation stored memories with regime metadata
but never wrote ``memory_applicability`` rows, so the matcher answered
"not applicable" for every real memory.  These tests pin the honest
semantics of the backfill: measured-only features, monotone widening,
practice_id idempotency, and regime-differentiated matching.
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.memory.regime import (
    ApplicabilityStore,
    OperatingRegime,
    RegimeMatcher,
    empty_regime,
)
from rosclaw.memory.regime.session_envelopes import (
    BUILDER_TAG,
    build_session_envelopes,
    features_from_samples,
)
from rosclaw.memory.regime.session_samples import extract_samples
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore

T0 = 1_785_300_000.0


def _telemetry_event(temp: float, ts: float) -> dict:
    return {
        "event_type": "health_check",
        "timestamp_ns": int(ts * 1e9),
        "payload": {"right": {"temperature_c": {"index": temp, "thumb": temp - 1}}},
    }


def _round_event(index: int, ts: float, result: str = "win") -> dict:
    return {
        "event_type": "rps.stress.round.resolved",
        "event_id": f"evt_{index:04d}",
        "timestamp_ns": int(ts * 1e9),
        "payload": {
            "round": {
                "round_id": f"stress_{index:06d}",
                "started_at": ts,
                "ended_at": ts + 1.5,
                "result": result,
                "robot_gesture_verified": result != "invalid",
                "robot_gesture_failure_reason": "joint_not_reached"
                if result == "invalid"
                else None,
            }
        },
    }


def _write_session(
    root: Path,
    practice_id: str,
    *,
    temp_start: float,
    temp_end: float,
    rounds: int = 10,
    invalid_every: int = 5,
) -> Path:
    session = root / "sessions" / practice_id
    (session / "raw").mkdir(parents=True)
    events = [
        {
            "event_type": "practice.session_started",
            "practice_id": practice_id,
            "session_id": f"sess_{practice_id}",
            "robot_id": "rh56_rps_robot",
            "body_id": "rh56_right_01",
            "timestamp_ns": int(T0 * 1e9),
            "payload": {},
        }
    ]
    for i in range(rounds):
        ts = T0 + 60 + i * 8.0
        temp = temp_start + (temp_end - temp_start) * (i / max(1, rounds - 1))
        events.append(_telemetry_event(temp, ts))
        result = "invalid" if (i + 1) % invalid_every == 0 else "win"
        events.append(_round_event(i + 1, ts, result))
    with (session / "raw" / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return session


def _seed_memory(
    store: InMemoryKnowledgeStore, practice_id: str, *, memory_id: str = "mem_x"
) -> None:
    store.insert(
        "memory_items",
        {
            "id": memory_id,
            "memory_type": "failure",
            "practice_id": practice_id,
            "body_id": "rh56_right_01",
            "task_id": "rh56_rps",
            "gesture_name": "rock",
            "failure_type": "joint_not_reached",
            "outcome": "failure",
            "status": "active",
        },
    )


def test_features_from_samples_measured_only() -> None:
    events = [
        _telemetry_event(40.0, T0),
        _round_event(1, T0, "win"),
        _telemetry_event(46.0, T0 + 120),
        _round_event(2, T0 + 120, "invalid"),
    ]
    samples = extract_samples(events)
    features = features_from_samples(samples)
    assert features.temperature_min == 40.0
    assert features.temperature_max == 46.0
    assert features.temperature_slope is not None and features.temperature_slope > 0
    assert features.invalid_rate == 0.5
    assert features.rounds == 2
    # position_error_p95 is never invented.
    assert "position_error_p95" not in features.measured


def test_features_missing_temperature_stays_unbounded() -> None:
    events = [_round_event(1, T0, "win"), _round_event(2, T0 + 8, "win")]
    features = features_from_samples(extract_samples(events))
    assert features.temperature_min is None
    assert "temperature_c" not in features.measured


def test_backfill_creates_observed_envelope(tmp_path: Path) -> None:
    store = InMemoryKnowledgeStore()
    session = _write_session(tmp_path, "prac_hot", temp_start=46.0, temp_end=52.0)
    _seed_memory(store, "prac_hot")

    result = build_session_envelopes(session, store)
    assert result["ok"] and result["envelopes_created"] == 1

    envelopes = ApplicabilityStore(store).for_memory("mem_x")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env.envelope_type == "observed"
    assert env.reason == BUILDER_TAG
    assert env.temperature_min == 46.0
    assert env.temperature_max == 52.0
    assert env.evidence_count == 1
    assert env.failure_count == 1
    assert env.evidence_refs == ["prac_hot"]
    assert env.gestures == ["rock"]
    assert env.failure_types == ["joint_not_reached"]
    assert "temperature_c" in env.required_features


def test_backfill_independent_envelopes_per_memory(tmp_path: Path) -> None:
    store = InMemoryKnowledgeStore()
    hot = _write_session(tmp_path, "prac_hot", temp_start=46.0, temp_end=52.0)
    cold = _write_session(tmp_path, "prac_cold", temp_start=34.0, temp_end=38.0, invalid_every=3)
    _seed_memory(store, "prac_hot", memory_id="mem_hot")
    _seed_memory(store, "prac_cold", memory_id="mem_cold")

    build_session_envelopes(hot, store)
    build_session_envelopes(cold, store)

    hot_env = ApplicabilityStore(store).for_memory("mem_hot")[0]
    cold_env = ApplicabilityStore(store).for_memory("mem_cold")[0]
    assert (hot_env.temperature_min, hot_env.temperature_max) == (46.0, 52.0)
    assert (cold_env.temperature_min, cold_env.temperature_max) == (34.0, 38.0)
    assert cold_env.evidence_refs == ["prac_cold"]


def test_backfill_idempotent_per_session(tmp_path: Path) -> None:
    store = InMemoryKnowledgeStore()
    session = _write_session(tmp_path, "prac_hot", temp_start=46.0, temp_end=52.0)
    _seed_memory(store, "prac_hot")
    build_session_envelopes(session, store)
    again = build_session_envelopes(session, store)
    assert again["envelopes_created"] == 0
    assert again["envelopes_already_counted"] == 1
    env = ApplicabilityStore(store).for_memory("mem_x")[0]
    assert env.evidence_count == 1


def _regime(temp: float, failure_rate: float) -> OperatingRegime:
    regime = empty_regime(
        robot_id="rh56_rps_robot", body_id="rh56_right_01", task_id="rh56_rps", now=T0
    )
    regime.temperature_c = temp
    regime.temperature_slope_c_per_min = 0.1
    regime.session_elapsed_sec = 300.0
    regime.cumulative_action_count = 40
    regime.recent_failure_rate = failure_rate
    regime.gesture_name = "rock"
    return regime


def test_backfill_discovers_reconfirmed_memory_via_evidence(tmp_path: Path) -> None:
    """The distiller's MERGE path keeps the ORIGINATING session's
    practice_id on the memory row; re-confirmations in later sessions are
    only visible through memory_evidence.source_event_id.  The backfill
    must attribute those sessions to the memory's envelope too."""
    store = InMemoryKnowledgeStore()
    cold = _write_session(tmp_path, "prac_cold", temp_start=34.0, temp_end=38.0)
    hot = _write_session(tmp_path, "prac_hot", temp_start=46.0, temp_end=52.0)
    _seed_memory(store, "prac_cold", memory_id="mem_shared")
    # The hot session re-confirmed the same memory: distill merged into it
    # and wrote an evidence row pointing at a HOT-session event.
    hot_event_ids = [
        json.loads(line)["event_id"]
        for line in (hot / "raw" / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event_id")
    ]
    store.insert(
        "memory_evidence",
        {
            "id": "evd_hot_1",
            "memory_id": "mem_shared",
            "evidence_type": "practice_event",
            "source_event_id": hot_event_ids[-1],
            "confidence": 1.0,
        },
    )

    build_session_envelopes(cold, store)
    result = build_session_envelopes(hot, store)
    assert result["envelopes_merged"] == 1

    env = ApplicabilityStore(store).for_memory("mem_shared")[0]
    assert env.evidence_count == 2
    assert env.temperature_min == 34.0 and env.temperature_max == 52.0
    assert set(env.evidence_refs) == {"prac_cold", "prac_hot"}


def test_matcher_prefers_regime_matching_envelope(tmp_path: Path) -> None:
    """The self-loop payoff: one memory observed in BOTH a cold and a hot
    session must score higher in the regime it was observed in."""
    store = InMemoryKnowledgeStore()
    cold = _write_session(tmp_path, "prac_cold", temp_start=34.0, temp_end=38.0)
    hot = _write_session(tmp_path, "prac_hot", temp_start=46.0, temp_end=52.0)
    # One memory id observed in both sessions: seed the row once per
    # practice but with the same id, so the second backfill MERGES.
    store.insert(
        "memory_items",
        {
            "id": "mem_shared",
            "memory_type": "failure",
            "practice_id": "prac_cold",
            "body_id": "rh56_right_01",
            "failure_type": "joint_not_reached",
            "outcome": "failure",
            "status": "active",
        },
    )
    build_session_envelopes(cold, store)
    # Simulate the same memory surfacing again in the hot session: the
    # production distiller keys memories by content hash, so re-distilling
    # an identical failure UPDATES the same memory id and re-points its
    # practice_id.  Mirror that here.
    store.insert(
        "memory_items",
        {
            "id": "mem_shared",
            "memory_type": "failure",
            "practice_id": "prac_hot",
            "body_id": "rh56_right_01",
            "failure_type": "joint_not_reached",
            "outcome": "failure",
            "status": "active",
        },
    )
    result = build_session_envelopes(hot, store)
    assert result["envelopes_merged"] == 1

    env = ApplicabilityStore(store).for_memory("mem_shared")[0]
    assert env.evidence_count == 2
    assert env.temperature_min == 34.0 and env.temperature_max == 52.0
    assert set(env.evidence_refs) == {"prac_cold", "prac_hot"}

    matcher = RegimeMatcher()
    envelopes = ApplicabilityStore(store).for_memory("mem_shared")
    cold_match = matcher.match("mem_shared", envelopes, _regime(36.0, 0.2))
    hot_match = matcher.match("mem_shared", envelopes, _regime(51.0, 0.2))
    far_match = matcher.match("mem_shared", envelopes, _regime(70.0, 0.2))
    assert cold_match.score > 0 and hot_match.score > 0
    assert far_match.score < cold_match.score  # out-of-envelope penalized
    # A memory with no envelopes at all stays inert (the pre-fix behavior).
    no_env = matcher.match("mem_nothing", [], _regime(36.0, 0.2))
    assert not no_env.applicable
