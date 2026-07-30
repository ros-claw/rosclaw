"""PR-PE-1 tests: PhysicalObservationBundle contract + Data Quality Gate
+ evidence link validation (Physical Evolution Lab §5/§6.3)."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.practice.data_quality import run_data_quality_gate
from rosclaw.practice.evidence_links import (
    assess_legacy_sessions,
    validate_legacy_manifest,
    validate_session_links,
)
from rosclaw.practice.physical_observation import (
    CameraObservation,
    ClockSyncReport,
    EvidenceIdentity,
    HandObservation,
    PhysicalObservationBundle,
    canonical_hash,
)

T0 = 1_785_300_000.0


def _bundle(**overrides) -> PhysicalObservationBundle:
    hand = HandObservation(
        body_id="rh56_right_01",
        body_snapshot_hash="body_abc123",
        position={"index": 400.0, "thumb": 250.0},
        force={"index": -20.0},
        current_ma={"index": 30.0},
        temperature_c={"index": 44.0},
        status="ok",
        transport_latency_ms=12.0,
        timestamp_ns=int(T0 * 1e9),
    )
    bundle = PhysicalObservationBundle(
        identity=EvidenceIdentity(
            experiment_id="exp_1",
            session_id="sess_1",
            episode_id="ep_1",
            observation_id="obs_1",
        ),
        host_monotonic_ns=int(T0 * 1e9),
        clock_sync=ClockSyncReport.from_pairs([(T0, T0 + 0.005)], reference="host_monotonic_ns"),
        camera=CameraObservation(
            camera_id="d435i_231122070092",
            camera_snapshot_hash="cam_xyz",
            color_artifact="artifacts/f1.png",
            depth_artifact="artifacts/f1.npy",
            frame_age_ms=33.0,
            rgb_depth_skew_ms=1.1,
            depth_valid_ratio=0.97,
            exposure=8000.0,
            sensor_health="ok",
            device_timestamp_s=T0,
        ),
        left_hand=None,
        right_hand=hand,
        derived={"gesture": "rock", "gesture_confidence": 0.9},
    )
    for key, value in overrides.items():
        object.__setattr__(bundle, key, value)
    return bundle


def test_bundle_roundtrip_and_valid() -> None:
    bundle = _bundle()
    assert bundle.validate() == []
    again = PhysicalObservationBundle.from_record(bundle.to_record())
    assert again.validate() == []
    assert again.identity.observation_id == "obs_1"
    assert again.camera.camera_id == "d435i_231122070092"
    assert again.right_hand is not None and again.right_hand.position["index"] == 400.0


def test_bundle_violations_are_named_not_raised() -> None:
    bundle = _bundle(identity=EvidenceIdentity())
    violations = bundle.validate()
    assert any("identity.observation_id" in v for v in violations)
    assert any("identity.session_id" in v for v in violations)

    no_hands = _bundle(right_hand=None)
    assert any("at least one hand" in v for v in no_hands.validate())

    bad_conf = _bundle(derived={"gesture_confidence": 1.5})
    assert any("gesture_confidence" in v for v in bad_conf.validate())


def test_clock_sync_report() -> None:
    report = ClockSyncReport.from_pairs([(1.0, 1.005), (2.0, 2.020)], reference="host")
    assert report.samples == 2
    assert abs(report.max_skew_ms - 20.0) < 1e-6
    assert abs(report.mean_skew_ms - 12.5) < 1e-6
    empty = ClockSyncReport.from_pairs([], reference="host")
    assert empty.samples == 0


def test_canonical_hash_deterministic() -> None:
    a = canonical_hash({"b": 1, "a": [1, 2]}, prefix="body")
    b = canonical_hash({"a": [1, 2], "b": 1}, prefix="body")
    assert a == b and a.startswith("body_")


# ---------------------------------------------------------------- gate fixtures


def _event(event_type: str, ts_ns: int, payload: dict, **kw) -> dict:
    event = {
        "schema_version": "practice.event.v1",
        "event_id": kw.pop("event_id", f"evt_{event_type}_{ts_ns}"),
        "event_type": event_type,
        "timestamp_ns": ts_ns,
        "practice_id": "prac_t",
        "session_id": "sess_t",
        "episode_id": "ep_t",
        "robot_id": "rh56_rps_robot",
        "body_id": "rh56_rps_robot",
        "trace_id": "trace_t",
        "payload": payload,
    }
    event.update(kw)
    return event


def _write_session(root: Path, name: str, events: list[dict]) -> Path:
    session = root / name
    (session / "raw").mkdir(parents=True)
    with (session / "raw" / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    (session / "manifest.yaml").write_text("ok: true\n")
    (session / "episode.json").write_text("{}\n")
    return session


def _good_events() -> list[dict]:
    events = []
    for i in range(20):
        ts = int((T0 + i * 0.05) * 1e9)
        events.append(
            _event(
                "frame_event",
                ts,
                {
                    "frame_number": i + 1,
                    "camera_frame_ts": T0 + i * 0.05,
                    "host_ts_ns": ts,
                    "has_depth": True,
                },
            )
        )
        events.append(
            _event(
                "rps.telemetry",
                ts + 1,
                {
                    "timestamp": T0 + i * 0.05,
                    "left": {"angle_actual": {"index": 400}, "timestamp": T0},
                    "right": {"angle_actual": {"index": 400}, "timestamp": T0},
                },
            )
        )
    events.append(
        _event(
            "health_check",
            int((T0 + 1.0) * 1e9),
            {"camera": {"alive": True, "last_frame_age_s": 0.03, "empty_streak": 0}, "rounds": 0},
        )
    )
    events.append(
        _event(
            "rps.gesture.executed",
            int((T0 + 1.1) * 1e9),
            {"gesture": "rock"},
            action_id="act_1",
        )
    )
    return events


def test_gate_passes_healthy_session(tmp_path: Path) -> None:
    session = _write_session(tmp_path, "prac_t", _good_events())
    report = run_data_quality_gate(session)
    assert report.passed
    assert report.usable_for_memory
    failed = [n for n, r in report.checks.items() if r.status == "fail"]
    assert failed == []
    # rgb_depth_skew is honestly unknown: this session format never
    # records per-frame rgb/depth skew (unknown ≠ pass for promotion).
    assert report.checks["rgb_depth_skew"].status == "unknown"
    assert not report.usable_for_promotion or "rgb_depth_skew" not in report.missing_ranges


def test_gate_catches_duplicate_ids_and_frame_reuse(tmp_path: Path) -> None:
    events = _good_events()
    dup = dict(events[0])
    events.append(dup)  # same event_id
    reuse = _event(
        "frame_event",
        int((T0 + 2.0) * 1e9),
        {"frame_number": 1, "camera_frame_ts": T0 + 2.0, "has_depth": True},
    )
    events.append(reuse)
    session = _write_session(tmp_path, "prac_bad", events)
    report = run_data_quality_gate(session)
    assert not report.passed
    assert report.checks["duplicate_event_ids"].status == "fail"
    assert report.checks["frame_reuse"].status == "fail"
    assert not report.usable_for_memory
    assert not report.usable_for_training


def test_gate_catches_nonfinite_and_missing_telemetry(tmp_path: Path) -> None:
    events = _good_events()
    events.append(
        _event(
            "rps.telemetry",
            int((T0 + 3.0) * 1e9),
            {"timestamp": T0 + 3.0, "left": {"angle_actual": {"index": float("nan")}}, "right": {}},
        )
    )
    session = _write_session(tmp_path, "prac_nan", events)
    report = run_data_quality_gate(session)
    assert report.checks["nan_inf"].status == "fail"
    assert report.checks["rh56_state_completeness"].status == "fail"
    assert "rh56_state" in report.degraded_sensors


def test_gate_empty_session_fails_loudly(tmp_path: Path) -> None:
    session = tmp_path / "empty"
    session.mkdir()
    report = run_data_quality_gate(session)
    assert not report.passed
    assert "no events" in report.checks["events_present"].detail


def test_session_links_and_legacy_assessment(tmp_path: Path) -> None:
    session = _write_session(tmp_path / "sessions", "prac_t", _good_events())
    links = validate_session_links(session)
    assert links.ok, links.missing_links

    broken = tmp_path / "sessions" / "prac_broken"
    (broken / "raw").mkdir(parents=True)
    (broken / "raw" / "events.jsonl").write_text('{"event_type": "x", "payload": {}}\n')
    broken_links = validate_session_links(broken)
    assert not broken_links.ok
    assert any("manifest.yaml" in link for link in broken_links.missing_links)

    assessment = assess_legacy_sessions(tmp_path / "sessions")
    assert assessment["ok"]
    assert assessment["totals"]["sessions"] == 2
    assert assessment["read_only"] is True


def test_legacy_manifest_validator(tmp_path: Path) -> None:
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_hash": "abc",
                "entries": [
                    {"kind": "canary_session", "practice_id": "prac_1"},
                    {"kind": "canary_session"},  # missing practice_id
                ],
            }
        )
    )
    report = validate_legacy_manifest(manifest)
    assert not report.ok
    assert any("without_practice_id" in link for link in report.missing_links)

    manifest.write_text(json.dumps({"config_hash": "abc", "entries": []}))
    assert validate_legacy_manifest(manifest).ok
