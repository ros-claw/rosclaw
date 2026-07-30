"""PR-PE-2 tests: multimodal sync layer + exporters + replay."""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.practice.multimodal_sync import build_bundles
from rosclaw.practice.sync_export import (
    export_all,
    export_bundles_lerobot,
    export_bundles_parquet,
    replay_observation,
    write_bundles_jsonl,
    write_bundles_mcap,
)

T0 = 1_785_300_000.0


def _event(event_type: str, ts_ns: int, payload: dict, **kw) -> dict:
    event = {
        "schema_version": "practice.event.v1",
        "event_id": f"evt_{event_type}_{ts_ns}_{kw.get('frame_number', '')}",
        "event_type": event_type,
        "timestamp_ns": ts_ns,
        "practice_id": "prac_sync",
        "session_id": "sess_sync",
        "episode_id": "ep_sync",
        "robot_id": "rh56_rps_robot",
        "body_id": "rh56_rps_robot",
        "trace_id": "trace_sync",
        "payload": payload,
    }
    event.update(kw)
    return event


def _write_session(root: Path, frames: int = 30) -> Path:
    session = root / "prac_sync"
    (session / "raw").mkdir(parents=True)
    (session / "keyframes").mkdir(parents=True)
    events = []
    for i in range(frames):
        ts = T0 + i * 0.1
        is_key = i % 10 == 0
        key_path = None
        if is_key:
            key_path = str(session / "keyframes" / f"color_{i:06d}.png")
            Path(key_path).write_bytes(f"png-{i}".encode())
        events.append(
            _event(
                "frame_event",
                int(ts * 1e9),
                {
                    "frame_number": i + 1,
                    "camera_frame_ts": ts,
                    "host_ts_ns": int(ts * 1e9) + 3_000_000,
                    "has_depth": True,
                    "keyframe": is_key,
                    "keyframe_path": key_path,
                    "human_label": "rock" if i % 2 == 0 else "paper",
                    "confidence": 0.9,
                    "frame_number2": None,
                },
                frame_number=i + 1,
            )
        )
        events.append(
            _event(
                "rps.telemetry",
                int((ts + 0.03) * 1e9),
                {
                    "timestamp": ts + 0.03,
                    "left": {
                        "timestamp": ts + 0.028,
                        "angle_actual": {"index": 400 + i, "thumb": 250},
                        "force_act": {"index": -20},
                        "current_ma": {"index": 30},
                        "temperature_c": {"index": 44},
                        "status": "ok",
                    },
                    "right": {
                        "timestamp": ts + 0.029,
                        "angle_actual": {"index": 500 + i, "thumb": 260},
                        "force_act": {"index": -22},
                        "current_ma": {"index": 31},
                        "temperature_c": {"index": 45},
                        "status": "ok",
                    },
                },
            )
        )
    events.append(
        _event(
            "health_check",
            int((T0 + frames * 0.1) * 1e9),
            {"camera": {"alive": True, "last_frame_age_s": 0.03, "empty_streak": 0}},
        )
    )
    with (session / "raw" / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return session


def test_build_bundles_aligns_nearest_telemetry(tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    sync = build_bundles(session, camera_id="d435i_test")
    assert sync.stats.frames_total == 30
    assert sync.stats.frames_aligned == 30  # no silent frame loss
    assert sync.stats.keyframes == 3
    assert len(sync.bundles) == 30

    bundle = sync.bundles[0]
    assert bundle.identity.observation_id
    assert bundle.identity.session_id == "sess_sync"
    assert bundle.camera.device_timestamp_s == T0
    assert bundle.camera.color_artifact and bundle.camera.color_artifact.endswith(
        "color_000000.png"
    )
    # Non-keyframe bundles honestly have no color artifact.
    assert sync.bundles[1].camera.color_artifact is None
    # Nearest telemetry aligned: right hand index 500 for frame 1.
    assert bundle.right_hand is not None
    assert bundle.right_hand.position["index"] == 500.0
    # Transport latency computed from host/device timestamp pair.
    assert bundle.right_hand.transport_latency_ms is not None
    assert 0.9 < bundle.right_hand.transport_latency_ms < 1.1
    # Unrecorded fields stay None (disclosed, not invented).
    assert bundle.camera.rgb_depth_skew_ms is None
    assert bundle.camera.depth_valid_ratio is None
    # Clock sync report covers the session.
    assert sync.clock_sync is not None and sync.clock_sync.samples == 30
    assert sync.clock_sync.max_skew_ms < 10.0


def test_bundles_jsonl_and_replay_integrity(tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    sync = build_bundles(session, camera_id="d435i_test")
    out = tmp_path / "sync_out" / "bundles.jsonl"
    result = write_bundles_jsonl(sync, out)
    assert result["records"] == 30

    key_bundle = sync.bundles[0]
    replayed = replay_observation(out, key_bundle.identity.observation_id)
    assert replayed is not None
    assert replayed["integrity"]["ok"] is True

    # Tamper with the artifact → integrity check must fail.
    Path(key_bundle.camera.color_artifact).write_bytes(b"tampered")
    replayed = replay_observation(out, key_bundle.identity.observation_id)
    assert replayed["integrity"]["ok"] is False

    assert replay_observation(out, "obs_does_not_exist") is None


def test_parquet_export_optional_dependency(tmp_path: Path) -> None:
    # pyarrow is the practice-export extra (optional — CI core env does
    # not install it; same importorskip pattern as test_export_parquet).
    import pytest

    pytest.importorskip("pyarrow")
    session = _write_session(tmp_path)
    sync = build_bundles(session, camera_id="d435i_test")
    parquet = export_bundles_parquet(sync, tmp_path / "out" / "bundles.parquet")
    assert parquet["rows"] == 30


def test_lerobot_mcap_exports(tmp_path: Path) -> None:
    session = _write_session(tmp_path)
    sync = build_bundles(session, camera_id="d435i_test")

    lerobot = export_bundles_lerobot(sync, tmp_path / "out" / "lerobot")
    assert lerobot["frames"] == 30
    assert lerobot["images_linked"] == 3
    meta = json.loads((tmp_path / "out" / "lerobot" / "meta.json").read_text())
    assert "never persisted" in meta["disclosure"]

    mcap = write_bundles_mcap(sync, tmp_path / "out" / "bundles.mcap")
    assert mcap["messages_bundles"] == 30
    assert (tmp_path / "out" / "bundles.mcap").stat().st_size > 1000

    everything = export_all(sync, tmp_path / "all")
    assert everything["stats"]["frames_total"] == 30
    assert everything["clock_sync"]["samples"] == 30
