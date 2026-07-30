"""Synchronized-layer exporters + replay (Physical Evolution Lab §6.4, PR-PE-2).

One session's :class:`SyncResult` becomes:

* ``bundles.jsonl`` — the canonical synchronized records (one
  PhysicalObservationBundle per line, content-addressed ids);
* ``bundles.parquet`` — the analysis-level synchronized table;
* ``lerobot/`` — a LeRobot-format dataset (observation.state = hand
  positions, action = hand targets, image when a keyframe artifact
  exists — never a fabricated frame);
* ``bundles.mcap`` — replay container (channels /observation, /camera,
  /hands/left, /hands/right).

Replay is by ``observation_id`` only (v3 §5.1: no directory-name
association).  Artifact references carry sha256 so a swapped or
truncated artifact fails the replay integrity check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multimodal_sync import SyncResult
from .physical_observation import PhysicalObservationBundle


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_row(bundle: PhysicalObservationBundle) -> dict[str, Any]:
    record = bundle.to_record()
    artifact = bundle.camera.color_artifact
    if artifact:
        record["camera"]["color_artifact_sha256"] = _sha256_file(Path(artifact))
    return record


def write_bundles_jsonl(sync: SyncResult, out_path: str | Path) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for bundle in sync.bundles:
            handle.write(json.dumps(_bundle_row(bundle), ensure_ascii=False, default=str) + "\n")
            written += 1
    return {"path": str(out_path), "records": written}


def export_bundles_parquet(sync: SyncResult, out_path: str | Path) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for Parquet export — install the "
            "'practice-export' extra (pip install rosclaw[practice-export])"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bundle in sync.bundles:
        row: dict[str, Any] = {
            "observation_id": bundle.identity.observation_id,
            "session_id": bundle.identity.session_id,
            "episode_id": bundle.identity.episode_id,
            "round_id": bundle.identity.round_id,
            "host_monotonic_ns": bundle.host_monotonic_ns,
            "camera_ts_s": bundle.camera.device_timestamp_s,
            "frame_age_ms": bundle.camera.frame_age_ms,
            "gesture": bundle.derived.get("gesture"),
            "gesture_confidence": bundle.derived.get("gesture_confidence"),
            "has_color_artifact": bundle.camera.color_artifact is not None,
            "sensor_health": bundle.camera.sensor_health,
            "trace_id": bundle.trace_id,
            "action_id": bundle.action_id,
        }
        for side, hand in (("left", bundle.left_hand), ("right", bundle.right_hand)):
            if hand is None:
                row[f"{side}_present"] = False
                continue
            row[f"{side}_present"] = True
            for joint in ("little", "ring", "middle", "index", "thumb", "thumb_rot"):
                row[f"{side}_pos_{joint}"] = hand.position.get(joint)
            row[f"{side}_transport_latency_ms"] = hand.transport_latency_ms
            temps = hand.temperature_c or {}
            row[f"{side}_temp_max"] = max(temps.values()) if temps else None
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return {"path": str(out_path), "rows": len(rows)}


def export_bundles_lerobot(sync: SyncResult, out_dir: str | Path) -> dict[str, Any]:
    """LeRobot-format episode dataset (v3 §6.4 Learning Layer).

    Only bundles with at least one hand AND a target-free observation
    become frames; images are linked ONLY when the keyframe artifact
    exists on disk (a missing image is a missing image, never a reused
    one)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    images_linked = 0
    for index, bundle in enumerate(sync.bundles):
        hand = bundle.right_hand or bundle.left_hand
        if hand is None:
            continue
        frame: dict[str, Any] = {
            "frame_index": index,
            "observation_id": bundle.identity.observation_id,
            "timestamp": bundle.camera.device_timestamp_s,
            "observation.state": [hand.position.get(j) for j in sorted(hand.position)],
            "observation.state_joints": sorted(hand.position),
            "task": bundle.derived.get("gesture") or "unknown",
        }
        artifact = bundle.camera.color_artifact
        if artifact and Path(artifact).is_file():
            frame["observation.image"] = artifact
            images_linked += 1
        frames.append(frame)
    meta = {
        "dataset_version": "lerobot.handcam.v1",
        "practice_id": sync.practice_id,
        "frames": len(frames),
        "images_linked": images_linked,
        "disclosure": (
            "images only where keyframe artifacts exist on disk; "
            "per-frame rgb/depth was never persisted in this corpus"
        ),
    }
    (out_dir / "data.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False, default=str) + "\n" for f in frames),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"path": str(out_dir), "frames": len(frames), "images_linked": images_linked}


def write_bundles_mcap(sync: SyncResult, out_path: str | Path) -> dict[str, Any]:
    from mcap.writer import Writer

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema_id = writer.register_schema(
            name="rosclaw.physical_observation.v1",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        channels = {
            name: writer.register_channel(topic=name, message_encoding="json", schema_id=schema_id)
            for name in ("/observation", "/camera", "/hands/left", "/hands/right")
        }
        for bundle in sync.bundles:
            log_time = bundle.host_monotonic_ns or 0
            record = bundle.to_record()
            writer.add_message(
                channel_id=channels["/observation"],
                log_time=log_time,
                publish_time=log_time,
                data=json.dumps(record, ensure_ascii=False, default=str).encode(),
            )
            writer.add_message(
                channel_id=channels["/camera"],
                log_time=log_time,
                publish_time=log_time,
                data=json.dumps(record["camera"], ensure_ascii=False, default=str).encode(),
            )
            for side, topic in (("left_hand", "/hands/left"), ("right_hand", "/hands/right")):
                if record.get(side):
                    writer.add_message(
                        channel_id=channels[topic],
                        log_time=log_time,
                        publish_time=log_time,
                        data=json.dumps(record[side], ensure_ascii=False, default=str).encode(),
                    )
            written += 1
        writer.finish()
    return {"path": str(out_path), "messages_bundles": written}


def export_all(sync: SyncResult, out_dir: str | Path) -> dict[str, Any]:
    """Run every exporter into one synchronized-layer directory.

    Optional-dependency exporters (pyarrow → Parquet) that are missing
    degrade to a DISCLOSED skip record — the other formats still land;
    a silently missing format would be worse than no export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        parquet = export_bundles_parquet(sync, out_dir / "bundles.parquet")
    except ModuleNotFoundError as exc:
        parquet = {"skipped": str(exc)}
    return {
        "practice_id": sync.practice_id,
        "jsonl": write_bundles_jsonl(sync, out_dir / "bundles.jsonl"),
        "parquet": parquet,
        "lerobot": export_bundles_lerobot(sync, out_dir / "lerobot"),
        "mcap": write_bundles_mcap(sync, out_dir / "bundles.mcap"),
        "stats": sync.stats.to_dict(),
        "clock_sync": sync.clock_sync.to_dict() if sync.clock_sync else None,
    }


def replay_observation(bundles_jsonl: str | Path, observation_id: str) -> dict[str, Any] | None:
    """Find one observation by id and verify its artifact integrity
    (replay = id-addressed, integrity-checked)."""
    with Path(bundles_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("identity") or {}).get("observation_id") != observation_id:
                continue
            artifact = (record.get("camera") or {}).get("color_artifact")
            expected = (record.get("camera") or {}).get("color_artifact_sha256")
            integrity: dict[str, Any] = {"artifact": artifact, "expected_sha256": expected}
            if artifact and expected:
                actual = _sha256_file(Path(artifact))
                integrity["actual_sha256"] = actual
                integrity["ok"] = actual == expected
            elif artifact:
                integrity["ok"] = Path(artifact).is_file()
            else:
                integrity["ok"] = None  # no artifact referenced
            return {"record": record, "integrity": integrity}
    return None
