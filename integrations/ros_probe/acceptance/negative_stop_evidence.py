"""Read-only negative-episode safety evidence, independent of task outcome.

Closed actuator evidence proves SIM cleaner state and lease release; advancing
Gazebo poses prove standstill. Neither source can replace the other. Historical
source hashes establish correspondence, not authentication or new live state.
"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from independent_stop import stop_geometry

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent


def closed_brush_stop(path, binding, stop):
    """Require complete genesis-to-close actuator history and OFF across stop."""
    if type(binding) is not dict or set(binding) != {
        "run_id",
        "body_snapshot_hash",
        "attachment_hash",
        "producer_id",
    }:
        raise ValueError("exact independent actuator binding required")
    if any(type(v) is not str or not v for v in binding.values()):
        raise ValueError("nonempty independent actuator identities required")
    samples = stop.get("samples") if type(stop) is dict else None
    geometry = stop_geometry(samples)
    start, end = samples[0]["time_sec"], samples[-1]["time_sec"]
    wall_start = datetime.fromisoformat(samples[0]["captured_at"])
    wall_end = datetime.fromisoformat(samples[-1]["captured_at"])
    path = Path(path)
    summary_path = Path(str(path) + ".summary.json")
    if not 0 < path.stat().st_size <= 1_000_000_000 or not 0 < summary_path.stat().st_size <= 65536:
        raise ValueError("bounded closed actuator audit required")
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    if (
        type(summary) is not dict
        or summary.get("complete") is not True
        or summary.get("writer_stopped") is not True
        or type(summary.get("events_written")) is not int
        or type(summary.get("dropped_events")) is not int
        or summary["dropped_events"] != 0
        or summary.get("writer_error") is not None
    ):
        raise ValueError("complete closed independent actuator audit required")
    previous = sequence = last_event = None
    last_capture = None
    enabled = False
    selected = []
    last_before = first_after = None
    file_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete actuator audit row required")
            file_hash.update(line)
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError("typed actuator audit row required")
            saved = row.pop("artifact_sha256", None)
            if (
                row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                or type(row.get("sequence")) is not int
                or row["sequence"] != (sequence or 0) + 1
                or row.get("previous_hash") != previous
                or saved != digest(row)
                or any(row.get(k) != v for k, v in binding.items())
                or row.get("source") != "simulator_owned_actuator"
                or row.get("evidence_domain") != "SIMULATION"
                or row.get("kind") != "brush_state_event"
            ):
                raise ValueError("independent actuator chain/source identity differs")
            previous, sequence = saved, row["sequence"]
            payload = row.get("payload")
            if type(payload) is not dict:
                raise ValueError("typed actuator event payload required")
            event = BrushStateEvent(**payload["event"])
            capture = datetime.fromisoformat(event.captured_at)
            if (
                event.artifact_hash() != payload.get("artifact_hash")
                or any(getattr(event, k) != v for k, v in binding.items())
                or event.complete is not True
                or event.evidence_domain != "SIMULATION"
                or type(event.enabled) is not bool
                or type(event.sequence) is not int
                or event.sequence != (0 if last_event is None else last_event.sequence + 1)
                or type(event.sim_time_sec) not in (int, float)
                or not math.isfinite(event.sim_time_sec)
                or event.sim_time_sec < 0
                or row.get("sim_time_sec") != event.sim_time_sec
                or event.kind not in {"WATERMARK", "TRANSITION"}
                or capture.tzinfo is None
                or (last_capture is not None and capture < last_capture)
                or (last_event is not None and event.sim_time_sec < last_event.sim_time_sec)
                or (last_event is None and (event.kind != "WATERMARK" or event.enabled))
                or (event.kind == "WATERMARK" and event.enabled != enabled)
                or type(payload.get("lease_remaining_sec")) not in (int, float)
                or not math.isfinite(payload["lease_remaining_sec"])
            ):
                raise ValueError("complete ordered actuator event and initial OFF required")
            enabled, last_event, last_capture = event.enabled, event, capture
            # Require both original SIM time and original wall receipt to agree
            # with the independently collected window; no rewritten timestamps.
            if event.sim_time_sec <= start and capture <= wall_start:
                last_before = (event, payload, capture)
            if start <= event.sim_time_sec <= end or wall_start <= capture <= wall_end:
                selected.append((event, payload, capture))
                if len(selected) > 512:
                    raise ValueError("bounded advancing actuator stop window required")
            if first_after is None and event.sim_time_sec >= end and capture >= wall_end:
                first_after = (event, payload, capture)
    if summary.get("events_written") != sequence or summary.get("last_event_hash") != previous:
        raise ValueError("actuator closure differs from complete event chain")
    if last_before is None or first_after is None or len(selected) < 20:
        raise ValueError("actuator history does not bracket independent stop window")
    window = [last_before, *selected, first_after]
    for event, payload, _capture in window:
        if event.enabled or payload["lease_remaining_sec"] > 0:
            raise ValueError("cleaner enabled or daemon lease still live during stop")
    for a, b in zip(window, window[1:], strict=False):
        if b[0].sim_time_sec - a[0].sim_time_sec > 0.3 or (b[2] - a[2]).total_seconds() > 0.3:
            raise ValueError("independent actuator stop evidence contains a source gap")
    return {
        "evidence_role": "CLOSED_ACTUATOR_STOP_CORRESPONDENCE_NOT_TASK_ACCEPTANCE",
        "physical_acceptance": "NOT_VERIFIED",
        "brush_off": True,
        "lease_released": True,
        "independent_stop": geometry,
        "stop_window_actuator_events": len(selected),
        "actuator_audit_sha256": file_hash.hexdigest(),
        "actuator_summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
    }
