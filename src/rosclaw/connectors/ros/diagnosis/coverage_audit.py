"""Passive coverage diagnostics. Predictions never enter mission verification."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CoverageAuditLog:
    """Bounded nonblocking producers; one append-only writer and explicit loss.

    Audit failure never controls motion or changes canonical verification.
    A dropped event or failed writer makes the audit incomplete, not successful.
    Each instance requires a new file, preventing accidental historical reuse.
    """

    def __init__(self, path, *, context=None, capacity=1024, max_event_bytes=2_000_000):
        if type(capacity) is not int or capacity <= 0 or max_event_bytes <= 0:
            raise ValueError("audit queue and event limits must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", buffering=1)
        self.context = dict(context or {})
        self.queue = queue.Queue(maxsize=capacity)
        self.max_event_bytes = max_event_bytes
        self.lock = threading.Lock()
        self.dropped = 0
        self.error = None
        self.closed = False
        self.closing = threading.Event()
        self.sequence = 0
        self.previous_hash = None
        self.thread = threading.Thread(target=self._write, daemon=True, name="coverage-audit")
        self.thread.start()

    def emit(self, kind, payload, *, sim_time=None):
        # No disk IO or waiting on queue capacity in ROS/action callbacks.
        with self.lock:
            if self.closed:
                self.dropped += 1
                return
            row = {
                "schema_version": "rosclaw.coverage_audit_event.v1",
                **self.context,
                "kind": kind,
                "captured_at": datetime.now(UTC).isoformat(),
                "wall_monotonic_sec": time.monotonic(),
                "sim_time_sec": sim_time,
                "payload": payload,
            }
            try:
                # Freeze mutable input before returning to its producer.
                encoded = json.dumps(row, allow_nan=False, separators=(",", ":"))
                if len(encoded.encode()) > self.max_event_bytes:
                    raise ValueError("event exceeds bounded size")
                self.queue.put_nowait(encoded)
            except (queue.Full, TypeError, ValueError):
                self.dropped += 1

    def _write(self):
        try:
            while not self.closing.is_set() or not self.queue.empty():
                try:
                    row = json.loads(self.queue.get(timeout=0.05))
                except queue.Empty:
                    continue
                self.sequence += 1
                row.update(sequence=self.sequence, previous_hash=self.previous_hash)
                row["artifact_sha256"] = digest(row)
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.previous_hash = row["artifact_sha256"]
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stream.close()

    def close(self, *, timeout=2):
        with self.lock:
            self.closed = True
        self.closing.set()
        self.thread.join(timeout=timeout)
        return {
            "path": str(self.path),
            "events_written": self.sequence,
            "dropped_events": self.dropped,
            "writer_error": self.error,
            "writer_stopped": not self.thread.is_alive(),
            "complete": not self.thread.is_alive() and self.error is None and self.dropped == 0,
            "last_event_hash": self.previous_hash,
            "evidence_role": "diagnostic_only",
        }


def read_audit(path):
    """Reject missing/reordered/mutated/unfinished events; no silent repair."""
    previous = None
    rows = []
    for sequence, line in enumerate(Path(path).read_text().splitlines(keepends=True), 1):
        if not line.endswith("\n"):
            raise ValueError("unfinished audit event")
        row = json.loads(line)
        saved = row.pop("artifact_sha256")
        if row["sequence"] != sequence or row["previous_hash"] != previous or digest(row) != saved:
            raise ValueError("audit chain integrity mismatch")
        row["artifact_sha256"] = saved
        rows.append(row)
        previous = saved
    return rows


def trajectory_metrics(trajectory, *, stop_speed_mps=0.01, turn_rate_radps=0.05):
    """Observed distances and mutually exclusive interval classes, in SIM time."""
    distance = rotation = stationary = turning = driving = 0.0
    for a, b in zip(trajectory, trajectory[1:], strict=False):
        dt = b["time_sec"] - a["time_sec"]
        if dt <= 0:
            raise ValueError("trajectory timestamps must increase")
        length = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        angle = abs((b["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        distance += length
        rotation += angle
        if length / dt >= stop_speed_mps:
            driving += dt
        elif angle / dt >= turn_rate_radps:
            turning += dt
        else:
            stationary += dt
    return {
        "sample_count": len(trajectory),
        "observed_distance_m": distance,
        "accumulated_rotation_rad": rotation,
        "driving_sec": driving,
        "stationary_turn_sec": turning,
        "stop_wait_sec": stationary,
        "sim_duration_sec": trajectory[-1]["time_sec"] - trajectory[0]["time_sec"]
        if trajectory
        else 0.0,
        "classification_thresholds": {
            "stop_speed_mps": stop_speed_mps,
            "turn_rate_radps": turn_rate_radps,
        },
        "classification_note": "stop_wait does not distinguish obstacle waits from idle",
    }


def plan_projection(grid, poses, *, frame_id):
    """Ideal sampled-path sweep, separately labelled prediction, not credit.

    Path orientations are mandatory. Inventing yaw from x/y would conceal
    uncertainty, particularly with non-circular attachments and turn segments.
    Geometric prediction makes no reachability or tracking guarantee.
    """
    verifier = CoverageVerifier(**grid)
    clock = 0.0
    previous = None
    for p in poses:
        if previous is None:
            intermediate = [(p["x"], p["y"], p["yaw"])]
        else:
            length = math.hypot(p["x"] - previous["x"], p["y"] - previous["y"])
            angle = (p["yaw"] - previous["yaw"] + math.pi) % (2 * math.pi) - math.pi
            count = max(1, math.ceil(length / 0.5))
            if count > 10000:
                raise ValueError("plan projection exceeds bounded segment size")
            intermediate = [
                (
                    previous["x"] + (p["x"] - previous["x"]) * i / count,
                    previous["y"] + (p["y"] - previous["y"]) * i / count,
                    previous["yaw"] + angle * i / count,
                )
                for i in range(1, count + 1)
            ]
        for x, y, yaw in intermediate:
            verifier.observe(
                CleaningPose(x=x, y=y, yaw=yaw, time_sec=clock, cleaning_enabled=True),
                frame_id=frame_id,
            )
            clock += 0.5
        previous = p
    return {
        "predicted_coverage_ratio": len(verifier.visits) / len(verifier.accessible),
        "predicted_cells": sorted(verifier.visits),
        "planned_length_m": sum(
            math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            for a, b in zip(poses, poses[1:], strict=False)
        ),
        "projection_method": "ideal_interpolated_path_sweep_with_brush_on",
        "evidence_role": "prediction_only_never_measured_credit",
    }
