"""Bridge from LeRobot shadow/proposal rollouts to the full Practice lifecycle."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from rosclaw.practice.config import DEFAULT_DATA_ROOT, PracticeSession, PracticeSummary
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout, generate_practice_id


def _compute_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_rollout_practice_session(
    trace_path: str | Path,
    rollout_result: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    practice_id: str | None = None,
) -> str:
    """Import a rollout JSONL trace into the Practice lifecycle.

    Returns the generated ``practice_id``.
    """
    trace_path = Path(trace_path)
    data_root = Path(data_root or DEFAULT_DATA_ROOT)
    practice_id = practice_id or generate_practice_id()

    layout = PracticeLayout(data_root)
    layout.ensure_directories()
    layout.create_session_dirs(practice_id)

    events: list[dict[str, Any]] = []
    if trace_path.exists():
        with open(trace_path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]

    target_events = layout.events_jsonl_path(practice_id)
    if trace_path.resolve() != target_events.resolve():
        shutil.copy2(trace_path, target_events)

    first_event = events[0] if events else {}
    last_event = events[-1] if events else {}

    robot_id = first_event.get("robot_id") or rollout_result.get("robot_id") or "default_robot"
    body_id = first_event.get("body_id") or rollout_result.get("body_id")
    task_id = first_event.get("task_id") or rollout_result.get("task_id") or "lerobot_rollout"
    session_id = first_event.get("session_id") or f"session_{practice_id}"
    episode_id = first_event.get("episode_id") or f"episode_{practice_id}"
    policy_id = first_event.get("policy_id") or rollout_result.get("policy_id")

    start_time_ns = first_event.get("timestamp_ns") or int(time.time_ns())
    end_time_ns = last_event.get("timestamp_ns") or start_time_ns
    duration_ms = (end_time_ns - start_time_ns) / 1e6 if end_time_ns >= start_time_ns else 0.0

    start_time_utc = _ns_to_iso(start_time_ns)
    session_dir = layout.session_dir(practice_id)

    session = PracticeSession(
        practice_id=practice_id,
        robot_id=robot_id,
        task_id=task_id,
        task_name=task_id,
        skill_id=None,
        session_dir=session_dir,
        start_time_ns=start_time_ns,
        start_time_utc=start_time_utc,
        robot_type=None,
        session_id=session_id,
        episode_id=episode_id,
        metadata={"body_id": body_id, "policy_id": policy_id, "source": "lerobot_rollout"},
    )

    stop_reason = rollout_result.get("stop_reason", "unknown")
    outcome = "SUCCESS" if stop_reason == "completed" else str(stop_reason).upper()
    summary = PracticeSummary(
        practice_id=practice_id,
        robot_id=robot_id,
        outcome=outcome,
        reward=None,
        duration_ms=duration_ms,
        event_count=len(events),
        failure_labels=rollout_result.get("errors", []),
        seekdb_committed=False,
    )

    layout.write_manifest(
        session,
        summary=summary,
        sources={"runtime": True, "provider": True, "sandbox": True},
    )
    layout.write_episode_json(practice_id, session, summary, sources={"runtime": True})
    layout.write_timeline_jsonl(practice_id, events)

    catalog = PracticeCatalog(layout.catalog_db_path)
    try:
        catalog.insert_practice(
            {
                "practice_id": practice_id,
                "session_id": session_id,
                "episode_id": episode_id,
                "robot_id": robot_id,
                "robot_type": session.robot_type,
                "task_id": task_id,
                "task_name": task_id,
                "skill_id": session.skill_id,
                "start_time": start_time_utc,
                "end_time": _ns_to_iso(end_time_ns),
                "duration_ms": duration_ms,
                "outcome": outcome,
                "reward": None,
                "manifest_path": str(layout.manifest_path(practice_id)),
                "events_jsonl_path": str(target_events),
                "replay_path": None,
                "failure_report_path": None,
                "seekdb_committed": 0,
            }
        )
        catalog.insert_session(
            {
                "session_id": session_id,
                "practice_id": practice_id,
                "body_id": body_id,
                "task_name": task_id,
                "started_at": start_time_utc,
                "ended_at": _ns_to_iso(end_time_ns),
                "status": "closed",
                "outcome": outcome,
                "event_count": len(events),
                "metadata": session.metadata,
            }
        )
        catalog.insert_episode(
            {
                "episode_id": episode_id,
                "session_id": session_id,
                "body_id": body_id,
                "skill_id": session.skill_id,
                "policy_id": policy_id,
                "started_at": start_time_utc,
                "ended_at": _ns_to_iso(end_time_ns),
                "outcome": outcome,
                "success": stop_reason == "completed",
                "failure_labels": summary.failure_labels,
                "metrics": rollout_result.get("metrics", {}),
            }
        )

        # Import events into the legacy events table for catalog-based queries.
        for ev in events:
            catalog.insert_event(
                {
                    "event_id": ev.get("event_id") or f"event_{practice_id}_{ev.get('sequence_id', 0)}",
                    "practice_id": practice_id,
                    "source": ev.get("source", "runtime"),
                    "event_type": ev.get("event_type", "unknown"),
                    "timestamp_ns": ev.get("timestamp_ns", start_time_ns),
                    "timestamp_utc": ev.get("timestamp_utc", start_time_utc),
                    "action_id": ev.get("action_id"),
                    "task_id": ev.get("task_id", task_id),
                    "skill_id": ev.get("skill_id"),
                    "payload_ref": json.dumps(ev.get("payload_ref", {}) or {}),
                    "tags": json.dumps(ev.get("tags", []) or []),
                }
            )

        # Register v2 artifacts for the events JSONL and timeline.
        timeline_path = layout.timeline_jsonl_path(practice_id)
        artifact_records = [
            {
                "artifact_id": f"artifact_{practice_id}_events_jsonl",
                "session_id": session_id,
                "episode_id": episode_id,
                "artifact_type": "events_jsonl",
                "path": str(target_events),
                "sha256": _compute_sha256(target_events),
                "size_bytes": target_events.stat().st_size,
                "schema_name": "practice.event.v1",
                "created_at": _ns_to_iso(end_time_ns),
                "metadata": {"role": "trace", "source": "lerobot_rollout"},
            },
        ]
        if timeline_path.exists():
            artifact_records.append(
                {
                    "artifact_id": f"artifact_{practice_id}_timeline_jsonl",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "artifact_type": "timeline_jsonl",
                    "path": str(timeline_path),
                    "sha256": _compute_sha256(timeline_path),
                    "size_bytes": timeline_path.stat().st_size,
                    "schema_name": "practice.event.v1",
                    "created_at": _ns_to_iso(end_time_ns),
                    "metadata": {"role": "timeline", "source": "lerobot_rollout"},
                }
            )
        for record in artifact_records:
            catalog.insert_artifact_v2(record)
    finally:
        catalog.close()

    return practice_id


def _ns_to_iso(ns: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat().replace("+00:00", "Z")
