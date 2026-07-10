"""Per-frame event sidecar for ROSClaw-rich LeRobotDatasets.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It materializes sandbox, failure, intervention, and outcome events as
``meta/rosclaw/events.parquet`` so downstream tooling can reason about when
safety decisions and operator interventions occurred without having to decode
integer columns and vocabularies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode


_EVENT_FIELDS: list[tuple[str, str, str | None]] = [
    ("safety", "decision", None),
    ("safety", "modified", None),
    ("failure", "active", None),
    ("failure", "code", None),
    ("intervention", "active", None),
    ("intervention", "source", None),
    ("action_context", "source", None),
    ("action_context", "was_clamped", None),
    ("frame", "done", None),
    ("frame", "success", None),
]


def _format_value(value: Any) -> str | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def build_events_rows(
    episode: NormalizedPracticeEpisode,
    *,
    episode_index: int = 0,
) -> list[dict[str, Any]]:
    """Build one row per known event field per frame.

    Unknown/missing values are omitted so consumers can distinguish ``None``
    (not recorded) from explicit ``False`` / ``UNKNOWN``.
    """
    rows: list[dict[str, Any]] = []
    for frame in episode.frames:
        base = {
            "episode_index": episode_index,
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
        }
        safety = frame.safety
        failure = frame.failure
        intervention = frame.intervention
        action_context = frame.action_context

        if safety is not None:
            if safety.decision is not None:
                rows.append({**base, "event_type": "sandbox_decision", "value": safety.decision})
            if safety.modified is not None:
                rows.append({**base, "event_type": "sandbox_modified", "value": safety.modified})
            if safety.risk_score is not None:
                rows.append({**base, "event_type": "risk_score", "value": safety.risk_score})
            if safety.reason_code is not None:
                rows.append({**base, "event_type": "reason_code", "value": safety.reason_code})

        if failure is not None:
            if failure.active is not None:
                rows.append({**base, "event_type": "failure_active", "value": failure.active})
            if failure.code is not None:
                rows.append({**base, "event_type": "failure_code", "value": failure.code})
            if failure.severity is not None:
                rows.append({**base, "event_type": "failure_severity", "value": failure.severity})

        if intervention is not None:
            if intervention.active is not None:
                rows.append({**base, "event_type": "intervention_active", "value": intervention.active})
            if intervention.source is not None:
                rows.append({**base, "event_type": "intervention_source", "value": intervention.source})
            if intervention.confidence is not None:
                rows.append({**base, "event_type": "intervention_confidence", "value": intervention.confidence})

        if action_context is not None:
            if action_context.source is not None:
                rows.append({**base, "event_type": "action_source", "value": action_context.source})
            if action_context.was_clamped is not None:
                rows.append({**base, "event_type": "was_clamped", "value": action_context.was_clamped})

        if frame.done is not None:
            rows.append({**base, "event_type": "done", "value": frame.done})
        if frame.success is not None:
            rows.append({**base, "event_type": "success", "value": frame.success})

    return rows


def write_events_parquet(
    episodes: list[NormalizedPracticeEpisode],
    output_dir: Path | str,
) -> Path:
    """Write ``meta/rosclaw/events.parquet`` (or JSONL fallback) from episodes."""
    output_dir = Path(output_dir)
    sidecar_dir = output_dir / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        rows.extend(build_events_rows(episode, episode_index=idx))

    try:
        import pandas as pd

        path = sidecar_dir / "events.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        import json

        path = sidecar_dir / "events.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path


__all__ = [
    "build_events_rows",
    "write_events_parquet",
]
