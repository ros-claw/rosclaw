"""Time-sync sidecar for ROSClaw-rich LeRobotDatasets.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It writes ``meta/rosclaw/sync_stats.parquet`` (or JSONL fallback) so
downstream tools can assess per-episode timing quality without decoding the
worker timestamps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode


def _safe_delta_ns(prev: int | None, curr: int | None) -> float | None:
    if prev is None or curr is None:
        return None
    delta = curr - prev
    return float(delta) / 1e9


def build_sync_stats_rows(
    episodes: list[NormalizedPracticeEpisode],
) -> list[dict[str, Any]]:
    """Build per-episode timing/sync statistics."""
    rows: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        if not episode.frames:
            continue
        timestamps = [f.timestamp for f in episode.frames]
        source_timestamps = [f.source_timestamp_ns for f in episode.frames if f.source_timestamp_ns is not None]
        episode_times = [f.episode_time_sec for f in episode.frames if f.episode_time_sec is not None]
        clock_domains = {f.clock_domain for f in episode.frames if f.clock_domain is not None}

        row: dict[str, Any] = {
            "episode_index": episode_index,
            "rosclaw_episode_id": episode.episode_id,
            "num_frames": len(episode.frames),
            "fps": episode.fps,
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "duration_sec": round(timestamps[-1] - timestamps[0], 6),
            "clock_domain": sorted(clock_domains)[0] if clock_domains else None,
            "clock_domain_count": len(clock_domains),
        }

        if source_timestamps:
            row["start_source_timestamp_ns"] = source_timestamps[0]
            row["end_source_timestamp_ns"] = source_timestamps[-1]
            deltas = [
                _safe_delta_ns(source_timestamps[i], source_timestamps[i + 1])
                for i in range(len(source_timestamps) - 1)
            ]
            valid_deltas = [d for d in deltas if d is not None and d >= 0]
            if valid_deltas:
                row["source_delta_min_sec"] = round(min(valid_deltas), 6)
                row["source_delta_max_sec"] = round(max(valid_deltas), 6)
                row["source_delta_mean_sec"] = round(sum(valid_deltas) / len(valid_deltas), 6)
            else:
                row["source_delta_min_sec"] = None
                row["source_delta_max_sec"] = None
                row["source_delta_mean_sec"] = None
            row["source_timestamp_missing_frames"] = len(episode.frames) - len(source_timestamps)
        else:
            row["source_timestamp_missing_frames"] = len(episode.frames)

        if episode_times:
            row["start_episode_time_sec"] = episode_times[0]
            row["end_episode_time_sec"] = episode_times[-1]
            time_deltas = [
                episode_times[i + 1] - episode_times[i]
                for i in range(len(episode_times) - 1)
            ]
            valid_time_deltas = [d for d in time_deltas if not math.isnan(d) and d >= 0]
            if valid_time_deltas:
                row["episode_time_delta_mean_sec"] = round(sum(valid_time_deltas) / len(valid_time_deltas), 6)
            else:
                row["episode_time_delta_mean_sec"] = None
        else:
            row["episode_time_missing_frames"] = len(episode.frames)

        rows.append(row)
    return rows


def write_sync_stats_parquet(
    episodes: list[NormalizedPracticeEpisode],
    output_dir: Path | str,
) -> Path:
    """Write ``meta/rosclaw/sync_stats.parquet`` (or JSONL fallback)."""
    output_dir = Path(output_dir)
    sidecar_dir = output_dir / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    rows = build_sync_stats_rows(episodes)

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        path = sidecar_dir / "sync_stats.parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        path = sidecar_dir / "sync_stats.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path


__all__ = [
    "build_sync_stats_rows",
    "write_sync_stats_parquet",
]
