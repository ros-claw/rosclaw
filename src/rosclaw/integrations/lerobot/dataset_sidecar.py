"""ROSClaw sidecar files for LeRobotDataset v3.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It writes episode-level metadata that standard LeRobotDataset ignores
but ROSClaw tooling can consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.practice_normalizer import NormalizedPracticeEpisode


def _count_intervention_frames(frames: list[Any]) -> int:
    return sum(
        1
        for f in frames
        if getattr(f, "intervention", None) and getattr(f.intervention, "active", False)
    )


def _count_sandbox_block_frames(frames: list[Any]) -> int:
    count = 0
    for f in frames:
        safety = getattr(f, "safety", None)
        if safety and getattr(safety, "decision", None) in ("BLOCK", "ESTOP"):
            count += 1
    return count


def _first_failure_code(frames: list[Any]) -> str | None:
    for f in frames:
        failure = getattr(f, "failure", None)
        if failure and getattr(failure, "active", False):
            return getattr(failure, "code", None) or None
    return None


def build_episodes_row(
    episode: NormalizedPracticeEpisode,
    *,
    episode_index: int = 0,
    exporter_version: str = "rosclaw.lerobot.p2.1.gate-a",
) -> dict[str, Any]:
    """Build a single row for ``meta/rosclaw/episodes.parquet``."""
    frames = episode.frames
    last_frame = frames[-1] if frames else None
    success = bool(getattr(last_frame, "success", False)) if last_frame else False
    failure_code = _first_failure_code(frames)

    robot = episode.robot
    task = episode.task

    return {
        "episode_index": episode_index,
        "rosclaw_episode_id": episode.episode_id,
        "robot_id": robot.robot_id if robot else None,
        "body_profile": robot.body_profile if robot else None,
        "body_hash": robot.body_hash if robot else None,
        "body_yaml_path": robot.body_yaml_path if robot else None,
        "eurdf_repo": None,
        "eurdf_revision": None,
        "provider_type": None,
        "provider_name": None,
        "policy_path": None,
        "practice_version": episode.schema_version,
        "exporter_version": exporter_version,
        "task_id": task.task_id if task else None,
        "task_text": task.text if task else None,
        "success": success,
        "failure_code": failure_code,
        "intervention_frames": _count_intervention_frames(frames),
        "sandbox_block_frames": _count_sandbox_block_frames(frames),
        "num_frames": len(frames),
        "fps": episode.fps,
        "start_wall_time": None,
        "end_wall_time": None,
    }


def write_episodes_parquet(
    episodes: list[NormalizedPracticeEpisode],
    output_dir: Path | str,
    *,
    exporter_version: str = "rosclaw.lerobot.p2.1.gate-a",
) -> Path:
    """Write ``meta/rosclaw/episodes.parquet``.

    This implementation uses pandas/pyarrow when available and falls back to a
    JSONL file if parquet libraries are not installed in the ROSClaw core.
    """
    output_dir = Path(output_dir)
    sidecar_dir = output_dir / "meta" / "rosclaw"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        build_episodes_row(ep, episode_index=i, exporter_version=exporter_version)
        for i, ep in enumerate(episodes)
    ]

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        path = sidecar_dir / "episodes.parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        # Fallback: write a JSONL sidecar so the data is still inspectable.
        import json

        path = sidecar_dir / "episodes.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path


__all__ = [
    "build_episodes_row",
    "write_episodes_parquet",
]
