"""LeRobot exporter for ROSClaw Practice sessions.

Generates a LeRobot-compatible dataset directory from a practice session's raw
``events.jsonl``.  The exporter focuses on the common physical-AI case where
``physical_feedback_event`` payloads provide time-aligned observations and
actions; all other events are preserved in ``rosclaw_extra.jsonl`` so no
context is lost.

The generated layout is compatible with the LeRobot ``LeRobotDataset``
convention (v2.1):

    <output_dir>/
      data/
        observation.state.parquet
        action.parquet
        episode_index.parquet
        timestamp.parquet
      meta/
        info.json
        tasks.jsonl
      rosclaw_extra.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.exporters.lerobot")


class LeRobotExporter:
    """Export a ROSClaw practice session to LeRobot dataset format."""

    def __init__(self, data_root: Path | str):
        self._layout = PracticeLayout(data_root)

    def export(
        self,
        practice_id: str,
        output_path: Path | str | None = None,
        *,
        task: str | None = None,
    ) -> Path:
        """Export ``practice_id`` events to a LeRobot dataset directory.

        Args:
            practice_id: The practice identifier.
            output_path: Destination directory. Defaults to
                ``<data_root>/datasets/lerobot/<practice_id>``.
            task: Optional language instruction / task description.

        Returns:
            Path to the exported dataset directory.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "LeRobot export requires pyarrow. Install with: pip install pyarrow"
            ) from e

        events = self._load_events(practice_id)
        if not events:
            raise ValueError(f"No events found for practice {practice_id}")

        if output_path is None:
            output_path = self._layout.data_root / "datasets" / "lerobot" / practice_id
        output_path = Path(output_path)
        data_dir = output_path / "data"
        meta_dir = output_path / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        meta = self._infer_metadata(events)
        task = task or meta.get("task_name") or meta.get("task_id") or "rosclaw practice"

        # Build per-frame observations/actions from physical feedback events.
        dofs = self._infer_dof_order(events)
        frames: list[dict[str, Any]] = []
        extra_events: list[dict[str, Any]] = []
        episode_index = 0
        index_in_episode = 0

        for ev in events:
            if ev.get("event_type") == "physical_feedback_event":
                payload = ev.get("payload") or {}
                state_vec = self._build_vector(payload, dofs, "observation")
                action_vec = self._build_vector(payload, dofs, "action")
                timestamp_s = (
                    ev.get("timestamp_ns") / 1e9 if ev.get("timestamp_ns") else None
                )
                frames.append(
                    {
                        "timestamp": timestamp_s,
                        "episode_index": episode_index,
                        "index_in_episode": index_in_episode,
                        "state": state_vec,
                        "action": action_vec,
                    }
                )
                index_in_episode += 1
            else:
                extra_events.append(ev)

        if not frames:
            raise ValueError(
                f"No physical_feedback_event events found for practice {practice_id}; "
                "cannot produce LeRobot frames."
            )

        # Write data/*.parquet
        timestamps = [f["timestamp"] for f in frames]
        episode_indices = [f["episode_index"] for f in frames]
        index_in_episodes = [f["index_in_episode"] for f in frames]
        states = [f["state"] for f in frames]
        actions = [f["action"] for f in frames]

        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.float32()),
                    "episode_index": pa.array(episode_indices, type=pa.int64()),
                    "index_in_episode": pa.array(index_in_episodes, type=pa.int64()),
                    "observation.state": pa.array(states, type=pa.list_(pa.float32())),
                }
            ),
            data_dir / "observation.state.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.float32()),
                    "episode_index": pa.array(episode_indices, type=pa.int64()),
                    "index_in_episode": pa.array(index_in_episodes, type=pa.int64()),
                    "action": pa.array(actions, type=pa.list_(pa.float32())),
                }
            ),
            data_dir / "action.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode_index], type=pa.int64()),
                    "episode_length": pa.array([len(frames)], type=pa.int64()),
                }
            ),
            data_dir / "episode_index.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.float32()),
                }
            ),
            data_dir / "timestamp.parquet",
        )

        # Write rosclaw extras.
        extras_path = output_path / "rosclaw_extra.jsonl"
        with open(extras_path, "w", encoding="utf-8") as f:
            for ev in extra_events:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

        # Write meta/info.json
        info = {
            "codebase_version": "v2.1",
            "repo_id": f"rosclaw/{practice_id}",
            "total_episodes": 1,
            "total_frames": len(frames),
            "sampling_rate": self._infer_sampling_rate(timestamps),
            "robot_id": meta.get("robot_id", ""),
            "body_id": meta.get("body_id", ""),
            "skill_id": meta.get("skill_id", ""),
            "task": task,
            "video": False,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [len(dofs)],
                    "names": dofs,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [len(dofs)],
                    "names": dofs,
                },
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index_in_episode": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
            },
        }
        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        # Write meta/tasks.jsonl
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

        logger.info(
            "Exported %d frames (%d extra events) to %s",
            len(frames),
            len(extra_events),
            output_path,
        )
        return output_path

    def _load_events(self, practice_id: str) -> list[dict[str, Any]]:
        """Load raw events from JSONL, falling back to the catalog path."""
        jsonl_path = self._layout.events_jsonl_path(practice_id)
        if not jsonl_path.exists():
            from rosclaw.practice.storage.catalog import PracticeCatalog

            catalog = PracticeCatalog(self._layout.catalog_db_path)
            try:
                record = catalog.get_practice(practice_id)
                if record and record.get("events_jsonl_path"):
                    jsonl_path = Path(record["events_jsonl_path"])
            finally:
                catalog.close()

        if not jsonl_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL line: %s", e)
        return events

    @staticmethod
    def _infer_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Infer session metadata from the first event that carries it."""
        meta: dict[str, Any] = {}
        for ev in events:
            for key in ("session_id", "episode_id", "robot_id", "body_id", "skill_id", "task_id", "task_name"):
                if key not in meta and ev.get(key):
                    meta[key] = ev[key]
            payload = ev.get("payload") or {}
            if "body_id" not in meta and payload.get("body_id"):
                meta["body_id"] = payload["body_id"]
            if all(meta.get(k) for k in ("session_id", "episode_id", "body_id")):
                break
        return meta

    @staticmethod
    def _infer_dof_order(events: list[dict[str, Any]]) -> list[str]:
        """Determine a stable DOF ordering from physical feedback events."""
        keys: set[str] = set()
        for ev in events:
            if ev.get("event_type") != "physical_feedback_event":
                continue
            payload = ev.get("payload") or {}
            for group in ("target", "actual", "force_net", "force_set", "speed"):
                keys.update(payload.get(group, {}).keys())
        return sorted(keys)

    @staticmethod
    def _build_vector(
        payload: dict[str, Any],
        dofs: list[str],
        mode: Literal["observation", "action"],
    ) -> list[float]:
        """Build a fixed-length float vector for a DOF list."""
        if mode == "observation":
            primary = payload.get("actual", {})
            fallback = payload.get("force_net", {})
        else:
            primary = payload.get("target", {})
            fallback = payload.get("force_set", {})

        vec: list[float] = []
        for dof in dofs:
            value = primary.get(dof)
            if value is None:
                value = fallback.get(dof)
            vec.append(float(value) if value is not None else 0.0)
        return vec

    @staticmethod
    def _infer_sampling_rate(timestamps: list[float | None]) -> float:
        """Estimate the sampling rate from consecutive timestamps."""
        valid = [t for t in timestamps if t is not None]
        if len(valid) < 2:
            return 0.0
        intervals = [valid[i + 1] - valid[i] for i in range(len(valid) - 1)]
        mean_interval = sum(intervals) / len(intervals)
        return round(1.0 / mean_interval, 2) if mean_interval > 0 else 0.0
