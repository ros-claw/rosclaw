"""Parquet exporter for ROSClaw Practice sessions.

Turns a practice session's raw ``events.jsonl`` into a time-aligned Parquet table
suitable for downstream analytics and ML pipelines. Each row corresponds to one
Practice event; ``physical_feedback_event`` payloads are flattened into
``observation_state`` and ``action`` columns, while other events contribute
context columns and a JSON ``metadata`` blob.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.exporters.parquet")


class ParquetExporter:
    """Export a ROSClaw practice session to Parquet."""

    def __init__(self, data_root: Path | str):
        self._layout = PracticeLayout(data_root)

    def export(
        self,
        practice_id: str,
        output_path: Path | str | None = None,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
    ) -> Path:
        """Export ``practice_id`` events to a Parquet file.

        Args:
            practice_id: The practice identifier.
            output_path: Destination Parquet file. Defaults to
                ``<data_root>/datasets/parquet/<practice_id>.parquet``.
            session_id: Optional session id for metadata.
            episode_id: Optional episode id for metadata.

        Returns:
            Path to the exported Parquet file.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "Parquet export requires pyarrow. Install with: pip install pyarrow"
            ) from e

        events = self._load_events(practice_id)
        if not events:
            raise ValueError(f"No events found for practice {practice_id}")

        if output_path is None:
            output_path = self._layout.data_root / "datasets" / "parquet" / f"{practice_id}.parquet"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta = self._infer_metadata(events)
        session_id = session_id or meta.get("session_id")
        episode_id = episode_id or meta.get("episode_id")

        rows: list[dict[str, Any]] = []
        for ev in events:
            payload = ev.get("payload") or {}
            row: dict[str, Any] = {
                "practice_id": ev.get("practice_id"),
                "session_id": ev.get("session_id"),
                "episode_id": ev.get("episode_id"),
                "robot_id": ev.get("robot_id"),
                "body_id": ev.get("body_id"),
                "skill_id": ev.get("skill_id"),
                "action_id": ev.get("action_id"),
                "policy_id": ev.get("policy_id"),
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "source": ev.get("source"),
                "timestamp_ns": ev.get("timestamp_ns"),
                "timestamp_s": (ev.get("timestamp_ns") / 1e9) if ev.get("timestamp_ns") else None,
                "observation_state": None,
                "action": None,
                "metadata": json.dumps(
                    {
                        "tags": ev.get("tags", []),
                        "trace_id": ev.get("trace_id"),
                        "parent_event_id": ev.get("parent_event_id"),
                        "frame_id": ev.get("frame_id"),
                        "quality": ev.get("quality"),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }

            if ev.get("event_type") == "physical_feedback_event":
                row["observation_state"] = json.dumps(
                    self._extract_observation_state(payload),
                    ensure_ascii=False,
                    default=str,
                )
                row["action"] = json.dumps(
                    self._extract_action(payload),
                    ensure_ascii=False,
                    default=str,
                )

            rows.append(row)

        table = self._records_to_table(rows)
        table = table.replace_schema_metadata(
            {
                "practice_id": practice_id,
                "session_id": session_id or "",
                "episode_id": episode_id or "",
                "robot_id": meta.get("robot_id", ""),
                "body_id": meta.get("body_id", ""),
                "skill_id": meta.get("skill_id", ""),
                "event_count": str(len(rows)),
                "exporter": "rosclaw.practice.exporters.parquet",
            }
        )
        pq.write_table(table, output_path)
        logger.info("Exported %d events to %s", len(rows), output_path)
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
            for key in ("session_id", "episode_id", "robot_id", "body_id", "skill_id"):
                if key not in meta and ev.get(key):
                    meta[key] = ev[key]
            payload = ev.get("payload") or {}
            if "body_id" not in meta and payload.get("body_id"):
                meta["body_id"] = payload["body_id"]
            if all(meta.get(k) for k in ("session_id", "episode_id", "body_id")):
                break
        return meta

    @staticmethod
    def _extract_observation_state(payload: dict[str, Any]) -> dict[str, Any]:
        """Map a physical feedback payload to an observation state dict."""
        obs: dict[str, Any] = {
            "actual": payload.get("actual", {}),
            "force_net": payload.get("force_net", {}),
            "force_baseline": payload.get("force_baseline", {}),
            "force_delta": payload.get("force_delta", {}),
            "current": payload.get("current", {}),
            "temperature": payload.get("temperature", {}),
            "status": payload.get("status", {}),
            "error": payload.get("error", {}),
            "primary_event": payload.get("primary_event", "unknown"),
        }
        return obs

    @staticmethod
    def _extract_action(payload: dict[str, Any]) -> dict[str, Any]:
        """Map a physical feedback payload to an action dict."""
        return {
            "target": payload.get("target", {}),
            "force_set": payload.get("force_set", {}),
            "speed": payload.get("speed", {}),
        }

    @staticmethod
    def _records_to_table(records: list[dict[str, Any]]) -> Any:
        """Convert a list of dicts to a PyArrow Table with uniform columns."""
        import pyarrow as pa

        if not records:
            return pa.Table.from_pydict({})

        keys: set[str] = set()
        for record in records:
            keys.update(record.keys())

        normalized: list[dict[str, Any]] = []
        for record in records:
            row = {}
            for k in keys:
                v = record.get(k)
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    row[k] = v
            normalized.append(row)

        return pa.Table.from_pylist(normalized)
