"""Filesystem layout and manifest helpers for rosclaw-practice."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from rosclaw.practice.config import PracticeSession, PracticeSummary

logger = logging.getLogger("rosclaw.practice.storage.layout")


class PracticeLayout:
    """Manages directory layout for a practice data root."""

    def __init__(self, data_root: Path | str):
        self._data_root = Path(data_root)

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def sessions_dir(self) -> Path:
        return self._data_root / "sessions"

    @property
    def indexes_dir(self) -> Path:
        return self._data_root / "indexes"

    @property
    def fallback_dir(self) -> Path:
        return self._data_root / "fallback"

    @property
    def catalog_db_path(self) -> Path:
        return self.indexes_dir / "practice_catalog.sqlite"

    def ensure_directories(self) -> None:
        for sub in (
            "sessions",
            "fallback",
            "fallback/archived",
            "indexes",
            "datasets/rlds",
            "datasets/lerobot",
        ):
            (self._data_root / sub).mkdir(parents=True, exist_ok=True)

    def session_dir(self, practice_id: str) -> Path:
        return self.sessions_dir / practice_id

    def raw_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "raw"

    def index_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "index"

    def reports_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "reports"

    def events_jsonl_path(self, practice_id: str) -> Path:
        return self.raw_dir(practice_id) / "events.jsonl"

    def timeline_jsonl_path(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "timeline.jsonl"

    def episode_json_path(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "episode.json"

    def manifest_path(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "manifest.yaml"

    def artifacts_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "artifacts"

    def provider_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "provider"

    def sandbox_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "sandbox"

    def runtime_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "runtime"

    def metrics_dir(self, practice_id: str) -> Path:
        return self.session_dir(practice_id) / "metrics"

    def create_session_dirs(self, practice_id: str) -> Path:
        session = self.session_dir(practice_id)
        for sub in (
            "raw",
            "frames/rgb",
            "frames/depth",
            "artifacts/frames",
            "artifacts/skill",
            "provider",
            "sandbox",
            "runtime",
            "metrics",
            "derived",
            "exports",
            "replay",
            "reports",
            "index",
        ):
            (session / sub).mkdir(parents=True, exist_ok=True)
        return session

    def write_manifest(
        self,
        session: PracticeSession,
        summary: PracticeSummary | None = None,
        sources: dict[str, bool] | None = None,
        seekdb_enabled: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        manifest: dict[str, Any] = {
            "schema_version": "practice.manifest.v1",
            "practice_id": session.practice_id,
            "session_id": session.session_id,
            "robot_id": session.robot_id,
            "robot_type": session.robot_type,
            "task": {
                "task_id": session.task_id,
                "task_name": session.task_name,
                "skill_id": session.skill_id,
            },
            "start_time": session.start_time_utc,
            "end_time": None,
            "duration_ms": None,
            "sources": sources or {},
            "artifacts": {
                "events_jsonl": str(self.events_jsonl_path(session.practice_id)),
                "timeline_jsonl": str(self.timeline_jsonl_path(session.practice_id)),
                "episode_json": str(self.episode_json_path(session.practice_id)),
                "frames": str(self.session_dir(session.practice_id) / "frames"),
                "artifacts": str(self.artifacts_dir(session.practice_id)),
                "provider": str(self.provider_dir(session.practice_id)),
                "sandbox": str(self.sandbox_dir(session.practice_id)),
                "runtime": str(self.runtime_dir(session.practice_id)),
                "metrics": str(self.metrics_dir(session.practice_id)),
            },
            "seekdb": {
                "enabled": seekdb_enabled,
                "committed": None,
                "table": "praxis_events",
            },
            "status": {
                "outcome": "running",
                "failure_labels": [],
                "reward": None,
            },
        }
        if extra:
            manifest.update(extra)
        if summary is not None:
            manifest["end_time"] = _utc_now_iso()
            manifest["duration_ms"] = summary.duration_ms
            manifest["status"]["outcome"] = summary.outcome
            manifest["status"]["reward"] = summary.reward
            manifest["status"]["failure_labels"] = summary.failure_labels
            manifest["seekdb"]["committed"] = summary.seekdb_committed
        path = self.manifest_path(session.practice_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    def read_manifest(self, practice_id: str) -> dict[str, Any] | None:
        path = self.manifest_path(practice_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed to read manifest %s: %s", path, e)
            return None

    def write_episode_json(
        self,
        practice_id: str,
        session: PracticeSession,
        summary: PracticeSummary,
        sources: dict[str, bool] | None = None,
    ) -> None:
        """Write a top-level episode.json summary for dashboards and exports."""
        path = self.episode_json_path(practice_id)
        payload: dict[str, Any] = {
            "schema_version": "practice.episode.v1",
            "practice_id": practice_id,
            "session_id": session.session_id,
            "robot_id": session.robot_id,
            "robot_type": session.robot_type,
            "task": {
                "task_id": session.task_id,
                "task_name": session.task_name,
                "skill_id": session.skill_id,
            },
            "sources": sources or {},
            "start_time": session.start_time_utc,
            "end_time": _utc_now_iso() if summary else None,
            "duration_ms": summary.duration_ms if summary else None,
            "outcome": summary.outcome if summary else "UNKNOWN",
            "reward": summary.reward if summary else None,
            "event_count": summary.event_count if summary else 0,
            "failure_labels": summary.failure_labels if summary else [],
            "artifacts": {
                "events_jsonl": str(self.events_jsonl_path(practice_id)),
                "timeline_jsonl": str(self.timeline_jsonl_path(practice_id)),
                "manifest": str(self.manifest_path(practice_id)),
                "frames": str(self.session_dir(practice_id) / "frames"),
                "artifacts": str(self.artifacts_dir(practice_id)),
                "provider": str(self.provider_dir(practice_id)),
                "sandbox": str(self.sandbox_dir(practice_id)),
                "runtime": str(self.runtime_dir(practice_id)),
                "metrics": str(self.metrics_dir(practice_id)),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def write_timeline_jsonl(self, practice_id: str, events: list[dict[str, Any]]) -> None:
        """Write the ordered timeline JSONL derived from raw events."""
        path = self.timeline_jsonl_path(practice_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def finalize_session(
        self,
        practice_id: str,
        session: PracticeSession,
        summary: PracticeSummary,
        sources: dict[str, bool] | None = None,
    ) -> None:
        """Write episode.json and timeline.jsonl from raw events."""
        self.write_episode_json(practice_id, session, summary, sources=sources)
        events: list[dict[str, Any]] = []
        jsonl_path = self.events_jsonl_path(practice_id)
        if jsonl_path.exists():
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    events = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                logger.warning("Failed to read events for timeline %s: %s", practice_id, e)
        self.write_timeline_jsonl(practice_id, events)


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def generate_practice_id() -> str:
    """Generate a unique practice id: prac_<utc_time>_<short_hash>."""
    from datetime import UTC, datetime
    from uuid import uuid4

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_hash = uuid4().hex[:6]
    return f"prac_{ts}_{short_hash}"
