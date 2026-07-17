"""Helpers for loading recorded Practice episode evidence.

Practice v1 callers addressed episodes by ``sessions/{episode_id}``.  The v2
recorder stores raw events under ``sessions/{practice_id}`` and links
``episode_id`` through the SQLite catalog.  This module keeps that compatibility
logic in one place for Memory, Know, and How.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.practice.config import resolve_data_root
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout


@dataclass
class PracticeEpisodeEvidence:
    """Resolved episode evidence and source paths."""

    data_root: Path
    query_episode_id: str
    session_dir: Path
    episode_path: Path
    manifest_path: Path
    events_path: Path
    provider_path: Path
    episode: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    provider: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    practice: dict[str, Any] | None = None
    found: bool = True
    reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def sources(self) -> list[str]:
        sources: set[str] = set()
        for event in self.events:
            source = event.get("source")
            if isinstance(source, str) and source:
                sources.add(source)
        return sorted(sources)


def load_episode_evidence(
    episode_id: str,
    data_root: str | Path | None = None,
) -> PracticeEpisodeEvidence:
    """Load episode summary, raw events, and provider result by episode id."""

    root = resolve_data_root(data_root)
    layout = PracticeLayout(root)

    legacy = _load_from_paths(layout, episode_id, session_dir=layout.session_dir(episode_id))
    if legacy.found:
        return legacy

    catalog_result = _load_from_catalog(layout, episode_id)
    if catalog_result is not None:
        return catalog_result

    return legacy


def _load_from_paths(
    layout: PracticeLayout,
    query_episode_id: str,
    *,
    session_dir: Path,
    episode_path: Path | None = None,
    manifest_path: Path | None = None,
    events_path: Path | None = None,
    provider_path: Path | None = None,
    session: dict[str, Any] | None = None,
    practice: dict[str, Any] | None = None,
    catalog_episode: dict[str, Any] | None = None,
) -> PracticeEpisodeEvidence:
    episode_path = episode_path or session_dir / "episode.json"
    manifest_path = manifest_path or session_dir / "manifest.yaml"
    events_path = events_path or session_dir / "raw" / "events.jsonl"
    provider_path = provider_path or session_dir / "provider" / "provider_result.json"

    if not session_dir.exists():
        return PracticeEpisodeEvidence(
            data_root=layout.data_root,
            query_episode_id=query_episode_id,
            session_dir=session_dir,
            episode_path=episode_path,
            manifest_path=manifest_path,
            events_path=events_path,
            provider_path=provider_path,
            session=session,
            practice=practice,
            found=False,
            reason=f"session not found: {session_dir}",
        )

    errors: list[str] = []
    episode: dict[str, Any] = {}
    if episode_path.exists():
        episode = _read_json_object(episode_path, errors, label="episode.json")
    elif manifest_path.exists():
        episode = _read_yaml_object(manifest_path, errors, label="manifest.yaml")

    episode = _merge_catalog_fields(
        episode,
        query_episode_id=query_episode_id,
        catalog_episode=catalog_episode,
        session=session,
        practice=practice,
    )

    events = _read_jsonl_objects(events_path, errors, label="events.jsonl")
    provider = None
    if provider_path.exists():
        provider_errors: list[str] = []
        provider = _read_json_object(
            provider_path,
            provider_errors,
            label="provider_result.json",
        )

    return PracticeEpisodeEvidence(
        data_root=layout.data_root,
        query_episode_id=query_episode_id,
        session_dir=session_dir,
        episode_path=episode_path,
        manifest_path=manifest_path,
        events_path=events_path,
        provider_path=provider_path,
        episode=episode,
        events=events,
        provider=provider,
        session=session,
        practice=practice,
        found=True,
        errors=errors,
    )


def _load_from_catalog(
    layout: PracticeLayout,
    episode_id: str,
) -> PracticeEpisodeEvidence | None:
    if not layout.catalog_db_path.exists():
        return None

    with PracticeCatalog(layout.catalog_db_path) as catalog:
        catalog_episode = catalog.get_episode(episode_id)
        session: dict[str, Any] | None = None
        practice: dict[str, Any] | None = None

        if catalog_episode:
            session_id = _as_nonempty_str(catalog_episode.get("session_id"))
            if session_id:
                session = catalog.get_session(session_id)
            practice_id = _as_nonempty_str(session.get("practice_id") if session else None)
            if practice_id:
                practice = catalog.get_practice(practice_id)

        if practice is None:
            practice = _find_practice_for_episode(catalog, episode_id)
            if practice and session is None:
                session_id = _as_nonempty_str(practice.get("session_id"))
                if session_id:
                    session = catalog.get_session(session_id)

    practice_id = _as_nonempty_str(practice.get("practice_id") if practice else None)
    if not practice_id:
        return None

    session_dir = layout.session_dir(practice_id)
    episode_path = _catalog_path(
        practice.get("episode_json_path") if practice else None,
        fallback=layout.episode_json_path(practice_id),
        root=layout.data_root,
    )
    manifest_path = _catalog_path(
        practice.get("manifest_path") if practice else None,
        fallback=layout.manifest_path(practice_id),
        root=layout.data_root,
    )
    events_path = _catalog_path(
        practice.get("events_jsonl_path") if practice else None,
        fallback=layout.events_jsonl_path(practice_id),
        root=layout.data_root,
    )
    provider_path = layout.provider_dir(practice_id) / "provider_result.json"

    return _load_from_paths(
        layout,
        episode_id,
        session_dir=session_dir,
        episode_path=episode_path,
        manifest_path=manifest_path,
        events_path=events_path,
        provider_path=provider_path,
        session=session,
        practice=practice,
        catalog_episode=catalog_episode,
    )


def _find_practice_for_episode(
    catalog: PracticeCatalog,
    episode_id: str,
) -> dict[str, Any] | None:
    for row in catalog.list_practices(limit=10_000):
        if row.get("episode_id") == episode_id:
            return row
    return None


def _merge_catalog_fields(
    episode: dict[str, Any],
    *,
    query_episode_id: str,
    catalog_episode: dict[str, Any] | None,
    session: dict[str, Any] | None,
    practice: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(episode)
    merged.setdefault("episode_id", query_episode_id)

    for key in ("practice_id", "session_id", "robot_id", "robot_type", "outcome", "reward"):
        value = _first_present(practice, session, catalog_episode, key=key)
        if value is not None and merged.get(key) in (None, ""):
            merged[key] = value

    body_id = _first_present(session, catalog_episode, practice, key="body_id")
    if body_id is not None and merged.get("body_id") in (None, ""):
        merged["body_id"] = body_id

    duration_ms = _first_present(practice, session, catalog_episode, key="duration_ms")
    if duration_ms is not None and merged.get("duration_ms") in (None, ""):
        merged["duration_ms"] = duration_ms

    task = {} if not isinstance(merged.get("task"), dict) else dict(merged["task"])

    for key in ("task_id", "task_name", "skill_id"):
        value = _first_present(practice, catalog_episode, session, key=key)
        if value is not None and task.get(key) in (None, ""):
            task[key] = value
    if task:
        merged["task"] = task

    if catalog_episode and "practice_episode" not in merged:
        merged["practice_episode"] = catalog_episode
    if session and "practice_session" not in merged:
        merged["practice_session"] = session

    return merged


def _read_json_object(path: Path, errors: list[str], *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to parse {label}: {exc}")
        return {}
    if isinstance(data, dict):
        return data
    errors.append(f"failed to parse {label}: expected object")
    return {}


def _read_yaml_object(path: Path, errors: list[str], *, label: str) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to parse {label}: {exc}")
        return {}
    if isinstance(data, dict):
        return data
    errors.append(f"failed to parse {label}: expected object")
    return {}


def _read_jsonl_objects(path: Path, errors: list[str], *, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
                else:
                    errors.append(f"failed to parse {label}: line {line_number} is not an object")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to parse {label}: {exc}")
        return []
    return records


def _catalog_path(value: Any, *, fallback: Path, root: Path) -> Path:
    path_text = _as_nonempty_str(value)
    if not path_text:
        return fallback
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _as_nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _first_present(*records: dict[str, Any] | None, key: str) -> Any:
    for record in records:
        if not record:
            continue
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None
