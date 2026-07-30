"""Receipt / Trace / Practice link validator + legacy migration (v3 §5, PR-PE-1).

Two read-only validators:

* :func:`validate_session_links` — one session's internal evidence links
  (manifest, catalog, episode records, event identity).  A missing link
  is an explicit failure with a named link, never a silent skip.
* :func:`assess_legacy_sessions` + :func:`validate_legacy_manifest` —
  migration assessment for the pre-Lab corpora (evo_rps_2026_01 and
  earlier): every historical session stays READABLE, and whatever the
  old format never recorded is reported as an explicit missing link.
  Raw data is never modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_quality import DataQualityReport, run_data_quality_gate


@dataclass
class LinkReport:
    subject: str
    ok: bool
    missing_links: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "ok": self.ok,
            "missing_links": self.missing_links,
            "details": self.details,
        }


def validate_session_links(session_dir: str | Path) -> LinkReport:
    """Validate one session's internal evidence links (READ-ONLY)."""
    session_dir = Path(session_dir)
    missing: list[str] = []
    details: dict[str, Any] = {}

    events_path = session_dir / "raw" / "events.jsonl"
    if not events_path.is_file():
        return LinkReport(
            subject=str(session_dir),
            ok=False,
            missing_links=["raw/events.jsonl"],
        )

    manifest_path = session_dir / "manifest.yaml"
    if not manifest_path.is_file():
        missing.append("manifest.yaml")

    events: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        missing.append("events:empty")
    else:
        first = events[0]
        for field_name in ("practice_id", "session_id", "episode_id", "schema_version"):
            if not first.get(field_name):
                missing.append(f"events.identity.{field_name}")
        details["practice_id"] = first.get("practice_id")
        details["event_count"] = len(events)
        traced = sum(1 for e in events if e.get("trace_id"))
        if traced == 0:
            missing.append("trace_id:none")
        details["trace_linked_ratio"] = round(traced / len(events), 4)

    # Episode records: every episode_id referenced by events should have a
    # manifest/episode record file if the session layout provides one.
    episode_json = session_dir / "episode.json"
    if not episode_json.is_file():
        missing.append("episode.json")

    return LinkReport(
        subject=str(session_dir), ok=not missing, missing_links=missing, details=details
    )


def assess_legacy_sessions(
    sessions_root: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the data-quality gate + link validator over every session in a
    legacy root (READ-ONLY migration assessment, v3 PR-PE-1 acceptance:
    historical sessions readable, missing links named, raw untouched)."""
    root = Path(sessions_root)
    if not root.is_dir():
        return {"ok": False, "reason": f"no such sessions root: {root}"}

    session_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "raw" / "events.jsonl").is_file()
    )
    skipped = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not (p / "raw" / "events.jsonl").is_file()
    )
    if limit is not None:
        session_dirs = session_dirs[:limit]

    per_session: list[dict[str, Any]] = []
    totals = {
        "sessions": 0,
        "quality_passed": 0,
        "usable_for_memory": 0,
        "usable_for_training": 0,
        "usable_for_promotion": 0,
        "links_ok": 0,
    }
    missing_link_counter: dict[str, int] = {}
    for session_dir in session_dirs:
        quality: DataQualityReport = run_data_quality_gate(session_dir)
        links = validate_session_links(session_dir)
        totals["sessions"] += 1
        totals["quality_passed"] += int(quality.passed)
        totals["usable_for_memory"] += int(quality.usable_for_memory)
        totals["usable_for_training"] += int(quality.usable_for_training)
        totals["usable_for_promotion"] += int(quality.usable_for_promotion)
        totals["links_ok"] += int(links.ok)
        for link in links.missing_links:
            missing_link_counter[link] = missing_link_counter.get(link, 0) + 1
        per_session.append(
            {
                "session": session_dir.name,
                "practice_id": quality.practice_id,
                "quality_passed": quality.passed,
                "usable_for_memory": quality.usable_for_memory,
                "usable_for_training": quality.usable_for_training,
                "usable_for_promotion": quality.usable_for_promotion,
                "missing_ranges": quality.missing_ranges,
                "degraded_sensors": quality.degraded_sensors,
                "missing_links": links.missing_links,
            }
        )

    return {
        "ok": True,
        "sessions_root": str(root),
        "totals": totals,
        "skipped_non_session_dirs": len(skipped),
        "missing_link_histogram": dict(sorted(missing_link_counter.items(), key=lambda kv: -kv[1])),
        "read_only": True,
        "sessions": per_session,
    }


def validate_legacy_manifest(manifest_path: str | Path) -> LinkReport:
    """Validate a legacy evidence manifest's identity fields (READ-ONLY).

    Pre-Lab manifests associated evidence implicitly (directory names,
    timestamp strings); migration must name every entry that lacks an
    explicit identity link instead of guessing."""
    path = Path(manifest_path)
    if not path.is_file():
        return LinkReport(subject=str(path), ok=False, missing_links=["manifest:missing"])
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return LinkReport(subject=str(path), ok=False, missing_links=["manifest:invalid_json"])

    entries = data.get("entries") or []
    missing: list[str] = []
    details: dict[str, Any] = {"entries": len(entries)}
    no_practice_link = 0
    no_config_hash = 0
    for entry in entries:
        kind = entry.get("kind")
        if kind in {"baseline_session", "canary_session", "recurrence_session"} and not entry.get(
            "practice_id"
        ):
            no_practice_link += 1
    if not data.get("config_hash"):
        no_config_hash = 1
    if no_practice_link:
        missing.append(f"session_entries_without_practice_id:{no_practice_link}")
    if no_config_hash:
        missing.append("config_hash:missing")
    details["session_entries_without_practice_id"] = no_practice_link
    return LinkReport(subject=str(path), ok=not missing, missing_links=missing, details=details)
