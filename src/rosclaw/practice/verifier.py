"""Practice verification logic for the ROSClaw data closed-loop.

``PracticeVerifier`` validates that a recorded practice session is complete,
consistent, and ready for distillation/export. It checks both the catalog
(SQLite) and the filesystem artifacts against their registered checksums.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.practice.artifact_store import ArtifactStore
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout

logger = logging.getLogger("rosclaw.practice.verifier")


@dataclass
class VerificationIssue:
    level: str  # "error" or "warning"
    scope: str
    message: str


@dataclass
class VerificationReport:
    practice_id: str
    passed: bool
    strict: bool
    issues: list[VerificationIssue] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def add(self, level: str, scope: str, message: str) -> None:
        self.issues.append(VerificationIssue(level=level, scope=scope, message=message))

    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)


class PracticeVerifier:
    """Validate a practice session and its artifacts."""

    def __init__(self, data_root: Path | str):
        self._data_root = Path(data_root)
        self._layout = PracticeLayout(self._data_root)
        self._artifact_store = ArtifactStore(self._data_root, layout=self._layout)

    def verify(
        self,
        practice_id: str,
        *,
        strict: bool = False,
        required_event_types: list[str] | None = None,
    ) -> VerificationReport:
        """Verify a practice session by id."""
        report = VerificationReport(practice_id=practice_id, passed=False, strict=strict)

        catalog_path = self._layout.catalog_db_path
        if not catalog_path.exists():
            report.add("error", "catalog", f"catalog sqlite missing: {catalog_path}")
            return report

        report.checked.append("catalog_exists")
        catalog = PracticeCatalog(catalog_path)
        try:
            self._verify_practice(catalog, practice_id, report)
            self._verify_session(catalog, practice_id, report)
            self._verify_episodes(catalog, practice_id, report, required_event_types)
            self._verify_artifacts(catalog, practice_id, report)
        finally:
            catalog.close()

        report.passed = not report.has_errors() and (not strict or not report.has_warnings())
        return report

    def _verify_practice(
        self, catalog: PracticeCatalog, practice_id: str, report: VerificationReport
    ) -> None:
        practice = catalog.get_practice(practice_id)
        report.checked.append("practice_record")
        if practice is None:
            report.add("error", "practice", f"practice {practice_id} not found in catalog")
            return

        manifest_path = Path(practice.get("manifest_path", ""))
        if not manifest_path.exists():
            report.add("error", "manifest", f"manifest missing: {manifest_path}")
        else:
            report.checked.append("manifest_exists")

        events_path = Path(practice.get("events_jsonl_path", ""))
        if not events_path.exists():
            report.add("error", "events_jsonl", f"events.jsonl missing: {events_path}")
        else:
            report.checked.append("events_jsonl_exists")
            self._verify_jsonl(events_path, report)

    def _verify_session(
        self, catalog: PracticeCatalog, practice_id: str, report: VerificationReport
    ) -> None:
        practice = catalog.get_practice(practice_id)
        if practice is None:
            return
        session_id = practice.get("session_id")
        if not session_id:
            report.add(
                "warning" if not report.strict else "error", "session", "practice has no session_id"
            )
            return

        session = catalog.get_session(session_id)
        report.checked.append("session_record")
        if session is None:
            report.add(
                "warning" if not report.strict else "error",
                "session",
                f"session {session_id} not found in catalog v2",
            )
            return

        # The filesystem layout names the session directory by practice_id,
        # not by the catalog v2 session_id, so resolve it from the practice record.
        practice = catalog.get_practice(practice_id)
        session_dir = self._layout.session_dir(practice_id)
        if practice is not None and practice.get("session_id") == session_id:
            pass
        if not session_dir.exists():
            report.add("error", "session_dir", f"session directory missing: {session_dir}")

    def _verify_episodes(
        self,
        catalog: PracticeCatalog,
        practice_id: str,
        report: VerificationReport,
        required_event_types: list[str] | None,
    ) -> None:
        practice = catalog.get_practice(practice_id)
        if practice is None:
            return
        session_id = practice.get("session_id")
        if not session_id:
            return

        episodes = catalog.list_episodes(session_id=session_id)
        report.checked.append("episode_records")
        if not episodes:
            report.add(
                "warning" if not report.strict else "error", "episodes", "no episode records found"
            )
            return

        for episode in episodes:
            episode_id = episode.get("episode_id")
            if episode.get("outcome") is None:
                report.add("warning", "episode", f"episode {episode_id} has no outcome")

        # Check required event types from the raw JSONL.
        events_path = Path(practice.get("events_jsonl_path", ""))
        if events_path.exists():
            event_types: set[str] = set()
            failure_events: list[dict[str, Any]] = []
            how_events: list[dict[str, Any]] = []
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError as e:
                        report.add("error", "events_jsonl", f"invalid JSON line: {e}")
                        continue
                    event_types.add(ev.get("event_type", ""))
                    if ev.get("event_type") == "failure_event":
                        failure_events.append(ev)
                    if ev.get("event_type") == "how_intervention_event":
                        how_events.append(ev)

            report.checked.append("event_types")
            required = required_event_types or []
            missing = [t for t in required if t not in event_types]
            for t in missing:
                report.add("error", "event_types", f"missing required event type: {t}")

            for ev in failure_events:
                payload = ev.get("payload", {})
                if not payload.get("failure_id"):
                    report.add("error", "failure_event", "failure_event missing failure_id")

            for ev in how_events:
                payload = ev.get("payload", {})
                if not payload.get("failure_id"):
                    report.add(
                        "error", "how_intervention", "how_intervention_event missing failure_id"
                    )
                if not payload.get("episode_id"):
                    report.add(
                        "warning", "how_intervention", "how_intervention_event missing episode_id"
                    )

            catalog_count = catalog.count_source_events(practice_id)
            jsonl_count = 0
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError as e:
                        report.add(
                            "error", "events_jsonl", f"invalid JSON line while counting events: {e}"
                        )
                        continue
                    if ev.get("event_type") not in {"runtime.start", "runtime.stop"}:
                        jsonl_count += 1
            report.checked.append("event_count")
            if catalog_count != jsonl_count:
                report.add(
                    "error",
                    "event_count",
                    f"catalog event count ({catalog_count}) != jsonl line count ({jsonl_count})",
                )

    def _verify_artifacts(
        self, catalog: PracticeCatalog, practice_id: str, report: VerificationReport
    ) -> None:
        practice = catalog.get_practice(practice_id)
        if practice is None:
            return
        session_id = practice.get("session_id")
        if not session_id:
            return

        artifacts = catalog.list_artifacts_v2(session_id=session_id)
        report.checked.append("artifact_records")
        if not artifacts:
            report.add(
                "warning" if not report.strict else "error",
                "artifacts",
                "no v2 artifact records found",
            )
            return

        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id")
            if not isinstance(artifact_id, str):
                report.add("error", "artifact", "artifact record missing artifact_id")
                continue
            artifact_path = Path(artifact.get("path") or "")
            expected_sha = artifact.get("sha256")
            if not artifact_path.exists():
                report.add("error", "artifact", f"{artifact_id}: file missing: {artifact_path}")
                continue
            actual_sha = ArtifactStore._compute_sha256(artifact_path)
            if expected_sha and actual_sha != expected_sha:
                report.add(
                    "error",
                    "artifact",
                    f"{artifact_id}: sha256 mismatch",
                )
                continue
            # size_bytes consistency
            expected_size = artifact.get("size_bytes")
            if expected_size is not None and artifact_path.stat().st_size != expected_size:
                report.add(
                    "warning",
                    "artifact",
                    f"{artifact_id}: size mismatch",
                )

    @staticmethod
    def _verify_jsonl(path: Path, report: VerificationReport) -> None:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    report.add("error", "events_jsonl", f"line {i}: {e}")
                    break
                if not isinstance(event, dict):
                    report.add("error", "events_jsonl", f"line {i}: event is not an object")
                    continue

                required_fields = (
                    "event_id",
                    "event_type",
                    "trace_id",
                    "timestamp_ns",
                    "timestamp_utc",
                )
                for field_name in required_fields:
                    if event.get(field_name) in (None, ""):
                        level = "error" if report.strict else "warning"
                        report.add(
                            level,
                            "event_envelope",
                            f"line {i}: missing required field: {field_name}",
                        )
                if "timestamp_ns" in event and not isinstance(event.get("timestamp_ns"), int):
                    level = "error" if report.strict else "warning"
                    report.add(
                        level, "event_envelope", f"line {i}: timestamp_ns must be an integer"
                    )


def format_report(report: VerificationReport) -> str:
    lines = [
        f"Practice: {report.practice_id}",
        f"Mode: {'strict' if report.strict else 'normal'}",
        f"Passed: {report.passed}",
        f"Checked: {', '.join(report.checked)}",
    ]
    if report.issues:
        lines.append("Issues:")
        for issue in report.issues:
            lines.append(f"  [{issue.level}] {issue.scope}: {issue.message}")
    else:
        lines.append("Issues: none")
    return "\n".join(lines)
