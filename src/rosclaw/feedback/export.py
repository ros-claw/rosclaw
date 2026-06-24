"""Export local telemetry/feedback events into a redacted bundle."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import FeedbackConfig
from .redactor.text_redactor import TextRedactor
from .store import directory_size_mb, event_file_for_date, read_events


class FeedbackExporter:
    """Export recent telemetry and feedback events to a local bundle."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.feedback_config = FeedbackConfig.load(home)
        self.text_redactor = TextRedactor(
            enabled=self.feedback_config.redaction.get("text", {}).get("enabled", True),
        )

    def export(
        self,
        *,
        days: int = 30,
        redact: bool = False,
        output_path: Path | str | None = None,
    ) -> Path:
        """Collect events and write a gzipped tar archive."""
        output_path = Path(output_path) if output_path else self._default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry_records = self._collect_events("telemetry", days, redact)
        feedback_records = self._collect_events("feedback", days, redact)

        with tarfile.open(output_path, "w:gz") as tar:
            self._add_jsonl(tar, telemetry_records, "telemetry.jsonl")
            self._add_jsonl(tar, feedback_records, "feedback.jsonl")
            manifest = {
                "schema_version": "rosclaw.feedback.export.v1",
                "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "days": days,
                "redacted": redact,
                "telemetry_event_count": len(telemetry_records),
                "feedback_event_count": len(feedback_records),
                "local_size_mb": directory_size_mb(self.home / "telemetry")
                + directory_size_mb(self.home / "feedback"),
            }
            self._add_json(tar, manifest, "manifest.json")

        return output_path

    def _collect_events(
        self,
        kind: str,
        days: int,
        redact: bool,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        today = datetime.now(UTC).date()
        for offset in range(days + 1):
            date = datetime.fromordinal(today.toordinal() - offset).replace(tzinfo=UTC)
            path = event_file_for_date(self.home, kind, date=date)
            for record in read_events(path):
                if redact:
                    record = self._redact_record(record)
                records.append(record)
        return records

    def _redact_record(self, record: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, str):
                result[key] = self.text_redactor.redact(value)
            elif isinstance(value, dict):
                result[key] = self._redact_record(value)
            elif isinstance(value, list):
                result[key] = [
                    self.text_redactor.redact(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    @staticmethod
    def _add_jsonl(tar: tarfile.TarFile, records: list[dict[str, Any]], name: str) -> None:
        if not records:
            return
        text = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
        encoded = text.encode("utf-8")
        info = tarfile.TarInfo(name=name)
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))

    @staticmethod
    def _add_json(tar: tarfile.TarFile, data: dict[str, Any], name: str) -> None:
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name=name)
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))

    def _default_output_path(self) -> Path:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return self.home / "feedback" / "bundles" / f"feedback_export_{ts}.tar.gz"
