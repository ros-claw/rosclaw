"""Tests for FeedbackExporter."""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.feedback.export import FeedbackExporter
from rosclaw.feedback.installation import InstallationManager
from rosclaw.feedback.store import append_event
from rosclaw.feedback.telemetry_client import TelemetryClient


class TestFeedbackExporter:
    def test_export_creates_tar_gz(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        client = TelemetryClient(home)
        client.record_event("command_completed", command_name="status", command_status="success")

        output = FeedbackExporter(home).export(days=7, redact=False)

        assert output.exists()
        assert output.suffix == ".gz"
        with tarfile.open(output, "r:gz") as tar:
            names = tar.getnames()
            assert "telemetry.jsonl" in names
            assert "manifest.json" in names

    def test_export_redacts(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        append_event(
            home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl",
            {"event_type": "test", "message": "email me at foo@example.com"},
        )

        output = FeedbackExporter(home).export(days=7, redact=True)

        with tarfile.open(output, "r:gz") as tar:
            member = tar.extractfile("telemetry.jsonl")
            lines = member.read().decode("utf-8").strip().split("\n")
            record = json.loads(lines[0])
            assert "[REDACTED_EMAIL]" in record["message"]
