"""Tests for TelemetryClient event recording."""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from rosclaw.feedback.installation import InstallationManager
from rosclaw.feedback.store import read_events
from rosclaw.feedback.telemetry_client import TelemetryClient, duration_bucket, error_class_bucket


class TestDurationAndErrorBuckets:
    def test_duration_bucket(self) -> None:
        assert duration_bucket(50) == "<100ms"
        assert duration_bucket(500) == "100ms-1s"
        assert duration_bucket(2500) == "1s-5s"
        assert duration_bucket(20000) == "5s-30s"
        assert duration_bucket(120000) == "30s-5m"
        assert duration_bucket(400000) == ">5m"

    def test_error_class_bucket(self) -> None:
        assert error_class_bucket(None) is None
        assert error_class_bucket(ValueError("boom")) == "ValidationError"
        assert error_class_bucket(ImportError("no module")) == "ImportError"
        assert error_class_bucket(RuntimeError("oops")) == "RuntimeError"
        assert error_class_bucket(TypeError("bad")) == "Unknown"


class TestTelemetryClientRecording:
    def test_record_event_writes_jsonl(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        client = TelemetryClient(home)

        client.record_event("command_completed", command_name="doctor", command_status="success")

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        assert len(events) == 1
        assert events[0]["event_type"] == "command_completed"
        assert events[0]["command_name"] == "doctor"
        assert events[0]["command_status"] == "success"
        assert events[0]["anonymous_installation_id"].startswith("rci_")

    def test_record_command_no_args_captured(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        client = TelemetryClient(home)

        class Args:
            command = "doctor"
            workspace = str(home)
            secret_arg = "super-secret"

        client.record_command(Args(), "success", 250)

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        payload = json.dumps(events[0])
        assert "secret_arg" not in payload
        assert "super-secret" not in payload

    def test_forbidden_fields_scrubbed(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        client = TelemetryClient(home)

        client.record_event(
            "command_completed",
            payload={"hostname": "mybox", "ip": "1.2.3.4", "allowed": True},
        )

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        payload = events[0].get("payload", {})
        assert "hostname" not in payload
        assert "ip" not in payload
        assert payload.get("allowed") is True

    @patch("urllib.request.urlopen")
    def test_upload_failure_silent(self, mock_urlopen, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        mock_urlopen.side_effect = urllib.error.URLError("offline")

        client = TelemetryClient(home)
        # Should not raise.
        client.record_event("command_completed", command_name="status")

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        assert len(events) == 1
