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


class TestDoctorTelemetryEvents:
    def test_doctor_records_started_and_completed_events(self, tmp_path: Path, monkeypatch) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        monkeypatch.setenv("ROSCLAW_HOME", str(home))

        import sys
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "doctor"]
        main()

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        types = [e["event_type"] for e in events]
        assert "doctor_started" in types
        assert "doctor_completed" in types
        completed = [e for e in events if e["event_type"] == "doctor_completed"][0]
        assert completed["command_name"] == "doctor"
        assert completed["command_status"] in ("success", "failure")


class TestHeartbeatTelemetry:
    def test_heartbeat_if_due_writes_event_and_updates_last_heartbeat(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        client = TelemetryClient(home)

        # Force an old last heartbeat so the next call is due.
        last_path = home / "telemetry" / "heartbeat" / "last_heartbeat.json"
        last_path.parent.mkdir(parents=True, exist_ok=True)
        last_path.write_text('{"timestamp": "2020-01-01T00:00:00Z"}', encoding="utf-8")

        result = client.heartbeat_if_due()
        assert result is not None

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        assert any(e["event_type"] == "heartbeat" for e in events)
        assert last_path.exists()

    def test_heartbeat_skipped_when_anonymous_id_missing(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        client = TelemetryClient(home)
        result = client.heartbeat_if_due()
        assert result is None


class TestDeviceInfoTelemetry:
    def test_event_includes_device_info_payload(self, tmp_path: Path, monkeypatch) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        monkeypatch.setenv("ROS_DISTRO", "humble")
        # Reset module cache so the ROS probe re-runs.
        from rosclaw.feedback import telemetry_client as tc_mod
        tc_mod._CACHED_ROS_DISTROS = None

        client = TelemetryClient(home)
        client.record_event("command_completed", command_name="status", command_status="success")

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        payload = events[0].get("payload", {})
        assert "os_version" in payload
        assert "cuda_available" in payload
        assert payload.get("ros_distro_present") == "humble"
        # Forbidden fields must never appear even in auto-collected device info.
        assert "hostname" not in payload
        assert "ip" not in payload
        assert "username" not in payload

    def test_robot_type_read_from_rosclaw_yaml(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        config_path = home / "config" / "rosclaw.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "runtime:\n  robot_id: sim_ur5e\n",
            encoding="utf-8",
        )

        client = TelemetryClient(home)
        client.record_event("command_completed", command_name="status", command_status="success")

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        payload = events[0].get("payload", {})
        assert payload.get("robot_type") == "sim_ur5e"

    def test_sensor_types_read_from_body_yaml(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        body_path = home / "body" / "body.yaml"
        body_path.parent.mkdir(parents=True, exist_ok=True)
        body_path.write_text(
            "installed_components:\n  sensors:\n    imu:\n    camera:\n",
            encoding="utf-8",
        )

        client = TelemetryClient(home)
        client.record_event("command_completed", command_name="status", command_status="success")

        events = read_events(home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl")
        payload = events[0].get("payload", {})
        assert payload.get("sensor_types") == ["camera", "imu"]
