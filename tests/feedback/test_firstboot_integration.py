"""Integration tests for firstboot telemetry wiring."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import yaml


class TestFirstbootTelemetryIntegration:
    def test_firstboot_creates_telemetry_and_feedback_dirs(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--telemetry"]
        code = main()
        assert code in (0, 2)

        assert (home / "telemetry" / "events").exists()
        assert (home / "feedback" / "events").exists()
        assert (home / "config" / "telemetry.yaml").exists()
        assert (home / "config" / "feedback.yaml").exists()
        assert (home / "config" / "installation.json").exists()

    def test_firstboot_records_firstboot_completed_event(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline", "--telemetry"]
        main()

        events_path = home / "telemetry" / "events" / f"{datetime.now(UTC).date().isoformat()}.jsonl"
        if events_path.exists():
            lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
            types = [e["event_type"] for e in lines]
            assert "firstboot_completed" in types

    def test_firstboot_telemetry_default_on(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / ".rosclaw"
        monkeypatch.setenv("ROSCLAW_HOME", str(home))
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "firstboot", "--yes", "--profile", "offline"]
        main()

        cfg = yaml.safe_load((home / "config" / "telemetry.yaml").read_text(encoding="utf-8"))
        assert cfg["mode"]["product_telemetry"] is True
