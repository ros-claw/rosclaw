"""Tests for TelemetryConfig and FeedbackConfig."""

from __future__ import annotations

from pathlib import Path

from rosclaw.feedback.config import FeedbackConfig, TelemetryConfig


class TestTelemetryConfig:
    def test_defaults_match_spec(self) -> None:
        cfg = TelemetryConfig()
        assert cfg.mode["enabled"] is True
        assert cfg.mode["product_telemetry"] is True
        assert cfg.mode["diagnostics_upload"] is False
        assert cfg.product_telemetry["heartbeat_interval_hours"] == 24
        assert cfg.diagnostics["enabled"] is False
        assert cfg.rich_feedback["enabled"] is False
        assert cfg.upload["timeout_seconds"] == 3

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        cfg = TelemetryConfig()
        cfg.mode["enabled"] = False
        cfg.mode["product_telemetry"] = False
        path = cfg.save(home)

        assert path.exists()
        loaded = TelemetryConfig.load(home)
        assert loaded.mode["enabled"] is False
        assert loaded.mode["product_telemetry"] is False
        assert loaded.product_telemetry["heartbeat_interval_hours"] == 24

    def test_opt_out_persisted(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        cfg = TelemetryConfig()
        cfg.product_telemetry["enabled"] = False
        cfg.save(home)

        loaded = TelemetryConfig.load(home)
        assert loaded.product_telemetry["enabled"] is False


class TestFeedbackConfig:
    def test_defaults_match_spec(self) -> None:
        cfg = FeedbackConfig()
        assert cfg.mode["enabled"] is True
        assert cfg.mode["local_store"] is True
        assert cfg.mode["upload"] is False
        assert cfg.retention["local_days"] == 30
        assert cfg.collect["crash_reports"]["enabled"] is True
        assert cfg.redaction["text"]["replace_emails"] is True
        assert "/camera" in cfg.redaction["mcap"]["deny_topics"]

    def test_save_and_load(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        cfg = FeedbackConfig()
        cfg.mode["upload"] = True
        _path = cfg.save(home)

        loaded = FeedbackConfig.load(home)
        assert loaded.mode["upload"] is True
        assert loaded.retention["max_local_size_mb"] == 512
