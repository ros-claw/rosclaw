"""Telemetry and feedback configuration generation for ROSClaw First Boot."""

from __future__ import annotations

from pathlib import Path

import yaml

from rosclaw.feedback.config import FeedbackConfig, TelemetryConfig


def generate_telemetry_yaml(home: Path, enabled: bool = True) -> Path:
    """Generate telemetry.yaml with the full v1 product telemetry spec."""
    config = TelemetryConfig()
    config.mode["enabled"] = True
    config.mode["product_telemetry"] = enabled
    config.mode["diagnostics_upload"] = False
    config.mode["rich_feedback_upload"] = False
    config.product_telemetry["enabled"] = enabled
    config.product_telemetry["opt_out"] = True
    config.diagnostics["enabled"] = False
    config.rich_feedback["enabled"] = False
    return config.save(home)


def generate_feedback_yaml(home: Path) -> Path:
    """Generate feedback.yaml with the full v1 local feedback spec."""
    return FeedbackConfig().save(home)


def load_telemetry_yaml(home: Path) -> dict:
    """Load telemetry.yaml if present, returning an empty dict on error."""
    path = home / "config" / "telemetry.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}


def is_product_telemetry_enabled(home: Path) -> bool:
    """Return whether product telemetry is currently enabled."""
    cfg = load_telemetry_yaml(home)
    mode = cfg.get("mode", {})
    return bool(mode.get("enabled", True) and mode.get("product_telemetry", True))
