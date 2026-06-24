"""Feedback and telemetry directory layout helpers."""

from __future__ import annotations

from pathlib import Path

TELEMETRY_DIRS = [
    "events",
    "heartbeat",
    "uploads",
]

FEEDBACK_DIRS = [
    "events",
    "crashes",
    "bundles",
    "redacted",
    "media/local_only",
    "consent",
]


def ensure_feedback_dirs(home: Path) -> None:
    """Create the telemetry and feedback directory trees under `home`."""
    for rel in TELEMETRY_DIRS:
        (home / "telemetry" / rel).mkdir(parents=True, exist_ok=True)
    for rel in FEEDBACK_DIRS:
        (home / "feedback" / rel).mkdir(parents=True, exist_ok=True)
