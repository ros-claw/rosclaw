"""Tests for practice camera episode directory schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.practice.storage.layout import PracticeLayout


def test_create_session_dirs_v2_schema(tmp_path: Path):
    """V2 session directory must contain expected subdirectories."""
    layout = PracticeLayout(tmp_path)
    practice_id = "prac_20240101T000000Z_abc123"
    session_dir = layout.create_session_dirs(practice_id)
    assert session_dir.exists()
    expected = [
        "raw",
        "frames/d405",
        "frames/d435i",
        "imu",
        "provider",
        "sandbox",
        "runtime",
        "metrics",
        "derived",
        "exports",
        "replay",
        "reports",
        "index",
    ]
    for sub in expected:
        assert (session_dir / sub).is_dir(), f"Missing {sub}"


def test_helper_methods_resolve(tmp_path: Path):
    """Helper methods should return correct paths."""
    layout = PracticeLayout(tmp_path)
    pid = "prac_test"
    assert layout.frames_dir(pid, "d405") == layout.session_dir(pid) / "frames" / "d405"
    assert layout.frames_dir(pid, "d435i") == layout.session_dir(pid) / "frames" / "d435i"
    assert layout.imu_dir(pid) == layout.session_dir(pid) / "imu"
    assert layout.provider_dir(pid) == layout.session_dir(pid) / "provider"
    assert layout.sandbox_dir(pid) == layout.session_dir(pid) / "sandbox"
    assert layout.runtime_dir(pid) == layout.session_dir(pid) / "runtime"
    assert layout.metrics_dir(pid) == layout.session_dir(pid) / "metrics"
