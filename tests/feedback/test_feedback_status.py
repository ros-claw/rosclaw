"""Tests for `rosclaw feedback status` output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rosclaw.feedback.cli import cmd_feedback_status
from rosclaw.feedback.installation import InstallationManager


class TestFeedbackStatus:
    def test_status_shows_anonymous_id(self, tmp_path: Path, capsys) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        args = argparse.Namespace(workspace=str(home))

        code = cmd_feedback_status(args)
        captured = capsys.readouterr()

        assert code == 0
        assert "ROSClaw Feedback & Telemetry Status" in captured.out
        assert "anonymous_installation_id:" in captured.out
        assert "rci_" in captured.out

    def test_status_local_store_section(self, tmp_path: Path, capsys) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        args = argparse.Namespace(workspace=str(home))

        cmd_feedback_status(args)
        captured = capsys.readouterr()

        assert "Local Store:" in captured.out
        assert re.search(r"telemetry_events:\s+\d+", captured.out)
