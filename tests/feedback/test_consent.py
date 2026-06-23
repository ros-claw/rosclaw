"""Tests for ConsentManager and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from rosclaw.feedback.cli import cmd_feedback_consent
from rosclaw.feedback.consent import ConsentManager


class TestConsentManager:
    def test_show_defaults(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        state = ConsentManager(home).show()
        assert state.product_telemetry is True
        assert state.diagnostics is False
        assert state.rich_feedback is False

    def test_set_diagnostics(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = ConsentManager(home)
        state = mgr.set_diagnostics(True)
        assert state.diagnostics is True
        assert (home / "config" / "installation.json").exists()

    def test_revoke_all(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = ConsentManager(home)
        mgr.set_diagnostics(True)
        mgr.set_rich_feedback(True)
        state = mgr.revoke_all()
        assert state.diagnostics is False
        assert state.rich_feedback is False


class TestConsentCLI:
    def test_consent_show(self, tmp_path: Path, capsys) -> None:
        home = tmp_path / ".rosclaw"
        args = argparse.Namespace(
            workspace=str(home),
            diagnostics=False,
            rich_feedback=False,
            revoke_diagnostics=False,
            revoke_all=False,
            show=True,
        )
        code = cmd_feedback_consent(args)
        captured = capsys.readouterr()
        assert code == 0
        assert "Product telemetry:" in captured.out
