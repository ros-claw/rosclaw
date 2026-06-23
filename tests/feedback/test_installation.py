"""Tests for InstallationManager."""

from __future__ import annotations

from pathlib import Path

from rosclaw.feedback.installation import InstallationManager


class TestInstallationManager:
    def test_ensure_installation_creates_installation_json(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = InstallationManager(home)
        install = mgr.ensure_installation()

        assert (home / "config" / "installation.json").exists()
        assert (home / "telemetry" / "local_salt").exists()
        assert install.installation_id
        assert install.telemetry_enabled is True
        assert install.diagnostics_enabled is False
        assert install.rich_feedback_enabled is False

    def test_anonymous_id_is_stable(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = InstallationManager(home)
        mgr.ensure_installation()

        anon1 = mgr.get_anonymous_installation_id()
        anon2 = mgr.get_anonymous_installation_id()

        assert anon1 is not None
        assert anon1 == anon2
        assert anon1.startswith("rci_")
        assert len(anon1) == 36  # "rci_" + 32 hex chars

    def test_anonymous_id_changes_on_reset(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = InstallationManager(home)
        mgr.ensure_installation()
        anon1 = mgr.get_anonymous_installation_id()

        mgr.reset_installation_id()
        anon2 = mgr.get_anonymous_installation_id()

        assert anon2 is not None
        assert anon1 != anon2
        assert anon2.startswith("rci_")

    def test_set_telemetry_enabled_persists(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        mgr = InstallationManager(home)
        mgr.ensure_installation(telemetry_enabled=True)
        assert mgr.get_installation().telemetry_enabled is True

        mgr.set_telemetry_enabled(False)
        assert mgr.get_installation().telemetry_enabled is False

        mgr.set_telemetry_enabled(True)
        assert mgr.get_installation().telemetry_enabled is True

    def test_no_hostname_or_username_in_installation(self, tmp_path: Path) -> None:
        import socket

        home = tmp_path / ".rosclaw"
        mgr = InstallationManager(home)
        mgr.ensure_installation()

        raw = (home / "config" / "installation.json").read_text(encoding="utf-8")
        assert socket.gethostname() not in raw
        assert "username" not in raw.lower()
