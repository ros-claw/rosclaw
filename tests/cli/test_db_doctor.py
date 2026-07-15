"""Tests for rosclaw db status / doctor CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path


class TestDbStatus:
    def test_db_status_memory_json(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "db", "status", "--backend", "memory", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["backend"] == "memory"
        assert data["ping"]["connected"] is True
        assert data["capabilities"]["persistent"] is False

    def test_db_status_sqlite_json(self, tmp_path: Path, capsys):
        from rosclaw.cli import main

        db_path = tmp_path / "knowledge.sqlite"
        sys.argv = ["rosclaw", "db", "status", "--backend", "sqlite", "--path", str(db_path), "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["backend"] == "sqlite"
        assert data["ping"]["connected"] is True
        assert data["capabilities"]["sqlite"] is True
        assert isinstance(data["ping"]["wal_size_bytes"], int)

    def test_db_status_unknown_backend(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "db", "status", "--backend", "postgres", "--json"]
        assert main() == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["connected"] is False
        assert "postgres" in data["error"]


class TestDbDoctor:
    def test_db_doctor_memory_json(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "db", "doctor", "--backend", "memory", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["checks"]
        assert not data["issues"]

    def test_db_doctor_sqlite_creates_migrations(self, tmp_path: Path, capsys):
        from rosclaw.cli import main

        db_path = tmp_path / "knowledge.sqlite"
        sys.argv = [
            "rosclaw",
            "db",
            "doctor",
            "--backend",
            "sqlite",
            "--path",
            str(db_path),
            "--fix",
            "--json",
        ]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert not data["issues"]
        # The baseline migration should have been applied.
        names = {c["name"] for c in data["checks"]}
        assert "migrations" in names
        assert "schema_migrations table" in names

    def test_db_doctor_detects_http_url_mismatch(self, capsys):
        from rosclaw.cli import main

        sys.argv = [
            "rosclaw",
            "db",
            "doctor",
            "--backend",
            "mysql",
            "--url",
            "http://localhost:2882",
            "--json",
        ]
        assert main() == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert any("HTTP" in issue for issue in data["issues"])

    def test_db_doctor_outbox_stats(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        # Create a rosclaw.yaml that enables the outbox.
        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: memory\n"
            f"storage:\n  outbox_enabled: true\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "doctor", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["outbox"]["total"] == 0
        assert data["outbox"]["pending"] == 0


class TestDbSubcommandHelp:
    def test_db_prints_help(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "db"]
        assert main() == 1
        captured = capsys.readouterr()
        assert "status" in captured.out
        assert "doctor" in captured.out
