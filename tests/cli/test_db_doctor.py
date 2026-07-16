"""Tests for rosclaw db status / doctor CLI commands."""

from __future__ import annotations

import contextlib
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
        sys.argv = [
            "rosclaw",
            "db",
            "status",
            "--backend",
            "sqlite",
            "--path",
            str(db_path),
            "--json",
        ]
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
    def test_db_doctor_memory_json(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()
        practice_root = tmp_path / "practice"
        practice_root.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: memory\n"
            f"practice:\n  output_dir: {practice_root}\n",
            encoding="utf-8",
        )

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
        practice_root = tmp_path / "practice"
        practice_root.mkdir()

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
            f"practice:\n  output_dir: {practice_root}\n"
            f"storage:\n  outbox_enabled: true\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "doctor", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["outbox"]["total"] == 0
        assert data["outbox"]["pending"] == 0

    def test_db_doctor_detects_event_count_mismatch(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()
        practice_root = tmp_path / "practice"
        practice_root.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: memory\n"
            f"practice:\n  output_dir: {practice_root}\n",
            encoding="utf-8",
        )

        session_dir = practice_root / "sessions" / "prac_test_0001"
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True)
        session_dir.joinpath("episode.json").write_text(
            json.dumps({"event_count": 5}),
            encoding="utf-8",
        )
        raw_dir.joinpath("events.jsonl").write_text(
            "{}\n{}\n{}\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "doctor", "--json"]
        assert main() == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert any("event_count" in issue for issue in data["issues"])
        assert data["latest_session"]["events_jsonl_lines"] == 3
        assert data["latest_session"]["event_count"] == 5

    def test_db_doctor_detects_timeline_residual(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()
        practice_root = tmp_path / "practice"
        practice_root.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: memory\n"
            f"practice:\n  output_dir: {practice_root}\n",
            encoding="utf-8",
        )

        session_dir = practice_root / "sessions" / "prac_test_0002"
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True)
        session_dir.joinpath("episode.json").write_text(
            json.dumps({"event_count": 1}),
            encoding="utf-8",
        )
        raw_dir.joinpath("events.jsonl").write_text("{}\n", encoding="utf-8")
        session_dir.joinpath("timeline.jsonl").write_text("{}\n", encoding="utf-8")

        sys.argv = ["rosclaw", "db", "doctor", "--json"]
        assert main() == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert any("timeline.jsonl" in issue for issue in data["issues"])


class TestDbStatusExtensions:
    def test_db_status_reports_vector_disabled_for_memory(self, capsys):
        from rosclaw.cli import main

        sys.argv = ["rosclaw", "db", "status", "--backend", "memory", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["vector"]["enabled"] is False

    def test_db_status_reports_vector_enabled(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: sqlite\n  seekdb_path: {tmp_path / 'knowledge.sqlite'}\n"
            f"storage:\n  vector_enabled: true\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "status", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["vector"]["enabled"] is True


class TestMigrationStatusHelper:
    def test_migration_status_sqlite_reports_pending(self, monkeypatch):
        from rosclaw.storage.cli import _migration_status

        class FakeClient:
            _connection = object()

        monkeypatch.setattr(
            "rosclaw.storage.cli.MigrationRunner.pending",
            lambda _self, _connection, _backend: ["001", "002"],
        )
        result = _migration_status(FakeClient(), "sqlite")
        assert result == {"pending": 2, "versions": ["001", "002"]}

    def test_migration_status_memory_is_zero(self):
        from rosclaw.storage.cli import _migration_status

        result = _migration_status(object(), "memory")
        assert result == {"pending": 0, "versions": []}

    def test_migration_status_mysql_uses_context_manager(self, monkeypatch):
        from rosclaw.storage.cli import _migration_status

        class FakeConnection:
            pass

        class FakeClient:
            @property
            def _connection(self):
                @contextlib.contextmanager
                def _cm():
                    yield FakeConnection()

                return _cm()

        captured = []

        def fake_pending(_self, connection, backend):
            captured.append(connection)
            return ["003"]

        monkeypatch.setattr("rosclaw.storage.cli.MigrationRunner.pending", fake_pending)
        result = _migration_status(FakeClient(), "mysql")
        assert result == {"pending": 1, "versions": ["003"]}
        assert len(captured) == 1
        assert isinstance(captured[0], FakeConnection)

    def test_db_doctor_vector_warmup_with_fix(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()
        practice_root = tmp_path / "practice"
        practice_root.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: sqlite\n  seekdb_path: {tmp_path / 'knowledge.sqlite'}\n"
            f"practice:\n  output_dir: {practice_root}\n"
            f"storage:\n  vector_enabled: true\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "doctor", "--fix", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert not data["issues"]
        vector_check = next((c for c in data["checks"] if c["name"] == "vector warmup"), None)
        assert vector_check is not None
        assert vector_check["ok"] is True

    def test_db_status_reports_vector_enabled(self, tmp_path: Path, capsys, monkeypatch):
        from rosclaw.cli import main

        home = tmp_path / "rosclaw_home"
        home.mkdir()

        def _resolve_home():
            return home

        monkeypatch.setattr("rosclaw.storage.cli.resolve_home", _resolve_home)

        config_path = home / "config"
        config_path.mkdir(parents=True)
        config_path.joinpath("rosclaw.yaml").write_text(
            f"schema_version: '1.0'\n"
            f"workspace:\n  home: {home}\n"
            f"runtime:\n  seekdb_backend: sqlite\n  seekdb_path: {tmp_path / 'knowledge.sqlite'}\n"
            f"storage:\n  vector_enabled: true\n",
            encoding="utf-8",
        )

        sys.argv = ["rosclaw", "db", "status", "--json"]
        assert main() == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["vector"]["enabled"] is True
