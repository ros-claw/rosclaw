"""Tests for rosclaw.storage.migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from rosclaw.storage.migrations import MigrationRunner, _split_statements


class _FakeMySQLResult:
    """Result set returned by a fake multi=True execution."""

    def __init__(self, with_rows: bool = False, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.with_rows = with_rows
        self._rows = rows or []

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeMySQLCursor:
    """Cursor mock that supports ``cursor.execute(sql, multi=True)``."""

    def __init__(self) -> None:
        self.executed: list[tuple[Any, ...]] = []
        self.multi_scripts: list[str] = []
        self._tables: set[str] = set()
        self._migrations: list[tuple[str, float]] = []
        self._description: tuple[tuple[str, ...], ...] | None = None
        self._rows: list[tuple[Any, ...]] = []

    def execute(
        self, sql: str, *args: Any, multi: bool = False, **kwargs: Any
    ) -> list[_FakeMySQLResult]:
        if multi:
            self.multi_scripts.append(sql)
            results: list[_FakeMySQLResult] = []
            for statement in _split_statements(sql):
                results.extend(self._run_one(statement))
            return results
        results = list(self._run_one(sql, args[0] if args else ()))
        if results:
            self._rows = results[0].fetchall()
        return []

    def _run_one(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> Generator[_FakeMySQLResult, None, None]:
        self.executed.append((sql, params))
        upper = sql.strip().upper()
        if upper.startswith("CREATE TABLE"):
            # Extract table name naively for the fake, tolerating "IF NOT EXISTS".
            tokens = sql.split()
            idx = tokens.index("TABLE") + 1
            if tokens[idx].upper() == "IF":
                idx = tokens.index("EXISTS", idx) + 1
            name = tokens[idx].strip("`(")
            self._tables.add(name)
        elif upper.startswith("REPLACE INTO SCHEMA_MIGRATIONS"):
            version, applied_at = params
            self._migrations = [m for m in self._migrations if m[0] != version]
            self._migrations.append((version, applied_at))
        elif upper.startswith("SELECT VERSION, APPLIED_AT"):
            self._description = (("version",), ("applied_at",))
            self._rows = list(self._migrations)
            yield _FakeMySQLResult(with_rows=True, rows=self._rows)
            return
        yield _FakeMySQLResult(with_rows=False)

    @property
    def description(self) -> tuple[tuple[str, ...], ...] | None:
        return self._description

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows
        self._rows = []
        return rows

    def __enter__(self) -> _FakeMySQLCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class _FakeMySQLConnection:
    """PyMySQL-style connection mock."""

    def __init__(self) -> None:
        self.cursor_instance = _FakeMySQLCursor()
        self.committed = False

    @contextmanager
    def cursor(self) -> Generator[_FakeMySQLCursor, None, None]:
        yield self.cursor_instance

    def commit(self) -> None:
        self.committed = True


def test_migration_applies_sqlite(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_test_table_sqlite.sql").write_text(
        "-- backend: sqlite\nCREATE TABLE IF NOT EXISTS test_table (id TEXT PRIMARY KEY);",
        encoding="utf-8",
    )
    (migrations_dir / "002_create_test_table_mysql.sql").write_text(
        "-- backend: mysql\nCREATE TABLE IF NOT EXISTS test_table (id VARCHAR(64) PRIMARY KEY);",
        encoding="utf-8",
    )

    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    runner = MigrationRunner(migrations_dir)
    applied = runner.apply(conn, "sqlite")
    assert "001" in applied
    assert "002" not in applied

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "test_table" in tables
    assert "schema_migrations" in tables

    # Idempotent: second apply returns nothing new.
    applied2 = runner.apply(conn, "sqlite")
    assert applied2 == []


def test_migration_tracks_versions(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_first.sql").write_text("-- backend: all\nSELECT 1;", encoding="utf-8")
    (migrations_dir / "002_second.sql").write_text("-- backend: all\nSELECT 1;", encoding="utf-8")

    conn = sqlite3.connect(":memory:")
    runner = MigrationRunner(migrations_dir)
    assert runner.apply(conn, "sqlite") == ["001", "002"]
    assert runner.apply(conn, "sqlite") == []


def test_migration_bad_filename(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "bad_name.sql").write_text("SELECT 1;", encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="version number"):
        MigrationRunner(migrations_dir).apply(conn, "sqlite")


def test_migration_applies_mysql(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_table.sql").write_text(
        "-- backend: all\nCREATE TABLE IF NOT EXISTS test_table (id VARCHAR(64) PRIMARY KEY);",
        encoding="utf-8",
    )
    (migrations_dir / "002_sqlite_only.sql").write_text(
        "-- backend: sqlite\nCREATE TABLE IF NOT EXISTS sqlite_only (id TEXT PRIMARY KEY);",
        encoding="utf-8",
    )

    conn = _FakeMySQLConnection()
    runner = MigrationRunner(migrations_dir)
    applied = runner.apply(conn, "mysql")
    assert "001" in applied
    assert "002" not in applied
    assert conn.committed
    assert "test_table" in conn.cursor_instance._tables
    assert "schema_migrations" in conn.cursor_instance._tables
    assert "sqlite_only" not in conn.cursor_instance._tables
    assert any(m[0] == "001" for m in conn.cursor_instance._migrations)


def test_mysql_complex_sql_passed_whole(tmp_path: Path) -> None:
    """MySQL multi=True should receive the full script so triggers/procedures survive."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    trigger_sql = (
        "-- backend: mysql\n"
        "CREATE TRIGGER log_insert BEFORE INSERT ON test_table\n"
        "FOR EACH ROW BEGIN\n"
        "  INSERT INTO audit_log (msg) VALUES ('new row');\n"
        "END;"
    )
    (migrations_dir / "001_trigger.sql").write_text(trigger_sql, encoding="utf-8")

    conn = _FakeMySQLConnection()
    MigrationRunner(migrations_dir).apply(conn, "mysql")
    # The multi=True path should submit the whole script in one call.
    assert any(trigger_sql in script for script in conn.cursor_instance.multi_scripts)


def test_rows_as_dicts_handles_tuple_rows() -> None:
    class FakeCursor:
        description = (("version",), ("applied_at",))

    cursor = FakeCursor()
    rows = [("001", 1.0), ("002", 2.0)]
    result = MigrationRunner._rows_as_dicts(cursor, rows)
    assert result == [
        {"version": "001", "applied_at": 1.0},
        {"version": "002", "applied_at": 2.0},
    ]
