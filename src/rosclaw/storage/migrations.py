"""Schema migration runner for SQLite and MySQL-compatible backends.

Migrations live as plain ``.sql`` files in ``src/rosclaw/storage/migrations``.
Each file may start with a backend marker comment such as ``-- backend: sqlite``
or ``-- backend: all`` (default).  The runner records applied versions in a
``schema_migrations`` table and skips already-applied migrations.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.storage.migrations")


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    Handles single-line ``--`` comments, ``/* */`` block comments, and string
    literals delimited by ``'`` or ``\"``.  Semicolons inside strings or
    comments are ignored so that migrations containing punctuation in data do
    not get mangled.
    """
    statements: list[str] = []
    current: list[str] = []
    i = 0
    n = len(sql)
    in_line_comment = False
    in_block_comment = False
    in_string = False
    string_char = ""

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            if ch == "\\" and nxt:
                current.extend([ch, nxt])
                i += 2
                continue
            if ch == string_char:
                if nxt == string_char:
                    current.extend([ch, nxt])
                    i += 2
                    continue
                in_string = False
                string_char = ""
            current.append(ch)
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch in ("'", '"'):
            in_string = True
            string_char = ch
            current.append(ch)
            i += 1
            continue

        if ch == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)
    return statements


def _read_backend(path: Path) -> str:
    """Return the backend target from the first comment line, defaulting to all."""
    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text else ""
    match = re.match(r"^--\s*backend:\s*(\w+)", first_line)
    return match.group(1).lower() if match else "all"


class MigrationRunner:
    """Apply versioned SQL migrations and track them in ``schema_migrations``."""

    def __init__(self, migrations_dir: Path | str | None = None) -> None:
        self._migrations_dir = (
            Path(migrations_dir) if migrations_dir else Path(__file__).parent / "migrations"
        )

    def apply(self, connection: Any, backend: str) -> list[str]:
        """Apply all pending migrations for *backend*.

        :param connection: ``sqlite3.Connection`` or a PyMySQL connection.
        :param backend: ``sqlite`` or ``mysql``.
        :return: list of applied migration version strings.
        """
        self._ensure_table(connection, backend)
        applied = self._applied_versions(connection, backend)
        applied_by_version = {row["version"]: row for row in applied}

        new: list[str] = []
        migration_paths = sorted(
            self._migrations_dir.glob("*.sql"),
            key=lambda p: int(self._version_from_filename(p)),
        )
        for path in migration_paths:
            version = self._version_from_filename(path)
            target = _read_backend(path)
            if target != "all" and target != backend:
                continue
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = applied_by_version.get(version)
            if previous is not None:
                recorded_checksum = previous.get("checksum")
                if recorded_checksum and recorded_checksum != checksum:
                    raise RuntimeError(
                        f"Migration {version} checksum mismatch for {path.name}; "
                        "applied migrations must not be modified"
                    )
                if not recorded_checksum:
                    self._update_checksum(connection, version, checksum, backend)
                    connection.commit()
                continue
            logger.info("Applying migration %s for backend %s", path.name, backend)
            sql = path.read_text(encoding="utf-8")
            if backend == "sqlite":
                connection.execute("SAVEPOINT rosclaw_migration")
                try:
                    self._execute_script(connection, sql, backend)
                    self._record(connection, version, checksum, backend)
                    connection.execute("RELEASE SAVEPOINT rosclaw_migration")
                except Exception:
                    connection.execute("ROLLBACK TO SAVEPOINT rosclaw_migration")
                    connection.execute("RELEASE SAVEPOINT rosclaw_migration")
                    raise
            else:
                try:
                    self._execute_script(connection, sql, backend)
                    self._record(connection, version, checksum, backend)
                except Exception:
                    connection.rollback()
                    raise
            connection.commit()
            new.append(version)

        return new

    @staticmethod
    def _version_from_filename(path: Path) -> str:
        """Migration versions are the leading numeric token of the filename."""
        match = re.match(r"(\d+)", path.name)
        if not match:
            raise ValueError(f"Migration filename must start with a version number: {path.name}")
        return match.group(1)

    def pending(self, connection: Any, backend: str) -> list[str]:
        """Return versions of migrations that have not yet been applied.

        This is a read-only helper for ``db status`` / ``db doctor``; it does
        not modify the database.
        """
        self._ensure_table(connection, backend)
        applied_versions = {row["version"] for row in self._applied_versions(connection, backend)}
        pending_versions: list[str] = []
        migration_paths = sorted(
            self._migrations_dir.glob("*.sql"),
            key=lambda p: int(self._version_from_filename(p)),
        )
        for path in migration_paths:
            version = self._version_from_filename(path)
            if version in applied_versions:
                continue
            target = _read_backend(path)
            if target != "all" and target != backend:
                continue
            pending_versions.append(version)
        return pending_versions

    @staticmethod
    def _ensure_table(connection: Any, backend: str) -> None:
        if backend == "sqlite":
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL,
                    checksum TEXT
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(schema_migrations)")}
            if "checksum" not in columns:
                connection.execute("ALTER TABLE schema_migrations ADD COLUMN checksum TEXT")
        else:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(64) PRIMARY KEY,
                        applied_at DOUBLE NOT NULL,
                        checksum VARCHAR(64) NULL
                    ) DEFAULT CHARACTER SET utf8mb4
                    """
                )
                cursor.execute("SHOW COLUMNS FROM schema_migrations")
                columns = {
                    row["Field"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()
                }
                if "checksum" not in columns:
                    cursor.execute(
                        "ALTER TABLE schema_migrations ADD COLUMN checksum VARCHAR(64) NULL"
                    )

    @staticmethod
    def _applied_versions(connection: Any, backend: str) -> list[dict[str, Any]]:
        if backend == "sqlite":
            cursor = connection.execute(
                "SELECT version, applied_at, checksum FROM schema_migrations"
            )
            return [
                {"version": row[0], "applied_at": row[1], "checksum": row[2]}
                for row in cursor.fetchall()
            ]
        with connection.cursor() as cursor:
            cursor.execute("SELECT version, applied_at, checksum FROM schema_migrations")
            rows = cursor.fetchall()
            return MigrationRunner._rows_as_dicts(cursor, rows)

    @staticmethod
    def _rows_as_dicts(cursor: Any, rows: list[Any]) -> list[dict[str, Any]]:
        """Convert cursor rows to dicts regardless of cursor type."""
        if not rows:
            return []
        # If rows are already dict-like, preserve them.
        first = rows[0]
        if isinstance(first, dict):
            return [dict(r) for r in rows]
        # PyMySQL tuple rows: map from cursor.description.
        columns = [desc[0] for desc in getattr(cursor, "description", []) or []]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    @staticmethod
    def _record(connection: Any, version: str, checksum: str, backend: str) -> None:
        if backend == "sqlite":
            connection.execute(
                "INSERT OR REPLACE INTO schema_migrations "
                "(version, applied_at, checksum) VALUES (?, ?, ?)",
                (version, time.time(), checksum),
            )
        else:
            with connection.cursor() as cursor:
                cursor.execute(
                    "REPLACE INTO schema_migrations "
                    "(version, applied_at, checksum) VALUES (%s, %s, %s)",
                    (version, time.time(), checksum),
                )

    @staticmethod
    def _update_checksum(connection: Any, version: str, checksum: str, backend: str) -> None:
        if backend == "sqlite":
            connection.execute(
                "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
                (checksum, version),
            )
        else:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE schema_migrations SET checksum = %s WHERE version = %s",
                    (checksum, version),
                )

    @staticmethod
    def _execute_script(connection: Any, sql: str, backend: str) -> None:
        if backend == "sqlite":
            for statement in _split_statements(sql):
                connection.execute(statement)
            return

        # MySQL: prefer multi=True so the server parses statements itself.
        # This correctly handles triggers, stored procedures, and backtick
        # identifiers that a naive semicolon split would mangle.
        with connection.cursor() as cursor:
            try:
                multi_supported = hasattr(cursor, "execute") and "multi" in (
                    cursor.execute.__code__.co_varnames
                    if hasattr(cursor.execute, "__code__")
                    else []
                )
            except Exception:  # noqa: BLE001
                multi_supported = False

            if multi_supported:
                for result in cursor.execute(sql, multi=True):
                    # Consume any result sets so the protocol stays clean.
                    if getattr(result, "with_rows", False):
                        result.fetchall()
            else:
                for statement in _split_statements(sql):
                    cursor.execute(statement)
