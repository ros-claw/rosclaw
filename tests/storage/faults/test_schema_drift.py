"""Fault injection: migration schema drift — runtime refuses the database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner


def _make_migrations(dir_path: Path, sql: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "001_create_widgets.sql").write_text(sql, encoding="utf-8")
    return dir_path


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    mig_dir = _make_migrations(
        tmp_path / "migrations",
        "CREATE TABLE widgets (id TEXT PRIMARY KEY, value INTEGER);\n",
    )
    db = tmp_path / "app.sqlite"
    conn = sqlite3.connect(str(db))
    runner = MigrationRunner(migrations_dir=mig_dir)
    applied = runner.apply(conn, "sqlite")
    assert applied == ["001"]

    # An operator/hot-patcher edits the applied migration: drift.
    (mig_dir / "001_create_widgets.sql").write_text(
        "CREATE TABLE widgets (id TEXT PRIMARY KEY, value INTEGER, extra TEXT);\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner.apply(conn, "sqlite")
    conn.close()


def test_pending_reports_nothing_after_clean_apply(tmp_path: Path) -> None:
    mig_dir = _make_migrations(
        tmp_path / "migrations",
        "CREATE TABLE widgets (id TEXT PRIMARY KEY);\n",
    )
    conn = sqlite3.connect(str(tmp_path / "app.sqlite"))
    runner = MigrationRunner(migrations_dir=mig_dir)
    runner.apply(conn, "sqlite")
    assert runner.pending(conn, "sqlite") == []
    conn.close()
