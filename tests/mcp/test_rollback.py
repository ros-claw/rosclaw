"""Tests for staged installation rollback support."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.rollback import RollbackContext


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    return tmp_path / "staging"


def test_rollback_restores_modified_file(staging: Path) -> None:
    original = staging.parent / "original.txt"
    original.write_text("before", encoding="utf-8")

    ctx = RollbackContext(staging)
    ctx.backup(original)
    original.write_text("after", encoding="utf-8")
    ctx.rollback()

    assert original.read_text(encoding="utf-8") == "before"


def test_rollback_removes_created_file(staging: Path) -> None:
    created = staging.parent / "created.txt"

    ctx = RollbackContext(staging)
    ctx.backup(created)
    created.write_text("new", encoding="utf-8")
    ctx.rollback()

    assert not created.exists()


def test_commit_discards_backups(staging: Path) -> None:
    original = staging.parent / "original.txt"
    original.write_text("keep", encoding="utf-8")

    ctx = RollbackContext(staging)
    ctx.backup(original)
    ctx.commit()

    assert original.read_text(encoding="utf-8") == "keep"
    assert not staging.exists()


def test_record_tracks_actions(staging: Path) -> None:
    original = staging.parent / "original.txt"
    created = staging.parent / "created.txt"
    original.write_text("keep", encoding="utf-8")

    ctx = RollbackContext(staging)
    ctx.backup(original)
    ctx.backup(created)
    record = ctx.record()

    assert any(r["action"] == "backed_up" for r in record)
    assert any(r["action"] == "created" for r in record)


def test_rollback_directory_backup(staging: Path) -> None:
    directory = staging.parent / "dir"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "file.txt").write_text("inside", encoding="utf-8")

    ctx = RollbackContext(staging)
    ctx.backup(directory)
    (directory / "file.txt").write_text("changed", encoding="utf-8")
    ctx.rollback()

    assert (directory / "file.txt").read_text(encoding="utf-8") == "inside"
