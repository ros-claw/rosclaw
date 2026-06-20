"""Tests for body history and export CLI commands."""

from __future__ import annotations

import json
import sys
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.cli import main as rosclaw_main


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _run(*argv: str) -> int:
    with patch.object(sys, "argv", ["rosclaw", *argv]):
        return rosclaw_main()


def _create_body(body_id: str) -> None:
    assert _run("body", "create", "--robot", "unitree-g1", "--name", body_id) == 0


def _snapshot_patch() -> tuple[str, ...]:
    """Return CLI args that create a meaningful update-state snapshot."""
    return ("--set", "runtime_state.health=good")


def test_history_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """history shows a helpful message when no snapshots exist."""
    _create_body("g1")
    ws = Path.home() / ".rosclaw"
    snap_dir = ws / "bodies" / "g1" / "snapshots"
    # Remove the initial snapshot created by body create to exercise the empty path.
    for path in list(snap_dir.iterdir()):
        path.unlink()

    assert _run("body", "history") == 0
    captured = capsys.readouterr()
    assert "No snapshots yet" in captured.out


def test_history_lists_snapshots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """history prints snapshots after update-state creates them."""
    _create_body("g1")
    assert _run("body", "update-state", *_snapshot_patch()) == 0
    assert _run("body", "update-state", *_snapshot_patch()) == 0

    assert _run("body", "history") == 0
    captured = capsys.readouterr()
    assert "body-" in captured.out
    assert "Current hash:" in captured.out


def test_history_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """history --json returns machine-readable snapshot metadata."""
    _create_body("g1")
    assert _run("body", "update-state", *_snapshot_patch()) == 0

    capsys.readouterr()  # flush earlier output
    assert _run("body", "history", "--json") == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 2  # initial create snapshot + update-state snapshot
    assert "timestamp" in data[0]
    assert "hash" in data[0]
    assert "snapshot" in data[0]
    assert "size" in data[0]
    assert data[0]["snapshot"].startswith("body-")


def test_history_per_body_isolation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Each body has its own snapshot history."""
    _create_body("g1-sim")
    _create_body("g1-real")
    assert _run("body", "update-state", "--body", "g1-sim", *_snapshot_patch()) == 0

    capsys.readouterr()  # flush earlier output
    assert _run("body", "history", "--body", "g1-sim", "--json") == 0
    sim_data = json.loads(capsys.readouterr().out)
    assert len(sim_data) == 2  # create + update-state

    capsys.readouterr()  # flush
    assert _run("body", "history", "--body", "g1-real", "--json") == 0
    real_data = json.loads(capsys.readouterr().out)
    assert len(real_data) == 1  # create snapshot only


def test_export_zip_default_format(tmp_path: Path) -> None:
    """export writes a zip archive containing the body directory."""
    _create_body("g1")
    dest = tmp_path / "export"
    dest.mkdir()
    assert _run("body", "export", str(dest)) == 0

    archives = list(dest.glob("*.zip"))
    assert len(archives) == 1
    archive = archives[0]
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
    assert any(name.startswith("g1/") and name.endswith("body.yaml") for name in names)


def test_export_tar_explicit_format(tmp_path: Path) -> None:
    """export --format tar writes a tar archive."""
    _create_body("g1")
    archive_path = tmp_path / "g1-export.tar"
    assert _run("body", "export", "--format", "tar", str(archive_path)) == 0

    assert archive_path.exists()
    with tarfile.open(archive_path, "r") as tf:
        members = tf.getmembers()
    assert any(m.name.startswith("g1/") and m.name.endswith("body.yaml") for m in members)


def test_export_per_body_isolation(tmp_path: Path) -> None:
    """export archives only the requested body."""
    _create_body("g1-sim")
    _create_body("g1-real")

    sim_archive = tmp_path / "sim.zip"
    real_archive = tmp_path / "real.zip"
    assert _run("body", "export", "--body", "g1-sim", str(sim_archive)) == 0
    assert _run("body", "export", "--body", "g1-real", str(real_archive)) == 0

    with zipfile.ZipFile(sim_archive, "r") as zf:
        assert any("g1-sim/" in name for name in zf.namelist())
        assert not any("g1-real/" in name for name in zf.namelist())

    with zipfile.ZipFile(real_archive, "r") as zf:
        assert any("g1-real/" in name for name in zf.namelist())
        assert not any("g1-sim/" in name for name in zf.namelist())


def test_export_no_body_linked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """export fails gracefully when no body is linked."""
    ws = tmp_path / "empty"
    ws.mkdir()
    archive = tmp_path / "out.zip"
    rc = _run("body", "export", "--workspace", str(ws), str(archive))
    assert rc == 1
    captured = capsys.readouterr()
    assert "No body linked" in captured.out
