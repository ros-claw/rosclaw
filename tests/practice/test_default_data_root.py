"""Regression tests for writable Practice data-root defaults."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rosclaw.practice.config import PracticeConfig
from rosclaw.practice.recorder import PracticeRecorder


def test_default_data_root_follows_rosclaw_home(tmp_path: Path, monkeypatch) -> None:
    """Library callers must not inherit the deployment-specific /data path."""
    home = tmp_path / "rosclaw-home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    monkeypatch.delenv("ROSCLAW_PRACTICE_DATA_ROOT", raising=False)

    expected = home / "data" / "practice"
    assert PracticeConfig().data_root_path == expected
    assert PracticeRecorder("test_bot").layout.data_root == expected


def test_practice_cli_honors_data_root_environment(tmp_path: Path) -> None:
    """A clean CLI process must route all default Practice writes to the override."""
    project_root = Path(__file__).resolve().parents[2]
    data_root = tmp_path / "practice-data"
    env = os.environ.copy()
    env["ROSCLAW_HOME"] = str(tmp_path / "rosclaw-home")
    env["ROSCLAW_PRACTICE_DATA_ROOT"] = str(data_root)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(project_root / "src"), env.get("PYTHONPATH")) if part
    )

    result = subprocess.run(
        [sys.executable, "-m", "rosclaw.cli", "practice", "list"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (data_root / "indexes" / "practice_catalog.sqlite").is_file()
