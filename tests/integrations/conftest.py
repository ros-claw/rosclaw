"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_rosclaw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use a per-test ROSClAW_HOME without leaking it to unrelated tests."""
    home = tmp_path / "rosclaw_home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    return home


@pytest.fixture
def fake_lerobot_info(tmp_path: Path, monkeypatch):
    """Create a fake `lerobot-info` executable on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "lerobot-info"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('LeRobot info stub')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return script
