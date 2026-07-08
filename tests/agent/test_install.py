"""Tests for `rosclaw agent install`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from rosclaw.agent.install import cmd_agent_install


def _make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "target": "universal",
        "project_root": str(tmp_path),
        "profile": None,
        "robot": None,
        "transport": None,
        "host": None,
        "port": None,
        "check": False,
        "dry_run": False,
        "skip_secrets": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _bootstrap_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_install_generates_cross_agent_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)

    assert cmd_agent_install(_make_args(tmp_path)) == 0

    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "ROSCLAW.md").exists()
    assert (tmp_path / ".agents/skills/rosclaw/SKILL.md").exists()
    assert (tmp_path / ".rosclaw/agent/context.snapshot.json").exists()

    snapshot = json.loads((tmp_path / ".rosclaw/agent/context.snapshot.json").read_text())
    assert len(snapshot["tools"]["available"]) == 13
    assert snapshot["policies"]["no_real_execution"] is True

    captured = capsys.readouterr()
    assert "ROSClaw universal agent integration installed." in captured.out
    assert "rosclaw agent install --project-root . --skip-secrets" in captured.out


async def test_install_dry_run_does_not_write(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_args(tmp_path, dry_run=True)) == 0
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".agents/skills/rosclaw/SKILL.md").exists()


async def test_install_preserves_existing_unmanaged_agents_md(tmp_path: Path) -> None:
    _bootstrap_project(tmp_path)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing Agent Guide\n\nKeep this project rule.\n", encoding="utf-8")

    assert cmd_agent_install(_make_args(tmp_path)) == 0

    content = agents.read_text(encoding="utf-8")
    assert "Keep this project rule." in content
    assert "<!-- ROSCLAW-MANAGED-BEGIN -->" in content
    assert "rosclaw agent install --project-root . --skip-secrets" in content
