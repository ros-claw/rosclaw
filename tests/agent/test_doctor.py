"""Tests for `rosclaw agent doctor`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rosclaw.agent.doctor import cmd_agent_doctor_claude_code
from rosclaw.agent.init_claude_code import cmd_agent_init_claude_code
from rosclaw.agent.install import cmd_agent_install


def _make_init_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(tmp_path),
        profile=None,
        robot=None,
        transport=None,
        host=None,
        port=None,
        check=False,
        dry_run=False,
        skip_secrets=True,
    )


def _make_doctor_args(
    tmp_path: Path,
    *,
    agent: str = "claude-code",
    skip_secrets: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        agent=agent,
        project_root=str(tmp_path),
        skip_secrets=skip_secrets,
    )


def _make_install_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        target="universal",
        project_root=str(tmp_path),
        profile=None,
        robot=None,
        transport=None,
        host=None,
        port=None,
        check=False,
        dry_run=False,
        skip_secrets=True,
    )


def _bootstrap_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_doctor_passes_after_init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_init_claude_code(_make_init_args(tmp_path)) == 0
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert "Onboarding files: OK" in captured.out


async def test_doctor_accepts_cross_agent_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0

    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="codex")) == 0
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="openclaw")) == 0

    captured = capsys.readouterr()
    assert "Agent target: codex" in captured.out
    assert "Agent target: openclaw" in captured.out


async def test_openclaw_doctor_requires_workspace_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    capsys.readouterr()
    (tmp_path / ".agents/skills/rosclaw/SKILL.md").unlink()

    assert cmd_agent_doctor_claude_code(
        _make_doctor_args(tmp_path, agent="openclaw")
    ) == 1

    captured = capsys.readouterr()
    assert "Onboarding files: FAILED" in captured.out
    assert ".agents/skills/rosclaw/SKILL.md" in captured.out


async def test_doctor_fails_without_init(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "Onboarding files: FAILED" in captured.out
