"""Tests for `rosclaw agent doctor claude-code`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rosclaw.agent.doctor import cmd_agent_doctor_claude_code
from rosclaw.agent.init_claude_code import cmd_agent_init_claude_code


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


def _make_doctor_args(tmp_path: Path, *, skip_secrets: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(tmp_path),
        skip_secrets=skip_secrets,
    )


def _bootstrap_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rosclaw"\n', encoding="utf-8")


async def test_doctor_passes_after_init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_init_claude_code(_make_init_args(tmp_path)) == 0
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert "Onboarding files: OK" in captured.out


async def test_doctor_fails_without_init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "Onboarding files: FAILED" in captured.out
