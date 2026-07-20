"""Tests for `rosclaw agent doctor`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import rosclaw.agent.doctor as doctor_module
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        f'[projects."{tmp_path.resolve()}"]\ntrust_level = "trusted"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="codex")) == 0
    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="openclaw")) == 0

    captured = capsys.readouterr()
    assert "Agent target: codex" in captured.out
    assert "Agent target: openclaw" in captured.out
    assert "Codex project trust: yes" in captured.out


async def test_codex_doctor_fails_when_repository_is_not_trusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    capsys.readouterr()

    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="codex")) == 1

    captured = capsys.readouterr()
    assert "Codex project trust: no" in captured.out
    assert "project MCP config is ignored" in captured.out


async def test_openclaw_doctor_requires_workspace_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_install(_make_install_args(tmp_path)) == 0
    capsys.readouterr()
    (tmp_path / ".agents/skills/rosclaw/SKILL.md").unlink()

    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path, agent="openclaw")) == 1

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


async def test_doctor_fails_when_configured_server_is_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bootstrap_project(tmp_path)
    assert cmd_agent_init_claude_code(_make_init_args(tmp_path)) == 0
    monkeypatch.setattr(
        doctor_module,
        "_check_server_reachable",
        lambda _profile: (False, "connection refused"),
    )

    assert cmd_agent_doctor_claude_code(_make_doctor_args(tmp_path)) == 1
