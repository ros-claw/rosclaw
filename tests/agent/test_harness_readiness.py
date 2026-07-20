"""Tests for external Agent harness readiness checks."""

from __future__ import annotations

from pathlib import Path

from rosclaw.agent.harness_readiness import inspect_codex_project_trust


def test_codex_trust_requires_exact_repository_root(tmp_path: Path) -> None:
    project = tmp_path / "parent" / "repo"
    project.mkdir(parents=True)
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        f'[projects."{project.parent}"]\ntrust_level = "trusted"\n',
        encoding="utf-8",
    )

    result = inspect_codex_project_trust(project, codex_home=codex_home)

    assert result.status == "not_listed"
    assert result.trusted is False


def test_codex_trust_accepts_exact_repository_root(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        f'[projects."{project.resolve()}"]\ntrust_level = "trusted"\n',
        encoding="utf-8",
    )

    result = inspect_codex_project_trust(project, codex_home=codex_home)

    assert result.status == "trusted"
    assert result.trusted is True


def test_codex_trust_reports_invalid_config(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("[projects\n", encoding="utf-8")

    result = inspect_codex_project_trust(project, codex_home=codex_home)

    assert result.status == "config_invalid"
    assert result.trusted is False
