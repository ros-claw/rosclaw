from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.installed import InstalledRegistry
from rosclaw.mcp.onboarding.source_installer import install_from_git


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "ROSClaw Test")
    _git(repo, "config", "user.email", "test@rosclaw.local")
    (repo / "mcp_server.py").write_text("print('v1')\n", encoding="utf-8")
    _git(repo, "add", "mcp_server.py")
    _git(repo, "commit", "-m", "v1")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "mcp_server.py").write_text("print('v2')\n", encoding="utf-8")
    _git(repo, "add", "mcp_server.py")
    _git(repo, "commit", "-m", "v2")
    return repo, first, _git(repo, "rev-parse", "HEAD")


def test_source_install_checks_out_exact_locked_commit(tmp_path: Path) -> None:
    repo, first, latest = _source_repository(tmp_path)
    assert first != latest

    result = install_from_git(
        str(repo),
        server_name="locked-mcp",
        home=tmp_path / "home",
        no_install_deps=True,
        revision=first,
    )

    assert result.success
    assert result.commit == first
    assert (result.local_path / "mcp_server.py").read_text(encoding="utf-8") == "print('v1')\n"


def test_source_install_rejects_option_like_revision_without_git_call(tmp_path: Path) -> None:
    result = install_from_git(
        "https://example.invalid/repo.git",
        home=tmp_path / "home",
        no_install_deps=True,
        revision="--upload-pack=evil",
    )

    assert not result.success
    assert result.commit is None
    assert result.errors == ["invalid git revision: '--upload-pack=evil'"]


def test_failed_revision_upgrade_preserves_previous_source(tmp_path: Path) -> None:
    repo, first, _latest = _source_repository(tmp_path)
    home = tmp_path / "home"
    installed = install_from_git(
        str(repo),
        server_name="locked-mcp",
        home=home,
        no_install_deps=True,
        revision=first,
    )
    assert installed.success
    runtime_config = installed.runtime_config_path.read_bytes()

    failed = install_from_git(
        str(repo),
        server_name="locked-mcp",
        home=home,
        no_install_deps=True,
        revision="0" * 40,
    )

    assert not failed.success
    assert (installed.local_path / "mcp_server.py").read_text(encoding="utf-8") == "print('v1')\n"
    assert _git(installed.local_path, "rev-parse", "HEAD") == first
    assert installed.runtime_config_path.read_bytes() == runtime_config
    assert not list(installed.local_path.parent.glob(".source-*"))


def test_failed_registry_update_rolls_back_source_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, first, latest = _source_repository(tmp_path)
    home = tmp_path / "home"
    installed = install_from_git(
        str(repo),
        server_name="locked-mcp",
        home=home,
        no_install_deps=True,
        revision=first,
    )
    assert installed.success
    runtime_config = installed.runtime_config_path.read_bytes()
    registry_path = InstalledRegistry(home=home).path
    registry = registry_path.read_bytes()

    def fail_registry_update(_self, _record) -> None:
        raise OSError("simulated registry write failure")

    monkeypatch.setattr(InstalledRegistry, "add", fail_registry_update)
    failed = install_from_git(
        str(repo),
        server_name="locked-mcp",
        home=home,
        no_install_deps=True,
        revision=latest,
    )

    assert not failed.success
    assert "registry write failure" in failed.errors[-1]
    assert (installed.local_path / "mcp_server.py").read_text(encoding="utf-8") == "print('v1')\n"
    assert _git(installed.local_path, "rev-parse", "HEAD") == first
    assert installed.runtime_config_path.read_bytes() == runtime_config
    assert registry_path.read_bytes() == registry
    assert not list(installed.local_path.parent.glob(".source-*"))
