"""Readiness checks owned by external Agent harnesses."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodexProjectTrust:
    """Whether Codex will load this repository's project configuration."""

    status: str
    config_path: Path
    project_root: Path

    @property
    def trusted(self) -> bool:
        return self.status == "trusted"

    @property
    def detail(self) -> str:
        if self.status == "trusted":
            return f"trusted in {self.config_path}"
        if self.status == "untrusted":
            return f"explicitly untrusted in {self.config_path}"
        if self.status == "not_listed":
            return f"exact repository root is not trusted in {self.config_path}"
        if self.status == "config_missing":
            return f"Codex user config not found at {self.config_path}"
        return f"Codex user config is invalid: {self.config_path}"


def inspect_codex_project_trust(
    project_root: Path,
    *,
    codex_home: Path | None = None,
) -> CodexProjectTrust:
    """Inspect the user-owned Codex trust entry without changing it."""
    root = project_root.expanduser().resolve()
    home = codex_home
    if home is None:
        configured = os.environ.get("CODEX_HOME")
        home = Path(configured).expanduser() if configured else Path.home() / ".codex"
    config_path = home / "config.toml"
    if not config_path.is_file():
        return CodexProjectTrust("config_missing", config_path, root)

    try:
        with config_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return CodexProjectTrust("config_invalid", config_path, root)

    projects = config.get("projects")
    entry = projects.get(str(root)) if isinstance(projects, dict) else None
    if not isinstance(entry, dict):
        return CodexProjectTrust("not_listed", config_path, root)
    trust_level = entry.get("trust_level")
    status = "trusted" if trust_level == "trusted" else "untrusted"
    return CodexProjectTrust(status, config_path, root)


__all__ = ["CodexProjectTrust", "inspect_codex_project_trust"]
