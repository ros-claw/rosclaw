"""Read-only source and protected-input closure checks for prepared episodes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping
from pathlib import Path


class EpisodeSourceBindingsError(ValueError):
    """A declared source bundle does not match its independent preparation."""


def validate_episode_source_bindings(
    workspace: Path,
    new_sources: Mapping[str, str],
    required_sources: Collection[str],
    protected_inputs: Mapping[str, str],
    expected_protected_inputs: Mapping[str, str],
) -> dict[str, str]:
    """Verify required source files and the complete independent input closure.

    Supply required paths and expected input hashes from independently pinned
    preparation, never from the candidate's own manifest. Additional declared
    source files are checked too. This creates no files and confers no runtime,
    model, middleware, or physical authority.
    """
    if isinstance(required_sources, str) or not required_sources:
        raise EpisodeSourceBindingsError("INVALID_REQUIRED_SOURCE_PATHS")
    if not all(isinstance(p, str) and p for p in required_sources):
        raise EpisodeSourceBindingsError("INVALID_REQUIRED_SOURCE_PATHS")
    if not isinstance(new_sources, Mapping) or not set(required_sources).issubset(new_sources):
        raise EpisodeSourceBindingsError("SOURCE_PRIMARY_BINDINGS_MISSING")
    if (
        not isinstance(protected_inputs, Mapping)
        or not isinstance(expected_protected_inputs, Mapping)
        or not expected_protected_inputs
        or dict(protected_inputs) != dict(expected_protected_inputs)
    ):
        raise EpisodeSourceBindingsError("SOURCE_PROTECTED_CLOSURE_MISMATCH")
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise EpisodeSourceBindingsError("SOURCE_WORKSPACE_NOT_DIRECTORY")
    checked: dict[str, str] = {}
    for bindings in (new_sources, protected_inputs):
        for name, digest in bindings.items():
            if not isinstance(name, str) or not name:
                raise EpisodeSourceBindingsError("SOURCE_PATH_INVALID")
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
                raise EpisodeSourceBindingsError("SOURCE_PATH_NOT_CANONICAL")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise EpisodeSourceBindingsError("SOURCE_SHA256_INVALID")
            try:
                path = (root / relative).resolve(strict=True)
            except FileNotFoundError as exc:
                raise EpisodeSourceBindingsError(f"SOURCE_FILE_MISSING:{name}") from exc
            if not path.is_relative_to(root) or not path.is_file():
                raise EpisodeSourceBindingsError(f"SOURCE_FILE_OUTSIDE_OR_INVALID:{name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise EpisodeSourceBindingsError(f"SOURCE_FILE_HASH_MISMATCH:{name}")
            checked[name] = digest
    return checked
