"""Read-only output layout admission for prepared native episodes.

Run this before launching the worker. A frozen file manifest does not preserve
empty directories, so successful hash admission alone cannot admit the first
clock or birth receipt write. This is preparation validation, not a sandbox or
an assurance against later filesystem changes or exhausted storage.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


class EpisodeLayoutError(ValueError):
    """Prepared output paths cannot be used by this episode."""


def validate_episode_output_layout(
    case_root: Path, output_paths: Mapping[str, Path]
) -> dict[str, str]:
    """Require distinct, absent outputs with existing writable private parents.

    No directory, receipt, clock, credential, or session is created. A symlink
    alias for the declared case root is supported; output parents must resolve
    within that root. Pin a marker file when transferring an otherwise empty
    output directory, and separately exercise the real writer in a disposable
    fixture during preparation.
    """
    root = Path(case_root).resolve(strict=True)
    if not root.is_dir():
        raise EpisodeLayoutError("CASE_ROOT_NOT_DIRECTORY")
    if not output_paths:
        raise EpisodeLayoutError("NO_DECLARED_OUTPUTS")
    checked: dict[str, str] = {}
    seen: set[Path] = set()
    for name, value in output_paths.items():
        if not isinstance(name, str) or not name.strip():
            raise EpisodeLayoutError("INVALID_OUTPUT_NAME")
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists() or path.is_symlink():
            raise EpisodeLayoutError(f"OUTPUT_ALREADY_EXISTS:{name}")
        try:
            parent = path.parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise EpisodeLayoutError(f"MISSING_OUTPUT_PARENT:{name}") from exc
        if not parent.is_dir():
            raise EpisodeLayoutError(f"OUTPUT_PARENT_NOT_DIRECTORY:{name}")
        if not parent.is_relative_to(root):
            raise EpisodeLayoutError(f"OUTPUT_PARENT_OUTSIDE_CASE:{name}")
        if not os.access(parent, os.W_OK | os.X_OK):
            raise EpisodeLayoutError(f"OUTPUT_PARENT_NOT_WRITABLE:{name}")
        resolved = parent / path.name
        if resolved in seen:
            raise EpisodeLayoutError(f"DUPLICATE_OUTPUT_PATH:{name}")
        seen.add(resolved)
        checked[name] = str(resolved)
    return checked
