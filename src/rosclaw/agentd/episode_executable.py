"""Read-only admission for a pinned executable used by a prepared episode."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path


class EpisodeExecutableError(ValueError):
    """A prepared executable cannot be dispatched by the current operator."""


def validate_episode_executable(path: Path, expected_sha256: str) -> dict[str, str | int]:
    """Check content and execute permission before publishing an episode clock.

    This never launches the asset or changes its mode. Preparation must also
    exercise a harmless real startup to check its loader and dependencies.
    Filesystem changes after admission and runtime failures remain possible.
    Symlinks are rejected for the pinned asset; declare its real file instead.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise EpisodeExecutableError("INVALID_EXECUTABLE_SHA256")
    asset = Path(path)
    try:
        info = asset.lstat()
    except FileNotFoundError as exc:
        raise EpisodeExecutableError("EXECUTABLE_MISSING") from exc
    if not stat.S_ISREG(info.st_mode):
        raise EpisodeExecutableError("EXECUTABLE_NOT_REGULAR_FILE")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise EpisodeExecutableError("EXECUTABLE_HASH_MISMATCH")
    if not info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) or not os.access(
        asset, os.X_OK
    ):
        raise EpisodeExecutableError("EXECUTABLE_PERMISSION_DENIED")
    return {
        "path": str(asset.resolve(strict=True)),
        "sha256": digest,
        "mode": stat.S_IMODE(info.st_mode),
    }
