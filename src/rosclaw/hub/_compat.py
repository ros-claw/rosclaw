"""Compatibility helpers for the ROSClaw Hub."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

PathLike = str | Path


def extractall_tar(tar: tarfile.TarFile, path: PathLike) -> None:
    """Extract a tar archive safely.

    On Python 3.12+ ``filter='data'`` is used to suppress the deprecation
    warning and reject unsafe archive members. Older runtimes use the legacy
    default because the ``filter`` parameter is not supported.
    """
    if sys.version_info >= (3, 12):
        tar.extractall(path=path, filter="data")
    else:
        tar.extractall(path=path)
