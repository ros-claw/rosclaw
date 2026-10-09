"""Read-only offline source checks with per-scan hardlink deduplication.

No cache survives a scan. Every declared path is opened and checked, including
all parent components, before and after reading. This is neither an atomic
filesystem snapshot nor protection against a privileged hostile OS; a file can
change after the call returns. A scan never grants execution or promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class FilePinScan:
    """Counts and identity of the owned declaration, not an execution receipt."""

    declaration_hash: str
    verified_paths: int
    unique_content_reads: int
    bytes_read: int
    logical_bytes: int


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_file(name: str) -> BufferedReader:
    """Open one absolute regular file without following any symlink component."""
    parts = Path(name).parts
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        for component in parts[1:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
    finally:
        os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("ordinary pinned file required")
        return cast(BufferedReader, os.fdopen(fd, "rb"))
    except BaseException:
        os.close(fd)
        raise


def _digest_exact_size(stream: BufferedReader, size: int) -> str:
    """Bound the read even if another process keeps appending to the file."""
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(256 * 1024, remaining))
        if not chunk:
            raise ValueError("pinned file length changed during verification")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise ValueError("pinned file length changed during verification")
    return "sha256:" + digest.hexdigest()


def verify_file_pins(pins: dict[str, str], *, max_bytes: int = 16 * 1024**3) -> FilePinScan:
    """Hash each stable inode once in this call, still verifying every alias.

    Retain a private declaration so later caller mutation cannot change the
    scope of this scan. The returned hash binds that exact owned declaration.
    File identity and metadata must stay unchanged through the final sweep.
    Missing files, links, special files, conflicting hashes, concurrent changes,
    and capacity failures are errors, never passing partial results.
    """
    if type(pins) is not dict or not 1 <= len(pins) <= 65536:
        raise ValueError("bounded nonempty ordinary pin mapping required")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("positive integer per-scan byte capacity required")
    owned = pins.copy()
    for name, expected in owned.items():
        if (
            type(name) is not str
            or not Path(name).is_absolute()
            or name.startswith("//")
            or str(Path(name)) != name
            or ".." in Path(name).parts
            or len(Path(name).parts) < 2
            or type(expected) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("canonical absolute paths and SHA256 pins required")
    declaration_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(owned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    cache: dict[tuple[int, ...], str] = {}
    snapshots: dict[str, tuple[int, ...]] = {}
    bytes_read = 0
    logical_bytes = 0
    for name, expected in owned.items():
        with _open_file(name) as stream:
            before = os.fstat(stream.fileno())
            key = _fingerprint(before)
            if key not in cache:
                if before.st_size > max_bytes - bytes_read:
                    raise ValueError("pinned file scan byte capacity exceeded")
                cache[key] = _digest_exact_size(stream, before.st_size)
                bytes_read += before.st_size
            if cache[key] != expected:
                raise ValueError("pinned content mismatch: " + name)
            if _fingerprint(os.fstat(stream.fileno())) != key:
                raise ValueError("opened file changed during verification: " + name)
            snapshots[name] = key
            logical_bytes += before.st_size
    for name, key in snapshots.items():
        with _open_file(name) as stream:
            if _fingerprint(os.fstat(stream.fileno())) != key:
                raise ValueError("path changed during verification: " + name)
    return FilePinScan(declaration_hash, len(snapshots), len(cache), bytes_read, logical_bytes)
