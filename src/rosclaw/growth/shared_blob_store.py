"""POSIX lossless read-only blob publication, not policy or evidence approval.

Only new target files are published. Caller sources and existing blobs are never
modified or repaired. Content identity is not provenance or a signature. This
module knows nothing about robots, simulators, policy loading or executors.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SharedBlobArtifact:
    blob_hash: str
    size_bytes: int
    blob_path: Path
    target_path: Path


def _local_directory(path: Path) -> None:
    if not path.is_dir() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("existing local non-symlink directory required")


def _regular_file(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("local regular file required; no symlinks")
    return info


def _digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_readonly_blob(
    source: Path,
    *,
    store: Path,
    target: Path,
    expected_hash: str,
    maximum_blob_bytes: int,
) -> SharedBlobArtifact:
    """Publish an owned immutable copy and a new same-volume hardlink.

    The caller must supply an exact content hash and explicit size bound. A
    duplicate logical artifact keeps all its bytes, but shares one inode. An
    existing target always rejects, including an identical file: this primitive
    is not an overwrite, migration or implicit recovery operation.
    """
    if os.name != "posix":
        raise ValueError("POSIX atomic hardlink publication required")
    if (
        type(expected_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
        or type(maximum_blob_bytes) is not int
        or not 1 <= maximum_blob_bytes <= 16 * 1024**3
    ):
        raise ValueError("canonical digest and bounded positive byte budget required")
    for directory in (store, target.parent, source.parent):
        _local_directory(directory)
    if target.exists() or target.is_symlink():
        raise ValueError("new target required; existing artifacts are never overwritten")
    if store.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError("same-volume target required; no hidden copy fallback")
    info = _regular_file(source)
    if not 1 <= info.st_size <= maximum_blob_bytes:
        raise ValueError("complete source exceeds declared byte budget or is empty")
    blob = store / f"{expected_hash[7:]}.blob"

    def check_blob() -> None:
        existing = _regular_file(blob)
        if (
            existing.st_mode & 0o222
            or existing.st_size != info.st_size
            or _digest(blob) != expected_hash
        ):
            raise ValueError("existing shared blob changed or is writable; no repair")

    descriptor, name = tempfile.mkstemp(prefix=".owned-blob-", dir=store)
    temporary = Path(name)
    try:
        total = 0
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output:
            source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(source_descriptor, "rb") as input_:
                while block := input_.read(1024**2):
                    total += len(block)
                    if total > maximum_blob_bytes:
                        raise ValueError("source grew beyond declared byte budget")
                    digest.update(block)
                    output.write(block)
            if total != info.st_size or "sha256:" + digest.hexdigest() != expected_hash:
                raise ValueError("complete source identity changed; nothing published")
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        try:
            os.link(temporary, blob)
        except FileExistsError:
            check_blob()
        else:
            _sync_directory(store)
        check_blob()
        # Link does not replace even if another process publishes the target.
        os.link(blob, target)
        _sync_directory(target.parent)
        return SharedBlobArtifact(expected_hash, total, blob, target)
    finally:
        # Only the exact private temporary created by this invocation is removed.
        temporary.unlink(missing_ok=True)
