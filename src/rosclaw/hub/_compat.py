"""Version-independent safe archive extraction for Hub bundles."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path, PurePosixPath

PathLike = str | Path

DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000
DEFAULT_MAX_UNCOMPRESSED_SIZE = 2 * 1024 * 1024 * 1024


def _safe_member_path(root: Path, name: str) -> tuple[PurePosixPath, Path]:
    member_path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or member_path.is_absolute()
        or ".." in member_path.parts
        or member_path.parts in ((), (".",))
    ):
        raise ValueError(f"Hub bundle contains an unsafe path: {name!r}")
    target = (root / member_path.as_posix()).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Hub bundle contains an unsafe path: {name!r}") from exc
    return member_path, target


def _validated_members(
    archive: tarfile.TarFile,
    root: Path,
    *,
    max_members: int,
    max_total_size: int,
) -> list[tuple[tarfile.TarInfo, PurePosixPath, Path]]:
    validated: list[tuple[tarfile.TarInfo, PurePosixPath, Path]] = []
    member_types: dict[str, str] = {}
    total_size = 0
    for member_number, member in enumerate(archive, start=1):
        if member_number > max_members:
            raise ValueError(f"Hub bundle member limit exceeded: more than {max_members} members")
        member_path, target = _safe_member_path(root, member.name)
        normalized = member_path.as_posix().rstrip("/")
        if normalized in member_types:
            raise ValueError(f"Hub bundle contains a duplicate member: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Hub bundle links are forbidden: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Hub bundle special files are forbidden: {member.name!r}")
        if member.size < 0:
            raise ValueError(f"Hub bundle member has a negative size: {member.name!r}")
        total_size += member.size
        if total_size > max_total_size:
            raise ValueError(
                f"Hub bundle uncompressed size limit exceeded: {total_size} > {max_total_size}"
            )
        member_types[normalized] = "directory" if member.isdir() else "file"
        validated.append((member, member_path, target))

    for _member, member_path, _target in validated:
        for parent in member_path.parents:
            normalized_parent = parent.as_posix().rstrip("/")
            if normalized_parent in ("", "."):
                continue
            if member_types.get(normalized_parent) == "file":
                raise ValueError(
                    f"Hub bundle file is used as a parent directory: {normalized_parent!r}"
                )
    return validated


def _copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Hub bundle member could not be read: {member.name!r}")
    remaining = member.size
    with source, target.open("xb") as output:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Hub bundle member ended early: {member.name!r}")
            output.write(chunk)
            remaining -= len(chunk)
        output.flush()
        os.fsync(output.fileno())
    target.chmod(member.mode & 0o755)


def extractall_tar(
    tar: tarfile.TarFile,
    path: PathLike,
    *,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_total_size: int = DEFAULT_MAX_UNCOMPRESSED_SIZE,
) -> None:
    """Extract only bounded regular files/directories after full validation.

    This does not rely on the Python 3.12 ``tarfile`` filter API, so Python
    3.11 receives the same traversal, link, duplicate, and special-file guards.
    """

    if max_members <= 0 or max_total_size < 0:
        raise ValueError("Hub bundle extraction limits must be positive")
    requested_root = Path(path).expanduser().absolute()
    if requested_root.is_symlink():
        raise ValueError("Hub bundle destination cannot be a symbolic link")
    root = requested_root.resolve()
    validated = _validated_members(
        tar,
        root,
        max_members=max_members,
        max_total_size=max_total_size,
    )

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for member, _member_path, target in validated:
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True, mode=member.mode & 0o755)
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        _copy_member(tar, member, target)


__all__ = [
    "DEFAULT_MAX_ARCHIVE_MEMBERS",
    "DEFAULT_MAX_UNCOMPRESSED_SIZE",
    "extractall_tar",
]
