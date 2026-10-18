"""Bounded SDK file reads, with explicit atomic-snapshot unlink semantics."""

import os
import stat


def read_bounded_regular(path, limit, *, allow_atomic_unlink=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("bounded original regular file required")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        unchanged_content_metadata = (before.st_size, before.st_mtime_ns) == (
            after.st_size,
            after.st_mtime_ns,
        )
        atomic_unlink_only = (
            allow_atomic_unlink is True
            and before.st_nlink == 1
            and after.st_nlink == 0
            and unchanged_content_metadata
        )
        if len(raw) != before.st_size or not unchanged_content_metadata:
            raise ValueError("original observation changed during read")
        if before.st_ctime_ns != after.st_ctime_ns and not atomic_unlink_only:
            # Linux rename can expose the replaced inode's zero link count
            # before its ctime update, or the incoming inode before its rename
            # ctime update. Link-count transitions alone do not identify this
            # race. Only explicit mutable snapshots may verify the SAME open
            # inode again; immutable originals retain the strict ctime rule.
            if allow_atomic_unlink is not True:
                raise ValueError("original observation changed during read")
            repeated = os.pread(fd, limit + 1, 0)
            final = os.fstat(fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_mode",
                "st_uid",
                "st_gid",
            )
            if (
                repeated != raw
                or final.st_ctime_ns != after.st_ctime_ns
                or any(
                    len({getattr(info, key) for info in (before, after, final)}) != 1
                    for key in stable_fields
                )
                or not 0 <= final.st_nlink <= after.st_nlink <= before.st_nlink <= 1
            ):
                raise ValueError("original observation changed during read")
        return raw
    finally:
        os.close(fd)
