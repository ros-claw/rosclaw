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
        if (
            len(raw) != before.st_size
            or not unchanged_content_metadata
            or (before.st_ctime_ns != after.st_ctime_ns and not atomic_unlink_only)
        ):
            raise ValueError("original observation changed during read")
        return raw
    finally:
        os.close(fd)
