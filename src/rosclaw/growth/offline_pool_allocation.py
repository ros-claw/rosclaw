"""One-time offline evaluation-pool bookkeeping, never execution authority.

A pool is consumed on allocation, BEFORE any labels or physics are observed.
Interrupted writes deliberately poison its allocation directory: no automatic
retry, release, renamed-output bypass, or different-candidate reuse is offered.
This ledger authenticates no prerequisite review or private-data isolation.
Applications must independently verify those before allocating or executing.
The operator must pin one canonical ledger root outside candidate-controlled
output directories. A second ledger root cannot detect use in the first;
these same-UID records are corruption checks, not signed review evidence.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LIMIT = 65536
_SCHEMA = "rosclaw.growth.offline_pool_allocation.v1"
_CEILING = "OFFLINE_EVALUATION_BOOKKEEPING_ONLY"


@dataclass(frozen=True)
class OfflinePoolBinding:
    """Fixed predeclared identity; hashes contain no private test-case rows."""

    pool_hash: str
    protocol_hash: str
    train_split_hash: str
    parent_hash: str
    candidate_hashes: tuple[str, ...]
    execution_commitment_hash: str

    def __post_init__(self) -> None:
        if type(self.candidate_hashes) is not tuple or not 1 <= len(self.candidate_hashes) <= 16:
            raise ValueError("one to sixteen unique fixed candidate hashes required")
        for value in (
            self.pool_hash,
            self.protocol_hash,
            self.train_split_hash,
            self.parent_hash,
            *self.candidate_hashes,
            self.execution_commitment_hash,
        ):
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise ValueError("exact sha256 content identities required")
        if len(set(self.candidate_hashes)) != len(self.candidate_hashes):
            raise ValueError("one to sixteen unique fixed candidate hashes required")
        if self.parent_hash in self.candidate_hashes:
            raise ValueError("parent cannot also identify a candidate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_hash": self.pool_hash,
            "protocol_hash": self.protocol_hash,
            "train_split_hash": self.train_split_hash,
            "parent_hash": self.parent_hash,
            "candidate_hashes": list(self.candidate_hashes),
            "execution_commitment_hash": self.execution_commitment_hash,
        }


def _record(binding: OfflinePoolBinding) -> dict[str, Any]:
    if type(binding) is not OfflinePoolBinding:
        raise ValueError("exact frozen offline pool binding required")
    # Revalidate even if somebody bypassed frozen-dataclass assignment.
    owned = OfflinePoolBinding(**vars(binding))
    result = {
        "schema": _SCHEMA,
        "binding": owned.to_dict(),
        "partition_after_allocation": "CONSUMED_NO_REUSE_AS_FRESH",
        "allocation_ceiling": _CEILING,
        "fresh_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    return dict(result, record_hash=CanonicalJSONSnapshot(result).content_hash)


def _private_directory(fd: int) -> None:
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("operator-owned mode0700 offline ledger directory required")


def _open_root(root: Path) -> int:
    # Walk every component by descriptor, not a check-then-follow pathname.
    # O_NOFOLLOW protects intermediates as well as the final root.
    path = root.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            if component in (".", ".."):
                raise ValueError("ordinary absolute offline ledger path required")
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        _private_directory(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def allocate_once(root: Path, binding: OfflinePoolBinding) -> dict[str, Any]:
    """Durably consume a pool; caller may proceed only after successful return.

    Root must already exist with mode0700. Exclusive mkdir is keyed ONLY by
    pool identity, not candidate, protocol, output path or execution identity.
    An existing directory always refuses allocation, even for an exact retry.
    Failure never deletes state. This is not a simulator admission decision.
    """
    record = _record(binding)
    data = CanonicalJSONSnapshot(record)._data
    if len(data) > _LIMIT:
        raise ValueError("bounded offline allocation record required")
    root_fd = _open_root(root)
    pool_fd = -1
    try:
        name = record["binding"]["pool_hash"][7:]
        os.mkdir(name, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        pool_fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_fd,
        )
        _private_directory(pool_fd)
        fd = os.open(
            "allocation.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=pool_fd,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(pool_fd)
        os.fsync(root_fd)
        return record
    finally:
        if pool_fd >= 0:
            os.close(pool_fd)
        os.close(root_fd)


def inspect_allocation(root: Path, binding: OfflinePoolBinding) -> dict[str, Any]:
    """Read exact prior identity; never grant retry or execution authorization."""
    expected = _record(binding)
    root_fd = _open_root(root)
    pool_fd = -1
    try:
        pool_fd = os.open(
            expected["binding"]["pool_hash"][7:],
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_fd,
        )
        _private_directory(pool_fd)
        fd = os.open(
            "allocation.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=pool_fd,
        )
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > _LIMIT
            ):
                raise ValueError("bounded private ordinary allocation record required")
            data = stream.read(_LIMIT + 1)
        if len(data) > _LIMIT:
            raise ValueError("bounded offline allocation record required")
        # Exact canonical bytes reject duplicate keys, whitespace rewrites,
        # nonfinite numbers and even re-sealed changed candidate identities.
        canonical = CanonicalJSONSnapshot(expected)._data
        if data != canonical:
            raise ValueError("offline pool allocation identity or record changed")
        result: dict[str, Any] = json.loads(data)
        return result
    finally:
        if pool_fd >= 0:
            os.close(pool_fd)
        os.close(root_fd)
