"""Finite immutable SDK fault audit plus an atomically published completion marker."""

import hashlib
import json
import os
import stat
from pathlib import Path

from simulation_current_facts import _read

MAX_ORIGINAL_BYTES = 10_000_000
ORIGINAL_NAME = "governed-fault-actor-outcome-original.json"
MARKER_NAME = "governed-fault-stop-confirmed-original.json"


def publish_completed_fault_outcome(root, outcome):
    root = Path(root)
    raw = (json.dumps(outcome, indent=2, allow_nan=False) + "\n").encode()
    if not 0 < len(raw) <= MAX_ORIGINAL_BYTES:
        raise ValueError("bounded original fault audit required")
    with (root / ORIGINAL_NAME).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fdatasync(stream.fileno())
    marker = {
        "schema_version": "rosclaw.completed_fault_audit.v1",
        "status": outcome["status"],
        "case": outcome["case"],
        "policy_sha256": outcome["policy_sha256"],
        "run_id": outcome["injection"]["run_id"],
        "body_snapshot_hash": outcome["injection"]["body_snapshot_hash"],
        "after_monotonic_ns": outcome["after_monotonic_ns"],
        "artifact_file": ORIGINAL_NAME,
        "artifact_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    pending = root / "governed-fault-stop-confirmed.pending"
    with pending.open("xb") as stream:
        stream.write((json.dumps(marker, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fdatasync(stream.fileno())
    if (root / MARKER_NAME).exists():
        raise ValueError("completed original marker already exists")
    pending.rename(root / MARKER_NAME)
    return marker


def read_completed_fault_outcome(root, *, expected_policy_sha256):
    root = Path(root)
    marker = json.loads(_read(root / MARKER_NAME, 32_000))
    if (
        type(marker) is not dict
        or set(marker)
        != {
            "schema_version",
            "status",
            "case",
            "policy_sha256",
            "run_id",
            "body_snapshot_hash",
            "after_monotonic_ns",
            "artifact_file",
            "artifact_bytes",
            "artifact_sha256",
        }
        or marker["schema_version"] != "rosclaw.completed_fault_audit.v1"
        or marker["policy_sha256"] != expected_policy_sha256
        or marker["artifact_file"] != ORIGINAL_NAME
        or type(marker["artifact_bytes"]) is not int
        or not 0 < marker["artifact_bytes"] <= MAX_ORIGINAL_BYTES
    ):
        raise ValueError("closed bounded completed original fault marker required")
    fd = os.open(root / ORIGINAL_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != marker["artifact_bytes"]:
            raise ValueError("original fault audit size or type changed")
        digest = hashlib.sha256()
        count = 0
        while count < before.st_size:
            block = os.read(fd, min(65_536, before.st_size - count))
            if not block:
                raise ValueError("original fault audit truncated")
            digest.update(block)
            count += len(block)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or digest.hexdigest() != marker["artifact_sha256"]:
            raise ValueError("original fault audit SHA or immutable bytes changed")
    finally:
        os.close(fd)
    return marker
