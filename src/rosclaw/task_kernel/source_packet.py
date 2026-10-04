"""Validate pinned declared source bytes before/after a bounded preparation CLI.

Paths are inside the task workspace and symlinks are refused. This is a local
cooperative provenance check, not OS isolation or a proof of physical behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from rosclaw.contracts.source_packet import SourcePacketRefV1, SourcePacketV1

MAX_PACKET_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source packet JSON key")
        result[key] = value
    return result


def _open_source(workspace: Path, relative: str, maximum: int) -> BinaryIO:
    root = workspace.resolve()
    path = root
    for part in relative.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError("symlink source path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("source path outside workspace")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(fd, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        stream.close()
        raise ValueError("source file type or size bound")
    return stream


def verify_source_packet(reference: object, run: object, workspace: Path) -> list[str]:
    """Fail closed when opt-in provenance or direct entry binding is invalid."""
    try:
        ref = SourcePacketRefV1.model_validate(reference)
        with _open_source(workspace, ref.path, MAX_PACKET_BYTES) as stream:
            raw = stream.read(MAX_PACKET_BYTES + 1)
        if len(raw) > MAX_PACKET_BYTES:
            raise ValueError("source packet grew beyond size bound")
        if hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise ValueError("pinned packet digest mismatch")
        packet = SourcePacketV1.model_validate(json.loads(raw, object_pairs_hook=_unique_object))
        if ref.path in packet.files:
            raise ValueError("packet cannot include itself in its hashed closure")
        if not isinstance(run, dict) or run.get("argv") != packet.preflight_argv:
            raise ValueError("acceptance argv must equal pinned canonical preflight argv")
        timeout = run.get("timeout_sec", 600)
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= 600):
            raise ValueError("preflight timeout must be finite, positive and at most 600s")
        for path, record in packet.files.items():
            digest = hashlib.sha256()
            count = 0
            with _open_source(workspace, path, record.size_bytes) as stream:
                while chunk := stream.read(min(1024 * 1024, record.size_bytes - count + 1)):
                    count += len(chunk)
                    if count > record.size_bytes:
                        raise ValueError("declared source file grew beyond size bound")
                    digest.update(chunk)
            if count != record.size_bytes or digest.hexdigest() != record.sha256:
                raise ValueError("declared source file digest or size mismatch")
    except (OSError, ValueError, TypeError, RecursionError, ValidationError):
        # Do not echo user packet values, argv, or file contents into tool errors.
        return ["SOURCE_PACKET_INVALID: pinned declared closure or canonical preflight binding failed"]
    return []
