"""Validate Native pathname UDS limits before bootstrap or persistent writes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def validate_native_socket_paths(home: Path) -> None:
    # sockaddr_un.sun_path includes the trailing NUL: 104 bytes on Darwin,
    # 108 on Linux. This check counts encoded filesystem bytes, not characters.
    limit = 103 if sys.platform == "darwin" else 107
    for name in ("operator.sock", "pi-bridge.sock", "operatord.sock"):
        path = home / "run" / name
        encoded = os.fsencode(path)
        if b"\0" in encoded or len(encoded) > limit:
            raise ValueError(
                f"Native Unix socket path requires {len(encoded)} bytes (limit {limit}): {path}. "
                "Choose a shorter ROSCLAW_HOME/run directory before starting Native Agent; "
                "no socket or control token was created."
            )
