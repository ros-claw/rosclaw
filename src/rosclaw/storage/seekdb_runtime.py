"""SeekDB 1.4 local-runtime lifecycle (PR-SDB-140-4, outline §二十).

1.4's embedded is no longer a library linked into the host process: the
engine runs as a separate background process and applications talk to it
over the MySQL protocol via the new `seekdb` bindings package:

    import seekdb
    instance = seekdb.open("./seekdb.db")   # owns the background engine
    options = instance.connection_options() # host/port/socket for clients

This module owns ONLY the lifecycle — start / health / connection options /
pid / socket / close / crash recovery — never storage semantics (those stay
in the store adapters).  Legacy pylibseekdb's one-embedded-target-per-process
limit belongs to ``legacy_embedded`` and is NOT carried here: 1.4 local
runtimes are multi-instance by design.

Availability: the `seekdb` package currently ships x86_64-only wheels
(1.4.0.dev2), so ``SeekDBLocalRuntime.available()`` is False on aarch64 —
callers must fail closed (no silent fallback to another backend).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.storage.seekdb_runtime")

# The lock file for crash recovery: a live runtime stamps its connection
# options here; a stale lock + dead pid = crashed engine => safe to reopen.
_LOCK_NAME = "runtime.lock.json"


@dataclass(frozen=True)
class LocalRuntimeInfo:
    db_dir: str
    pid: int | None
    started_at: float
    connection_options: dict[str, Any]
    version: str | None


class LocalRuntimeUnavailableError(RuntimeError):
    """The seekdb bindings package is not installable on this platform."""


class SeekDBLocalRuntime:
    """Owns one 1.4 local-runtime instance rooted at ``db_dir``."""

    def __init__(self, db_dir: str | Path):
        self._db_dir = Path(db_dir).resolve()
        self._instance: Any | None = None
        self._info: LocalRuntimeInfo | None = None

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------

    @staticmethod
    def available() -> bool:
        """True when the new seekdb bindings are importable (x86_64 today)."""
        try:
            import seekdb  # noqa: F401

            return hasattr(seekdb, "open")
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def bindings_version() -> str | None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version("seekdb")
            except PackageNotFoundError:
                return None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> LocalRuntimeInfo:
        """Open (start) the runtime; idempotent; recovers from a stale lock."""
        if self._instance is not None:
            assert self._info is not None
            return self._info
        if not self.available():
            raise LocalRuntimeUnavailableError(
                "the 'seekdb' bindings package is not available on this "
                "platform (1.4.0.dev2 ships x86_64 wheels only).  Use "
                "ROSCLAW_SEEKDB_MODE=server or legacy_embedded."
            )
        self._db_dir.mkdir(parents=True, exist_ok=True)
        stale = self._read_lock()
        if stale is not None:
            pid = stale.get("pid")
            if pid and _pid_alive(int(pid)):
                raise RuntimeError(
                    f"seekdb local runtime at {self._db_dir} is already owned by "
                    f"live pid {pid} — multi-instance means multiple DIRECTORIES, "
                    "not multiple owners of one directory"
                )
            logger.info("recovering stale runtime lock at %s (dead pid %s)", self._db_dir, pid)
            self._lock_path().unlink(missing_ok=True)

        import seekdb

        started = time.time()
        self._instance = seekdb.open(str(self._db_dir))
        options = dict(self._instance.connection_options())
        pid = _pid_from_options(options) or _first_pid_listening(options)
        self._info = LocalRuntimeInfo(
            db_dir=str(self._db_dir),
            pid=pid,
            started_at=started,
            connection_options=options,
            version=self.bindings_version(),
        )
        self._write_lock()
        logger.info(
            "seekdb local runtime started: dir=%s pid=%s version=%s",
            self._db_dir,
            pid,
            self._info.version,
        )
        return self._info

    def health(self) -> dict[str, Any]:
        """Health probe: instance open, lock present, pid alive."""
        lock = self._read_lock()
        pid = (lock or {}).get("pid") or (self._info.pid if self._info else None)
        alive = bool(pid) and _pid_alive(int(pid))
        return {
            "open": self._instance is not None,
            "lock_present": lock is not None,
            "pid": pid,
            "pid_alive": alive,
            "db_dir": str(self._db_dir),
            "healthy": self._instance is not None and alive,
        }

    def connection_options(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("runtime not started")
        return dict(self._info.connection_options)

    def close(self) -> None:
        """Graceful shutdown; removes the ownership lock."""
        if self._instance is None:
            return
        instance, self._instance = self._instance, None
        try:
            close = getattr(instance, "close", None)
            if callable(close):
                close()
        finally:
            self._lock_path().unlink(missing_ok=True)
            self._info = None
            logger.info("seekdb local runtime closed: dir=%s", self._db_dir)

    # ------------------------------------------------------------------
    # crash recovery
    # ------------------------------------------------------------------

    def recover_if_crashed(self) -> bool:
        """True if a stale lock was found and cleared (engine died badly)."""
        if self._instance is not None:
            return False
        lock = self._read_lock()
        if lock is None:
            return False
        pid = lock.get("pid")
        if pid and _pid_alive(int(pid)):
            return False
        logger.warning(
            "seekdb local runtime at %s crashed (pid %s gone); clearing lock for reopen",
            self._db_dir,
            pid,
        )
        self._lock_path().unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _lock_path(self) -> Path:
        return self._db_dir / _LOCK_NAME

    def _write_lock(self) -> None:
        import json

        assert self._info is not None
        payload = {
            "pid": self._info.pid,
            "started_at": self._info.started_at,
            "version": self._info.version,
            "connection_options": self._info.connection_options,
        }
        tmp = self._lock_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(self._lock_path())

    def _read_lock(self) -> dict[str, Any] | None:
        import json

        try:
            return json.loads(self._lock_path().read_text())
        except Exception:  # noqa: BLE001
            return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_from_options(options: dict[str, Any]) -> int | None:
    for key in ("pid", "process_id"):
        value = options.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _first_pid_listening(options: dict[str, Any]) -> int | None:
    """Best-effort: find the seekdb process serving this runtime's port."""
    port = options.get("port")
    if not port:
        return None
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-f", f"seekdb.*{port}"], capture_output=True, text=True, timeout=5
        ).stdout.split()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None
