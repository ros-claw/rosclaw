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
    # PR-SDB-140-5 (P0-9): who owns the engine lifecycle.  The OWNER started
    # it and closes it; ATTACHERS share its connection options and must never
    # start a second engine nor close the owner's.
    role: str = "owner"


class LocalRuntimeUnavailableError(RuntimeError):
    """The seekdb bindings package is not installable on this platform."""


class SeekDBLocalRuntime:
    """Owns one 1.4 local-runtime instance rooted at ``db_dir``."""

    def __init__(self, db_dir: str | Path):
        self._db_dir = Path(db_dir).resolve()
        self._instance: Any | None = None
        self._info: LocalRuntimeInfo | None = None
        self._role = "owner"

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
        pid = (
            _pid_from_options(options)
            or _first_pid_listening(options)
            or _pid_for_db_dir(self._db_dir)
        )
        if pid is None:
            # P0-1 class rule: a runtime we cannot identify is not usable —
            # ownership, attach, and crash recovery all key off this pid.
            try:
                close = getattr(self._instance, "close", None)
                if callable(close):
                    close()
            finally:
                self._instance = None
            raise RuntimeError(
                f"could not resolve the engine pid for local runtime at "
                f"{self._db_dir} (no pid in connection options, none "
                f"listening on its port, no process rooted at the db dir)"
            )
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

    # ------------------------------------------------------------------
    # shared-instance semantics (PR-SDB-140-5, P0-9)
    # ------------------------------------------------------------------

    @property
    def role(self) -> str:
        return self._role

    def has_live_owner(self) -> bool:
        """True when the lock records a live engine pid for this dir.

        The recorded pid is the ENGINE's, never the client process's, so a
        live pid means the instance is owned (by us or another process) and
        the correct move is attach, not a second start.
        """
        lock = self._read_lock()
        if lock is None:
            return False
        pid = lock.get("pid")
        return bool(pid) and _pid_alive(int(pid))

    def attach(self) -> LocalRuntimeInfo:
        """Attach to a live owner WITHOUT starting or owning the engine.

        The attacher shares the recorded connection options; close() on an
        attacher releases only local state — the owner's engine keeps
        running and its lock stays in place.
        """
        if self._info is not None:
            return self._info
        lock = self._read_lock()
        pid = (lock or {}).get("pid")
        if not lock or not pid or not _pid_alive(int(pid)):
            raise RuntimeError(
                f"no live seekdb local runtime to attach at {self._db_dir} "
                f"(lock {'missing' if lock is None else 'stale'})"
            )
        self._role = "attacher"
        self._info = LocalRuntimeInfo(
            db_dir=str(self._db_dir),
            pid=int(pid),
            started_at=float(lock.get("started_at") or 0.0),
            connection_options=dict(lock.get("connection_options") or {}),
            version=lock.get("version"),
            role="attacher",
        )
        logger.info("seekdb local runtime attached: dir=%s owner pid=%s", self._db_dir, pid)
        return self._info

    def start_or_attach(self) -> LocalRuntimeInfo:
        """Owner when no live runtime exists, attacher when one does."""
        if self._info is not None:
            return self._info
        if self.has_live_owner():
            return self.attach()
        self.recover_if_crashed()
        return self.start()

    def release(self) -> None:
        """Attacher detach: drop local state; the engine is NOT closed."""
        if self._role != "attacher":
            self.close()
            return
        self._info = None
        logger.info("seekdb local runtime detached (engine left running): dir=%s", self._db_dir)

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


# ---------------------------------------------------------------------------
# P0-8: composition — the store OWNS the runtime lifecycle
# ---------------------------------------------------------------------------

from rosclaw.memory.seekdb_client import StructuredStore  # noqa: E402


class LocalRuntimeStructuredStore(StructuredStore):
    """StructuredStore over a 1.4 local runtime, owning its lifecycle.

    Composition, not attachment-by-side-effect (PR-SDB-140-5, P0-8):

        LocalRuntimeStructuredStore
          ├── SeekDBLocalRuntime   (engine lifecycle: start/attach/close)
          └── inner StructuredStore (storage semantics, built from the
                                     runtime's REAL connection options)

    connect:    runtime start-or-attach → inner store built from the actual
                connection_options (host/port OR unix_socket, never dropped)
                → inner.connect()
    disconnect: inner.disconnect() → owner: runtime.close(); attacher:
                runtime.release() (the owner's engine keeps running).
    """

    def __init__(self, db_dir: str | Path, *, database: str = "rosclaw"):
        self._runtime = SeekDBLocalRuntime(db_dir)
        self._database = database
        self._inner: Any | None = None

    # -- lifecycle (the point of the composition) ---------------------------

    def connect(self) -> None:
        if self._inner is not None:
            return
        self._runtime.recover_if_crashed()
        info = self._runtime.start_or_attach()
        self._inner = self._build_inner(info.connection_options)
        try:
            self._inner.connect()
        except BaseException:
            self._inner = None
            if self._runtime.role == "attacher":
                self._runtime.release()
            else:
                self._runtime.close()
            raise
        logger.info(
            "LocalRuntimeStructuredStore connected (%s, role=%s, pid=%s)",
            self._runtime._db_dir,
            self._runtime.role,
            info.pid,
        )

    def _build_inner(self, options: dict[str, Any]) -> Any:
        from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

        socket_path = options.get("unix_socket") or options.get("socket")
        host = options.get("host") or "127.0.0.1"
        port = int(options.get("port") or 2881)
        user = options.get("user") or "root"
        if socket_path:
            # The socket IS the instance identity (P0-7) — never fall through
            # to a default host:port behind its back.  pyseekdb passes
            # **kwargs to pymysql, and pymysql prefers unix_socket when set.
            return SeekDBServerRetrievalStore(
                host="localhost",
                user=user,
                database=self._database,
                unix_socket=str(socket_path),
            )
        return SeekDBServerRetrievalStore(
            host=host,
            port=port,
            user=user,
            database=self._database,
        )

    def is_connected(self) -> bool:
        return self._inner is not None and self._inner.is_connected()

    def disconnect(self) -> None:
        inner, self._inner = self._inner, None
        try:
            if inner is not None:
                inner.disconnect()
        finally:
            if self._runtime.role == "attacher":
                self._runtime.release()
            else:
                self._runtime.close()

    # -- storage semantics: delegate to the inner store ----------------------

    def _require_inner(self) -> Any:
        if self._inner is None:
            raise RuntimeError("LocalRuntimeStructuredStore is not connected")
        return self._inner

    def insert(self, table: str, record: dict) -> str:
        return self._require_inner().insert(table, record)

    def query(
        self, table: str, filters: dict | None = None, order_by: str | None = None, limit: int = 100
    ) -> list[dict]:
        return self._require_inner().query(table, filters, order_by, limit)

    def update(self, table: str, record_id: str, updates: dict) -> bool:
        return self._require_inner().update(table, record_id, updates)

    def count(self, table: str, filters: dict | None = None) -> int:
        return self._require_inner().count(table, filters)

    def delete(self, table: str, record_id: str) -> bool:
        return self._require_inner().delete(table, record_id)

    def delete_where(self, table: str, filters: dict) -> int:
        return self._require_inner().delete_where(table, filters)

    def __getattr__(self, name: str) -> Any:
        # retrieval-plane extras (fulltext_search / similar / hybrid_search…)
        # delegate when the inner store provides them.
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(name)

    @property
    def runtime(self) -> SeekDBLocalRuntime:
        return self._runtime


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


def _pid_for_db_dir(db_dir: Path) -> int | None:
    """Resolve the engine process rooted at this db dir by /proc cmdline.

    The socket-only (unix_socket) case has no port to scan; the engine
    process still carries the db dir in its cmdline/environment footprint.
    """
    target = str(db_dir).encode()
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read()
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            if "seekdb" in comm and target in cmd:
                return int(pid)
        except (OSError, PermissionError):
            continue
    return None
