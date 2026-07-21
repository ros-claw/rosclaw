"""Integrity-checked durable event ledger for rosclawd."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import sqlite3
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rosclaw.firstboot.workspace import get_rosclaw_home

LEDGER_SCHEMA_VERSION = "rosclaw.daemon.ledger.v1"
_HMAC_CONTEXT = b"rosclaw.daemon.ledger.event.v1\0"
_ANCHOR_CONTEXT = b"rosclaw.daemon.ledger.anchor.v1\0"


class LedgerError(RuntimeError):
    """The daemon ledger cannot be opened or updated safely."""


class LedgerIntegrityError(LedgerError):
    """Persisted ledger content failed an integrity invariant."""


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable, authenticated daemon state transition."""

    sequence: int
    event_type: str
    entity_kind: str
    entity_id: str
    created_at: str
    payload: dict[str, Any]
    previous_mac: str
    event_mac: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "entity_kind": self.entity_kind,
            "entity_id": self.entity_id,
            "created_at": self.created_at,
            "payload": self.payload,
            "previous_mac": self.previous_mac,
            "event_mac": self.event_mac,
        }


class DaemonLedger:
    """Append-only SQLite ledger authenticated by a daemon-private HMAC key."""

    def __init__(self, path: str | Path, *, key_path: str | Path | None = None) -> None:
        self.path = _absolute_path(path)
        self.key_path = _absolute_path(key_path if key_path is not None else f"{self.path}.key")
        self.anchor_path = _absolute_path(f"{self.path}.anchor")
        self._lock = threading.RLock()
        self._closed = False
        self._failure: str | None = None
        self._observed_data_version: int | None = None
        _prepare_private_parent(self.path.parent)
        _prepare_private_parent(self.key_path.parent)
        self._database_existed = self.path.exists()
        self._key = _load_or_create_key(self.key_path)
        self._key_id = hashlib.sha256(self._key).hexdigest()
        self._connection = self._open_database()
        try:
            self._initialize_schema()
            if self._event_count() == 0:
                if self._database_existed or self.anchor_path.exists():
                    raise LedgerIntegrityError(
                        "daemon ledger rollback detected: an existing ledger is empty"
                    )
                self._append_locked(
                    "LEDGER_CREATED",
                    entity_kind="LEDGER",
                    entity_id=LEDGER_SCHEMA_VERSION,
                    payload={
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "key_id": self._key_id,
                    },
                )
            self._verify_chain_locked()
            self._verify_anchor_locked()
            self._observed_data_version = self._data_version_locked()
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    def __enter__(self) -> DaemonLedger:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def append(
        self,
        event_type: str,
        *,
        entity_kind: str,
        entity_id: str,
        payload: dict[str, Any],
    ) -> LedgerEvent:
        """Durably append one authenticated event."""

        _validate_identifier("event_type", event_type)
        _validate_identifier("entity_kind", entity_kind)
        _validate_identifier("entity_id", entity_id)
        if not isinstance(payload, dict):
            raise TypeError("ledger payload must be a JSON object")
        with self._lock:
            self._require_open()
            return self._append_locked(
                event_type,
                entity_kind=entity_kind,
                entity_id=entity_id,
                payload=payload,
            )

    def events(
        self,
        *,
        entity_kind: str | None = None,
        entity_id: str | None = None,
    ) -> list[LedgerEvent]:
        """Read verified events, optionally scoped to one durable entity."""

        if entity_id is not None and entity_kind is None:
            raise ValueError("entity_id filtering requires entity_kind")
        with self._lock:
            self._require_open()
            self._verify_external_changes_locked()
            query = (
                "SELECT sequence, event_type, entity_kind, entity_id, created_at, "
                "payload_json, previous_mac, event_mac FROM ledger_events"
            )
            parameters: tuple[str, ...] = ()
            if entity_kind is not None and entity_id is not None:
                query += " WHERE entity_kind = ? AND entity_id = ?"
                parameters = (entity_kind, entity_id)
            elif entity_kind is not None:
                query += " WHERE entity_kind = ?"
                parameters = (entity_kind,)
            query += " ORDER BY sequence"
            rows = self._connection.execute(query, parameters).fetchall()
            return [self._decode_event(row) for row in rows]

    def status(self) -> dict[str, Any]:
        """Return non-secret durability and integrity status."""

        with self._lock:
            self._require_open()
            self._verify_external_changes_locked()
            sequence, event_mac = self._head_locked()
            return {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "path": str(self.path),
                "anchor_path": str(self.anchor_path),
                "integrity_verified": True,
                "event_count": sequence,
                "head_sequence": sequence,
                "head_mac": event_mac,
                "key_id": self._key_id,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _open_database(self) -> sqlite3.Connection:
        _assert_private_regular_file(self.path, required=False)
        connection = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        try:
            os.chmod(self.path, 0o600)
            _assert_private_regular_file(self.path, required=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA busy_timeout=5000")
        except Exception:
            connection.close()
            raise
        return connection

    def _initialize_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_events (
                sequence INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                entity_kind TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_mac TEXT NOT NULL,
                event_mac TEXT NOT NULL UNIQUE
            ) STRICT
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS ledger_events_entity
            ON ledger_events(entity_kind, entity_id, sequence)
            """
        )

    def _append_locked(
        self,
        event_type: str,
        *,
        entity_kind: str,
        entity_id: str,
        payload: dict[str, Any],
    ) -> LedgerEvent:
        payload_json = _canonical_json(payload)
        created_at = _utc_now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._verify_external_changes_locked()
            previous_sequence, previous_mac = self._head_locked()
            sequence = previous_sequence + 1
            event_mac = _event_mac(
                self._key,
                sequence=sequence,
                event_type=event_type,
                entity_kind=entity_kind,
                entity_id=entity_id,
                created_at=created_at,
                payload_json=payload_json,
                previous_mac=previous_mac,
            )
            # Advance the independently signed head before committing the event.
            # A crash between these writes makes the next startup fail closed
            # instead of accepting a committed event whose rollback is unwitnessed.
            self._write_anchor(sequence, event_mac)
            self._connection.execute(
                """
                INSERT INTO ledger_events (
                    sequence, event_type, entity_kind, entity_id, created_at,
                    payload_json, previous_mac, event_mac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    event_type,
                    entity_kind,
                    entity_id,
                    created_at,
                    payload_json,
                    previous_mac,
                    event_mac,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception as exc:
            with contextlib.suppress(sqlite3.Error):
                self._connection.execute("ROLLBACK")
            self._failure = f"{type(exc).__name__}: {exc}"[:512]
            raise
        return LedgerEvent(
            sequence=sequence,
            event_type=event_type,
            entity_kind=entity_kind,
            entity_id=entity_id,
            created_at=created_at,
            payload=cast(dict[str, Any], json.loads(payload_json)),
            previous_mac=previous_mac,
            event_mac=event_mac,
        )

    def _verify_chain_locked(self) -> None:
        integrity = self._connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise LedgerIntegrityError("daemon ledger failed SQLite integrity_check")
        rows = self._connection.execute(
            """
            SELECT sequence, event_type, entity_kind, entity_id, created_at,
                   payload_json, previous_mac, event_mac
            FROM ledger_events ORDER BY sequence
            """
        ).fetchall()
        expected_sequence = 1
        previous_mac = "GENESIS"
        first_event: LedgerEvent | None = None
        for row in rows:
            event = self._decode_event(row)
            if first_event is None:
                first_event = event
            if event.sequence != expected_sequence:
                raise LedgerIntegrityError("daemon ledger event sequence is not contiguous")
            if not hmac.compare_digest(event.previous_mac, previous_mac):
                raise LedgerIntegrityError("daemon ledger previous-MAC chain is invalid")
            payload_json = _canonical_json(event.payload)
            expected_mac = _event_mac(
                self._key,
                sequence=event.sequence,
                event_type=event.event_type,
                entity_kind=event.entity_kind,
                entity_id=event.entity_id,
                created_at=event.created_at,
                payload_json=payload_json,
                previous_mac=event.previous_mac,
            )
            if not hmac.compare_digest(event.event_mac, expected_mac):
                raise LedgerIntegrityError(
                    f"daemon ledger event {event.sequence} failed HMAC verification"
                )
            previous_mac = event.event_mac
            expected_sequence += 1
        if first_event is None:
            raise LedgerIntegrityError("daemon ledger has no creation event")
        if (
            first_event.event_type != "LEDGER_CREATED"
            or first_event.entity_kind != "LEDGER"
            or first_event.entity_id != LEDGER_SCHEMA_VERSION
            or first_event.payload.get("schema_version") != LEDGER_SCHEMA_VERSION
            or first_event.payload.get("key_id") != self._key_id
        ):
            raise LedgerIntegrityError("daemon ledger creation event is invalid")

    def _verify_anchor_locked(self) -> None:
        anchor = _read_private_json(self.anchor_path)
        expected_fields = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "key_id": self._key_id,
            "head_sequence": anchor.get("head_sequence"),
            "head_mac": anchor.get("head_mac"),
        }
        expected_mac = _anchor_mac(self._key, expected_fields)
        anchor_mac = str(anchor.get("anchor_mac", ""))
        if not hmac.compare_digest(anchor_mac, expected_mac):
            raise LedgerIntegrityError("daemon ledger anchor failed HMAC verification")
        try:
            anchor_sequence = int(anchor["head_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerIntegrityError("daemon ledger anchor sequence is invalid") from exc
        anchor_head_mac = str(anchor.get("head_mac", ""))
        head_sequence, head_mac = self._head_locked()
        if anchor_sequence > head_sequence:
            raise LedgerIntegrityError(
                "daemon ledger rollback detected: database head precedes signed anchor"
            )
        anchored_row = self._connection.execute(
            "SELECT event_mac FROM ledger_events WHERE sequence = ?",
            (anchor_sequence,),
        ).fetchone()
        if anchored_row is None or not hmac.compare_digest(
            str(anchored_row["event_mac"]), anchor_head_mac
        ):
            raise LedgerIntegrityError(
                "daemon ledger rollback detected: signed anchor is not in the event chain"
            )
        if head_sequence > anchor_sequence:
            self._write_anchor(head_sequence, head_mac)
        elif not hmac.compare_digest(head_mac, anchor_head_mac):
            raise LedgerIntegrityError("daemon ledger anchor does not match the database head")

    def _write_anchor(self, sequence: int, event_mac: str) -> None:
        fields: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "key_id": self._key_id,
            "head_sequence": sequence,
            "head_mac": event_mac,
        }
        payload = dict(fields)
        payload["anchor_mac"] = _anchor_mac(self._key, fields)
        _atomic_private_write(self.anchor_path, (_canonical_json(payload) + "\n").encode())

    def _decode_event(self, row: sqlite3.Row) -> LedgerEvent:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError("daemon ledger event payload is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LedgerIntegrityError("daemon ledger event payload must be an object")
        return LedgerEvent(
            sequence=int(row["sequence"]),
            event_type=str(row["event_type"]),
            entity_kind=str(row["entity_kind"]),
            entity_id=str(row["entity_id"]),
            created_at=str(row["created_at"]),
            payload=cast(dict[str, Any], payload),
            previous_mac=str(row["previous_mac"]),
            event_mac=str(row["event_mac"]),
        )

    def _event_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM ledger_events").fetchone()
        return int(row[0]) if row is not None else 0

    def _data_version_locked(self) -> int:
        row = self._connection.execute("PRAGMA data_version").fetchone()
        if row is None:
            raise LedgerIntegrityError("daemon ledger data_version is unavailable")
        return int(row[0])

    def _verify_external_changes_locked(self) -> None:
        if self._observed_data_version is None:
            return
        current = self._data_version_locked()
        try:
            if current != self._observed_data_version:
                self._verify_chain_locked()
            self._verify_anchor_locked()
        except Exception as exc:
            self._failure = f"external ledger modification: {type(exc).__name__}: {exc}"[:512]
            raise
        self._observed_data_version = current

    def _head_locked(self) -> tuple[int, str]:
        row = self._connection.execute(
            "SELECT sequence, event_mac FROM ledger_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, "GENESIS"
        return int(row["sequence"]), str(row["event_mac"])

    def _require_open(self) -> None:
        if self._closed:
            raise LedgerError("daemon ledger is closed")
        if self._failure is not None:
            raise LedgerError(
                "daemon ledger is unavailable after an integrity or durable-write "
                f"failure: {self._failure}"
            )


def get_daemon_ledger_path(path: str | Path | None = None) -> Path:
    """Resolve the daemon event database path."""

    configured = path or os.environ.get("ROSCLAW_DAEMON_LEDGER")
    if configured is not None:
        return _absolute_path(configured)
    return get_rosclaw_home() / "state" / "daemon" / "ledger.sqlite3"


def get_daemon_ledger_key_path(path: str | Path | None = None) -> Path:
    """Resolve the daemon-private ledger HMAC key path."""

    configured = path or os.environ.get("ROSCLAW_DAEMON_LEDGER_KEY")
    if configured is not None:
        return _absolute_path(configured)
    return get_rosclaw_home() / "state" / "daemon" / "ledger.key"


def _canonical_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"daemon ledger payload is not canonical JSON: {exc}") from exc


def _event_mac(
    key: bytes,
    *,
    sequence: int,
    event_type: str,
    entity_kind: str,
    entity_id: str,
    created_at: str,
    payload_json: str,
    previous_mac: str,
) -> str:
    fields: dict[str, Any] = {
        "sequence": sequence,
        "event_type": event_type,
        "entity_kind": entity_kind,
        "entity_id": entity_id,
        "created_at": created_at,
        "payload_json": payload_json,
        "previous_mac": previous_mac,
    }
    message = _HMAC_CONTEXT + _canonical_json(fields).encode("utf-8")
    return f"hmac-sha256:{hmac.new(key, message, hashlib.sha256).hexdigest()}"


def _anchor_mac(key: bytes, fields: dict[str, Any]) -> str:
    message = _ANCHOR_CONTEXT + _canonical_json(fields).encode("utf-8")
    return f"hmac-sha256:{hmac.new(key, message, hashlib.sha256).hexdigest()}"


def _load_or_create_key(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, create_flags, 0o600)
        key = os.urandom(32)
        try:
            _write_all(descriptor, key)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
        _assert_private_regular_file(path, required=True)
        return key
    try:
        metadata = os.fstat(descriptor)
        _assert_private_metadata(path, metadata)
        key = os.read(descriptor, 4096)
    finally:
        os.close(descriptor)
    if len(key) != 32:
        raise LedgerIntegrityError("daemon ledger key must contain exactly 32 bytes")
    return key


def _prepare_private_parent(path: Path) -> None:
    _assert_no_symlink_ancestry(path)
    if path.is_symlink():
        raise LedgerError(f"daemon ledger directory cannot be a symbolic link: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise LedgerError(f"daemon ledger parent must be a directory: {path}")
    if metadata.st_uid != os.geteuid():
        raise LedgerError(f"daemon ledger directory must be owned by uid {os.geteuid()}: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LedgerError(f"daemon ledger directory must not grant group/world access: {path}")


def _assert_private_regular_file(path: Path, *, required: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if required:
            raise LedgerError(f"daemon ledger file does not exist: {path}") from None
        return
    _assert_private_metadata(path, metadata)


def _read_private_json(path: Path) -> dict[str, Any]:
    _assert_private_regular_file(path, required=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        _assert_private_metadata(path, metadata)
        raw = os.read(descriptor, 64 * 1024)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError("daemon ledger anchor is invalid JSON") from exc
    if not isinstance(value, dict):
        raise LedgerIntegrityError("daemon ledger anchor must be a JSON object")
    return cast(dict[str, Any], value)


def _atomic_private_write(path: Path, data: bytes) -> None:
    _assert_private_regular_file(path, required=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        _write_all(descriptor, view)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _assert_private_metadata(path: Path, metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise LedgerError(f"daemon ledger file must be a regular file: {path}")
    if metadata.st_uid != os.geteuid():
        raise LedgerError(f"daemon ledger file must be owned by uid {os.geteuid()}: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LedgerError(f"daemon ledger file must not grant group/world access: {path}")


def _write_all(descriptor: int, data: bytes | memoryview) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while persisting daemon ledger state")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _assert_no_symlink_ancestry(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise LedgerError(
                f"daemon ledger path ancestry cannot contain a symbolic link: {current}"
            )


def _validate_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(
            f"{name} must be a non-empty string of at most 256 characters "
            "without control characters"
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "DaemonLedger",
    "LedgerError",
    "LedgerEvent",
    "LedgerIntegrityError",
    "get_daemon_ledger_key_path",
    "get_daemon_ledger_path",
]
