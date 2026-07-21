"""Durable rosclawd ledger contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rosclaw.daemon.ledger import (
    DaemonLedger,
    LedgerError,
    LedgerIntegrityError,
    get_daemon_ledger_key_path,
    get_daemon_ledger_path,
)


def test_authenticated_event_survives_ledger_reopen(tmp_path: Path) -> None:
    database = tmp_path / "state" / "daemon-ledger.sqlite3"
    key = tmp_path / "secrets" / "daemon-ledger.key"

    with DaemonLedger(database, key_path=key) as ledger:
        written = ledger.append(
            "PERMIT_REGISTERED",
            entity_kind="PERMIT",
            entity_id="permit-1",
            payload={"permit_id": "permit-1", "max_uses": 1},
        )

    with DaemonLedger(database, key_path=key) as reopened:
        events = reopened.events(entity_kind="PERMIT", entity_id="permit-1")
        status = reopened.status()

    assert [event.to_dict() for event in events] == [written.to_dict()]
    assert status["integrity_verified"] is True
    assert status["event_count"] == 2  # LEDGER_CREATED + domain event


def test_ledger_refuses_offline_payload_tampering(tmp_path: Path) -> None:
    database = tmp_path / "state" / "daemon-ledger.sqlite3"
    key = tmp_path / "secrets" / "daemon-ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "PERMIT_CONSUMED",
            entity_kind="PERMIT",
            entity_id="permit-1",
            payload={"permit_id": "permit-1", "action_id": "action-1"},
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ledger_events SET payload_json = ? WHERE sequence = 2",
            ('{"action_id":"attacker-action","permit_id":"permit-1"}',),
        )

    with pytest.raises(LedgerIntegrityError, match="HMAC"):
        DaemonLedger(database, key_path=key)


def test_open_ledger_detects_an_external_sqlite_mutation(tmp_path: Path) -> None:
    database = tmp_path / "state" / "daemon-ledger.sqlite3"
    key = tmp_path / "secrets" / "daemon-ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id="action-1",
            payload={"action_id": "action-1"},
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE ledger_events SET payload_json = ? WHERE sequence = 2",
                ('{"action_id":"mutated"}',),
            )

        with pytest.raises(LedgerIntegrityError, match="HMAC"):
            ledger.status()
        with pytest.raises(LedgerError, match="integrity or durable-write"):
            ledger.events()


def test_open_ledger_detects_external_anchor_tampering(tmp_path: Path) -> None:
    database = tmp_path / "state" / "daemon-ledger.sqlite3"
    key = tmp_path / "secrets" / "daemon-ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        anchor = json.loads(ledger.anchor_path.read_text(encoding="utf-8"))
        anchor["head_sequence"] = 999
        ledger.anchor_path.write_text(json.dumps(anchor), encoding="utf-8")

        with pytest.raises(LedgerIntegrityError, match="anchor.*HMAC"):
            ledger.status()
        with pytest.raises(LedgerError, match="integrity or durable-write"):
            ledger.events()


def test_ledger_refuses_rollback_to_an_older_valid_chain(tmp_path: Path) -> None:
    database = tmp_path / "state" / "daemon-ledger.sqlite3"
    key = tmp_path / "secrets" / "daemon-ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id="action-1",
            payload={"action_id": "action-1"},
        )
        ledger.append(
            "ACTION_TERMINAL",
            entity_kind="ACTION",
            entity_id="action-1",
            payload={"action_id": "action-1", "state": "FINISHED"},
        )

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM ledger_events WHERE sequence = 3")

    with pytest.raises(LedgerIntegrityError, match="rollback"):
        DaemonLedger(database, key_path=key)


def test_daemon_ledger_defaults_to_private_workspace_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))

    assert get_daemon_ledger_path() == tmp_path / "home" / "state" / "daemon" / "ledger.sqlite3"
    assert get_daemon_ledger_key_path() == tmp_path / "home" / "state" / "daemon" / "ledger.key"

    monkeypatch.setenv("ROSCLAW_DAEMON_LEDGER", str(tmp_path / "custom" / "events.db"))
    monkeypatch.setenv("ROSCLAW_DAEMON_LEDGER_KEY", str(tmp_path / "custom" / "events.key"))
    assert get_daemon_ledger_path() == tmp_path / "custom" / "events.db"
    assert get_daemon_ledger_key_path() == tmp_path / "custom" / "events.key"


def test_ledger_rejects_symlink_in_private_path_ancestry(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir(mode=0o700)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(LedgerError, match="symbolic link"):
        DaemonLedger(
            linked_root / "state" / "ledger.sqlite3",
            key_path=linked_root / "state" / "ledger.key",
        )


def test_ledger_rejects_group_readable_machine_key(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key):
        pass
    key.chmod(0o640)

    with pytest.raises(LedgerError, match="group/world"):
        DaemonLedger(database, key_path=key)


def test_ledger_rejects_signed_anchor_tampering(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        anchor = ledger.anchor_path
    payload = json.loads(anchor.read_text(encoding="utf-8"))
    payload["head_sequence"] = 999
    anchor.write_text(json.dumps(payload), encoding="utf-8")
    anchor.chmod(0o600)

    with pytest.raises(LedgerIntegrityError, match="anchor.*HMAC"):
        DaemonLedger(database, key_path=key)


def test_ledger_rejects_replaced_machine_key(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key):
        pass
    key.write_bytes(b"x" * 32)
    key.chmod(0o600)

    with pytest.raises(LedgerIntegrityError, match="HMAC"):
        DaemonLedger(database, key_path=key)


def test_anchor_write_failure_does_not_commit_an_unwitnessed_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:

        def fail_anchor(_sequence: int, _event_mac: str) -> None:
            raise OSError("injected anchor durability failure")

        monkeypatch.setattr(ledger, "_write_anchor", fail_anchor)
        with pytest.raises(OSError, match="anchor durability"):
            ledger.append(
                "PERMIT_CONSUMED",
                entity_kind="PERMIT",
                entity_id="permit-1",
                payload={"action_id": "action-1"},
            )
        with pytest.raises(LedgerError, match="integrity or durable-write"):
            ledger.status()
        with sqlite3.connect(database) as connection:
            event_count = connection.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0]

    assert event_count == 1
    with DaemonLedger(database, key_path=key) as reopened:
        assert reopened.status()["event_count"] == 1
