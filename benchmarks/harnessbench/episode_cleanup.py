"""Private native-episode process cleanup with explicit durable ledger evidence.

This is a cooperative operator helper, not an OS deadline or hardware stop.
Call it on the owning event loop before closing any caller connection. It does
not create/migrate a database or cancel ROS actions. A caller must supply the
fresh episode's declared HOME; socket-safe symlinks may resolve to that HOME.
"""

from __future__ import annotations

import os
import pwd
import sqlite3
import time
from pathlib import Path

from rosclaw.task_kernel.operation_manager import (
    OPERATION_TERMINAL,
    OperationCancellationUnresolvedError,
    OperationManager,
)
from rosclaw.task_kernel.process_identity import ProcessIdentity, session_members
from rosclaw.task_kernel.service import TaskKernel


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _persisted(path: Path, operation_id: str) -> str:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT state FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise ValueError("PRIVATE_OPERATION_DISAPPEARED")
        return str(row[0])
    finally:
        conn.close()


async def cleanup_owned_episode(
    home: Path,
    *,
    declared_home: Path,
    timeout_s: float = 8,
    protected_homes: tuple[Path, ...] = (),
) -> dict:
    """Cancel only proved process operations in this explicitly owned HOME.

    Commit failure, identity mismatch, live members, unsupported providers, or
    elapsed budget leaves cleanup UNCONFIRMED. Physical process stop and durable
    terminal state are separate fields. Existing terminal rows never change.
    ``timeout_s`` is a cooperative qualification budget; close is joined even
    when settlement consumes the remaining budget, and overruns are reported.
    """
    home, declared_home = Path(home), Path(declared_home)
    main_home = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".rosclaw"
    forbidden = (main_home, *protected_homes)
    if home.resolve() != declared_home.resolve() or any(
        home.resolve() == Path(p).resolve() for p in forbidden
    ):
        raise ValueError("OWNED_PRIVATE_HOME_REQUIRED_MAIN_HOME_FORBIDDEN")
    if timeout_s <= 0:
        raise ValueError("POSITIVE_CLEANUP_BUDGET_REQUIRED")
    path = home / "agentd" / "missions.db"
    if not path.exists():
        return {"status": "NO_PRIVATE_LEDGER", "operations": []}
    resolved_db = path.resolve()
    if not resolved_db.is_relative_to(home.resolve()) or any(
        resolved_db.is_relative_to(Path(p).resolve()) for p in forbidden
    ):
        raise ValueError("PRIVATE_LEDGER_PATH_SCOPE_REQUIRED")
    start = time.monotonic()
    conn = _connect(path)
    manager = OperationManager(TaskKernel(conn, home), conn)
    records: list[dict] = []
    close_error = None
    try:
        rows = conn.execute("SELECT * FROM operations").fetchall()
        for row in rows:
            operation_id = str(row["operation_id"])
            record = {"operation_id": operation_id, "before_state": row["state"]}
            members = []
            record["durable_ledger_status"] = "UNCONFIRMED"
            record["process_birth_observation"] = "NOT_CAPTURED"
            record["physical_stop"] = "NOT_PROVEN"
            try:
                identity = ProcessIdentity.parse(row["process_identity_json"] or "")
                current = ProcessIdentity.capture(identity.pid) if identity else None
                if identity and identity.matches():
                    members = session_members(identity)
                    record["process_birth_observation"] = "MATCHED_LIVE_CAPTURED_MEMBERS"
                elif identity and current is None:
                    record["process_birth_observation"] = (
                        "CAPTURED_LEADER_GONE_MEMBERS_NOT_OBSERVED"
                    )
                elif identity:
                    record["process_birth_observation"] = "LIVE_IDENTITY_MISMATCH"
                if row["state"] not in OPERATION_TERMINAL:
                    if row["provider"] != "process":
                        raise ValueError("NON_PROCESS_PROVIDER_STOP_NOT_AUTHORIZED")
                    if time.monotonic() - start >= timeout_s:
                        raise TimeoutError("CLEANUP_QUALIFICATION_BUDGET_EXHAUSTED")
                    await manager.cancel(operation_id, reason="OWNED_EPISODE_END")
                record["observed_state"] = manager.get(operation_id)["state"]
                conn.commit()
                record["persisted_state"] = _persisted(path, operation_id)
                record["durable_ledger_status"] = (
                    "TERMINAL"
                    if (record["persisted_state"] in OPERATION_TERMINAL)
                    else "NONTERMINAL"
                )
                record["status"] = "DURABLE_TERMINAL_PHYSICAL_NOT_PROVEN"
            except Exception as exc:
                # Preserve a truthful pending cancel intent, never fabricate stop.
                if isinstance(exc, OperationCancellationUnresolvedError):
                    try:
                        conn.commit()
                    except Exception:
                        conn.rollback()
                else:
                    conn.rollback()
                record["status"] = "STOP_UNCONFIRMED"
                record["error_class"] = type(exc).__name__
                record["persisted_state"] = _persisted(path, operation_id)
                record["durable_ledger_status"] = (
                    "TERMINAL"
                    if (record["persisted_state"] in OPERATION_TERMINAL)
                    else "NONTERMINAL"
                )
            finally:
                record["remaining_captured_births"] = [
                    member.pid
                    for member in members
                    if member.same_birth(ProcessIdentity.capture(member.pid))
                ]
                if (
                    record["remaining_captured_births"]
                    or record.get("persisted_state") not in OPERATION_TERMINAL
                ):
                    record["status"] = "STOP_UNCONFIRMED"
                record["captured_member_count"] = len(members)
                if members and not record["remaining_captured_births"]:
                    record["physical_stop"] = "CAPTURED_MEMBERS_GONE_ONLY"
                    if record["status"] != "STOP_UNCONFIRMED":
                        record["status"] = "CONFIRMED_CAPTURED_PROCESS_STOP"
                if (
                    row["state"] not in OPERATION_TERMINAL
                    and record["process_birth_observation"] != "MATCHED_LIVE_CAPTURED_MEMBERS"
                ):
                    record["status"] = "STOP_UNCONFIRMED"
                records.append(record)
    finally:
        try:
            await manager.close()
        except Exception as exc:
            close_error = type(exc).__name__
        finally:
            conn.close()
    elapsed = time.monotonic() - start
    good = (
        not close_error
        and elapsed <= timeout_s
        and all(row["status"] != "STOP_UNCONFIRMED" for row in records)
    )
    return {
        "status": (
            "NO_OPERATIONS"
            if good and not records
            else "CONFIRMED_CAPTURED_PROCESS_STOP"
            if good and all(r["physical_stop"] == "CAPTURED_MEMBERS_GONE_ONLY" for r in records)
            else "DURABLE_TERMINAL_PHYSICAL_NOT_PROVEN"
            if good
            else "STOP_UNCONFIRMED"
        ),
        "operations": records,
        "close_error_class": close_error,
        "elapsed_s": elapsed,
        "budget_s": timeout_s,
        "scope": "DURABLE_LEDGER_AND_EXPLICIT_CAPTURED_BIRTH_OBSERVATIONS_NOT_WHOLE_TREE_OR_HARDWARE_STOP",
    }
