"""Role leases with epoch + conflict_key CAS (PR-TF-072, 总纲 §10.4).

Roles are leases scoped to a team epoch. The DB enforces at most one
ACTIVE lease per (team, epoch, conflict_key). Conflicting views (two
robots believing they hold the same role) resolve conservatively: contest
both, stop the contested resource, request re-coordination.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.team.role import RoleLeaseV1


class RoleConflictError(ValidationError):
    """An ACTIVE lease for this conflict_key already exists in this epoch."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class RoleLeaseStore:
    def __init__(self, conn: sqlite3.Connection, *, team_id: str) -> None:
        self._conn = conn
        self._team_id = team_id

    def award(self, lease: RoleLeaseV1) -> RoleLeaseV1:
        """CAS-award: fails if an ACTIVE lease exists for the conflict_key
        in this epoch, or if the lease references an older epoch."""
        current_epoch = self._current_epoch()
        if lease.team_epoch != current_epoch:
            raise ValidationError(
                f"lease epoch {lease.team_epoch} != current {current_epoch} — "
                "old-epoch awards are rejected"
            )
        if lease.holder and not self._member_ready(lease.holder):
            raise ValidationError(f"holder {lease.holder!r} is not READY")
        try:
            self._conn.execute(
                "INSERT INTO role_leases (lease_id, team_id, team_epoch, "
                "conflict_key, holder, lease_json, state, expires_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)",
                (
                    new_id("rlease"),
                    self._team_id,
                    lease.team_epoch,
                    lease.conflict_key,
                    lease.holder,
                    lease.model_dump_json(),
                    lease.expires_at,
                    _utcnow(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RoleConflictError(
                f"conflict_key {lease.conflict_key!r} already has an ACTIVE "
                f"lease in epoch {lease.team_epoch}"
            ) from exc
        return lease

    def renew(self, conflict_key: str, holder: str, *, new_expires_at: str) -> None:
        row = self._active(conflict_key)
        if row is None or row["holder"] != holder:
            raise ValidationError(f"no ACTIVE lease for {conflict_key!r} held by {holder!r}")
        lease = RoleLeaseV1(**json.loads(row["lease_json"]))
        lease = lease.model_copy(update={"expires_at": new_expires_at})
        self._conn.execute(
            "UPDATE role_leases SET lease_json = ?, expires_at = ?, updated_at = ? "
            "WHERE lease_id = ?",
            (lease.model_dump_json(), new_expires_at, _utcnow(), row["lease_id"]),
        )

    def revoke(self, conflict_key: str, *, reason: str) -> None:
        row = self._active(conflict_key)
        if row is None:
            return
        self._conn.execute(
            "UPDATE role_leases SET state = 'REVOKED', updated_at = ? WHERE lease_id = ?",
            (_utcnow(), row["lease_id"]),
        )

    def sweep_expired(self) -> list[str]:
        """Expire ACTIVE leases past expiry; also invalidate non-current
        epochs. Returns conflict_keys that became free."""
        now = _utcnow()
        epoch = self._current_epoch()
        rows = self._conn.execute(
            "SELECT lease_id, conflict_key, team_epoch, expires_at FROM role_leases "
            "WHERE team_id = ? AND state = 'ACTIVE'",
            (self._team_id,),
        ).fetchall()
        freed: list[str] = []
        for row in rows:
            if row["expires_at"] < now or int(row["team_epoch"]) != epoch:
                self._conn.execute(
                    "UPDATE role_leases SET state = 'EXPIRED', updated_at = ? WHERE lease_id = ?",
                    (now, row["lease_id"]),
                )
                freed.append(row["conflict_key"])
        return freed

    def contest(self, conflict_key: str) -> None:
        """Conservative conflict policy: mark the ACTIVE lease CONTESTED —
        both parties must stop contesting the resource and re-coordinate."""
        row = self._active(conflict_key)
        if row is not None:
            self._conn.execute(
                "UPDATE role_leases SET state = 'CONTESTED', updated_at = ? WHERE lease_id = ?",
                (_utcnow(), row["lease_id"]),
            )

    def active_holder(self, conflict_key: str) -> str | None:
        row = self._active(conflict_key)
        return row["holder"] if row else None

    def active_leases(self) -> list[RoleLeaseV1]:
        rows = self._conn.execute(
            "SELECT lease_json FROM role_leases WHERE team_id = ? AND state = 'ACTIVE'",
            (self._team_id,),
        ).fetchall()
        return [RoleLeaseV1(**json.loads(r["lease_json"])) for r in rows]

    # ------------------------------------------------------------------
    def _active(self, conflict_key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM role_leases WHERE team_id = ? AND conflict_key = ? AND state = 'ACTIVE'",
            (self._team_id, conflict_key),
        ).fetchone()

    def _current_epoch(self) -> int:
        row = self._conn.execute(
            "SELECT MAX(epoch) AS e FROM team_epochs WHERE team_id = ?", (self._team_id,)
        ).fetchone()
        return int(row["e"] or 0)

    def _member_ready(self, member_id: str) -> bool:
        row = self._conn.execute(
            "SELECT state FROM team_members WHERE team_id = ? AND member_id = ?",
            (self._team_id, member_id),
        ).fetchone()
        return row is not None and row["state"] == "READY"
