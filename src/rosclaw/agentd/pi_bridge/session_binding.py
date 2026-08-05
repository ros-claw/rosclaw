"""SessionBinding 存储与 writer lease（重构规格 §12/§13，PR-PNA-1）。

规则：
- 一个 Pi Session 只能有一个 ACTIVE binding（DB 部分唯一索引）；
- 一个 Mission 同时只有一个 writer lease（唯一索引 + 过期回收）；
- lease 过期即回收（进程崩溃不永久占锁）；
- 所有绑定/lease 操作写审计事件（经调用方 journal）。
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta

from rosclaw.contracts.common import new_id
from rosclaw.contracts.pi.session_binding import PiSessionBindingV1, PiSessionLeaseV1

LEASE_TTL_SEC = 120.0


class BindingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class SessionBindingStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- bindings ---------------------------------------------------------------

    def bind(
        self,
        *,
        pi_session_id: str,
        pi_session_path: str,
        mission_id: str,
        body_id: str,
        execution_mode: str,
        created_by: str,
        parent_binding_id: str | None = None,
        source_mission_id: str | None = None,
    ) -> PiSessionBindingV1:
        if not pi_session_id or not mission_id:
            raise BindingError("INVALID_ARGUMENT", "pi_session_id and mission_id required")
        existing = self.binding_for_session(pi_session_id)
        if existing is not None:
            if existing.mission_id == mission_id:
                return existing
            raise BindingError(
                "SESSION_ALREADY_BOUND",
                f"pi session {pi_session_id} is already bound to "
                f"{existing.mission_id} — detach first (no silent rebind)",
            )
        binding = PiSessionBindingV1(
            binding_id=new_id("psb"),
            pi_session_id=pi_session_id,
            pi_session_path=pi_session_path,
            mission_id=mission_id,
            body_id=body_id,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            created_at=_iso(_now()),
            created_by=created_by,
            parent_binding_id=parent_binding_id,
            source_mission_id=source_mission_id,
        )
        try:
            self._conn.execute(
                "INSERT INTO pi_session_bindings (binding_id, pi_session_id, pi_session_path, "
                "mission_id, body_id, execution_mode, created_at, created_by, "
                "parent_binding_id, source_mission_id, status, binding_revision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    binding.binding_id,
                    binding.pi_session_id,
                    binding.pi_session_path,
                    binding.mission_id,
                    binding.body_id,
                    binding.execution_mode,
                    binding.created_at,
                    binding.created_by,
                    binding.parent_binding_id,
                    binding.source_mission_id,
                    binding.status,
                    binding.binding_revision,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise BindingError(
                "SESSION_ALREADY_BOUND", f"pi session {pi_session_id} already has an active binding"
            ) from exc
        return binding

    def binding_for_session(self, pi_session_id: str) -> PiSessionBindingV1 | None:
        row = self._conn.execute(
            "SELECT * FROM pi_session_bindings WHERE pi_session_id = ? AND status = 'ACTIVE'",
            (pi_session_id,),
        ).fetchone()
        return self._row_to_binding(row) if row else None

    def detach(self, binding_id: str) -> None:
        self._conn.execute(
            "UPDATE pi_session_bindings SET status = 'DETACHED', "
            "binding_revision = binding_revision + 1 WHERE binding_id = ?",
            (binding_id,),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_binding(row: sqlite3.Row) -> PiSessionBindingV1:
        return PiSessionBindingV1(
            binding_id=row["binding_id"],
            pi_session_id=row["pi_session_id"],
            pi_session_path=row["pi_session_path"],
            mission_id=row["mission_id"],
            body_id=row["body_id"],
            execution_mode=row["execution_mode"],
            created_at=row["created_at"],
            created_by=row["created_by"],
            parent_binding_id=row["parent_binding_id"],
            source_mission_id=row["source_mission_id"],
            status=row["status"],
            binding_revision=row["binding_revision"],
        )

    # -- writer lease -------------------------------------------------------------

    def acquire_lease(
        self,
        *,
        mission_id: str,
        pi_session_id: str,
        owner_pid: int,
        owner_uid: int,
        ttl_sec: float = LEASE_TTL_SEC,
    ) -> tuple[PiSessionLeaseV1, str]:
        """获取 writer lease；返回 (lease, token)。token 只返回一次（库里存 hash）。

        过期 lease 先回收；未过期 lease 属他人 → WRITER_HELD（fail closed）。
        """
        now = _now()
        with self._conn:  # transaction
            self._conn.execute(
                "DELETE FROM pi_session_leases WHERE mission_id = ? AND expires_at < ?",
                (mission_id, _iso(now)),
            )
            held = self._conn.execute(
                "SELECT * FROM pi_session_leases WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()
            if held is not None and held["pi_session_id"] != pi_session_id:
                raise BindingError(
                    "WRITER_HELD",
                    f"mission {mission_id} writer lease held by session "
                    f"{held['pi_session_id']} (pid {held['owner_pid']}) until {held['expires_at']}",
                )
            token = secrets.token_urlsafe(32)
            lease = PiSessionLeaseV1(
                lease_id=new_id("psl"),
                mission_id=mission_id,
                pi_session_id=pi_session_id,
                owner_pid=owner_pid,
                owner_uid=owner_uid,
                lease_token_hash=hashlib.sha256(token.encode()).hexdigest(),
                issued_at=_iso(now),
                expires_at=_iso(now + timedelta(seconds=ttl_sec)),
                heartbeat_at=_iso(now),
            )
            self._conn.execute("DELETE FROM pi_session_leases WHERE mission_id = ?", (mission_id,))
            self._conn.execute(
                "INSERT INTO pi_session_leases (lease_id, mission_id, pi_session_id, owner_pid, "
                "owner_uid, host_id, lease_token_hash, issued_at, expires_at, heartbeat_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    lease.lease_id,
                    lease.mission_id,
                    lease.pi_session_id,
                    lease.owner_pid,
                    lease.owner_uid,
                    lease.host_id,
                    lease.lease_token_hash,
                    lease.issued_at,
                    lease.expires_at,
                    lease.heartbeat_at,
                ),
            )
        return lease, token

    def heartbeat_lease(self, mission_id: str, pi_session_id: str, token: str) -> PiSessionLeaseV1:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT * FROM pi_session_leases WHERE mission_id = ? AND pi_session_id = ?",
            (mission_id, pi_session_id),
        ).fetchone()
        if row is None:
            raise BindingError("LEASE_NOT_FOUND", "no active lease for this session/mission")
        if row["lease_token_hash"] != token_hash:
            raise BindingError("LEASE_TOKEN_MISMATCH", "lease token does not match")
        if row["expires_at"] < _iso(_now()):
            raise BindingError("LEASE_EXPIRED", "lease expired — re-acquire")
        now = _now()
        expires = _iso(now + timedelta(seconds=LEASE_TTL_SEC))
        self._conn.execute(
            "UPDATE pi_session_leases SET heartbeat_at = ?, expires_at = ? WHERE lease_id = ?",
            (_iso(now), expires, row["lease_id"]),
        )
        self._conn.commit()
        row_keys = set(row.keys())
        return PiSessionLeaseV1(
            **{k: row[k] for k in row_keys if k not in ("heartbeat_at", "expires_at")},
            heartbeat_at=_iso(now),
            expires_at=expires,
        )

    def release_lease(self, mission_id: str, pi_session_id: str, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cursor = self._conn.execute(
            "DELETE FROM pi_session_leases WHERE mission_id = ? AND pi_session_id = ? "
            "AND lease_token_hash = ?",
            (mission_id, pi_session_id, token_hash),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def writer_of(self, mission_id: str) -> PiSessionLeaseV1 | None:
        row = self._conn.execute(
            "SELECT * FROM pi_session_leases WHERE mission_id = ?", (mission_id,)
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < _iso(_now()):
            return None
        return PiSessionLeaseV1(**{k: row[k] for k in set(row.keys())})
